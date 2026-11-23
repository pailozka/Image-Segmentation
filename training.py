"""Reproducible training: python training.py DATA --config configs/attention.json."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import IGNORE_INDEX, SegmentationDataset, discover_samples, sequence_group, split_samples
from losses import CrossEntropyDiceLoss
from metrics import SegmentationMetrics
from network import build_model
from utils import (
    environment_info,
    load_checkpoint,
    resolve_device,
    sample_manifest,
    save_checkpoint,
    seed_everything,
    seed_worker,
    write_json,
)


def prepare_splits(root, config):
    root = Path(root)
    if (root / "data_seg_public").is_dir():
        root = root / "data_seg_public"
    image_root = root / ("leftImg8bit" if (root / "leftImg8bit").is_dir() else "img")
    if (image_root / "train").is_dir() or (image_root / "val").is_dir():
        train = discover_samples(root, config.encoding, "train")
        val = discover_samples(root, config.encoding, "val")
        train_sequences = {sequence_group(s.key) for s in train} - {None}
        val_sequences = {sequence_group(s.key) for s in val} - {None}
        if train_sequences & val_sequences:
            raise ValueError("Sequence groups overlap between train and validation")
    else:
        # Never silently mix an official test split into a random training split.
        if (image_root / "test").exists():
            raise ValueError("A test directory exists but train/val directories are missing")
        train, val = split_samples(discover_samples(root, config.encoding), config.val_fraction, config.seed)
    manifest = {"train": sample_manifest(train), "val": sample_manifest(val)}
    if {s["image_sha256"] for s in manifest["train"]} & {s["image_sha256"] for s in manifest["val"]}:
        raise ValueError("Identical image content occurs in both training and validation")
    return train, val, manifest


def make_loader(samples, config, workers, training=False, epoch=0, device="cpu"):
    dataset = SegmentationDataset(
        samples,
        config.image_size,
        config.encoding,
        augment=training and config.augment,
        ignore_void=config.ignore_void,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed + epoch),
    )


def run_epoch(model, loader, criterion, device, config, optimizer=None, scaler=None):
    is_training = optimizer is not None
    model.train(is_training)
    metrics = SegmentationMetrics(excluded_classes=(0,) if config.ignore_void else (), device=device)
    total_loss, images = 0.0, 0
    with torch.set_grad_enabled(is_training):
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            # Skip all-ignored batches, including the optimizer's weight decay.
            if not (target != IGNORE_INDEX).any():
                continue
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check data and optimizer settings")
            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(image)
            images += len(image)
            metrics.update(logits.detach(), target)
    result = metrics.compute()
    if not images or result["mean_iou"] is None:
        raise ValueError("No evaluable pixels; check mask encoding and ignore policy")
    result["loss"] = total_loss / images
    return result


def write_history(output, history):
    write_json(output / "history.json", history)
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "learning_rate", "train_loss", "val_loss", "val_mean_iou", "val_mean_dice"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train_loss": row["train"]["loss"],
                    "val_loss": row["val"]["loss"],
                    "val_mean_iou": row["val"]["mean_iou"],
                    "val_mean_dice": row["val"]["mean_dice"],
                }
            )


def training(dataset_path, config=None, output="runs/default", device="auto", workers=0, resume=None):
    config = config or TrainConfig()
    output, device = Path(output), resolve_device(device)
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is not empty: {output}; use a new output or --resume")
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume from this run's last.pt and use the same --output directory")
    seed_everything(config.seed, config.deterministic)
    train, val, manifest = prepare_splits(dataset_path, config)
    checkpoint = None
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        if checkpoint["config"] != config.to_dict() or checkpoint["splits"] != manifest:
            raise ValueError("Resume requires identical configuration and dataset contents/splits")
        if checkpoint["workers"] != workers or checkpoint["environment"]["device"] != str(device):
            raise ValueError("Resume requires the original worker count and device")
        if not (output / "best.pt").is_file():
            raise FileNotFoundError("Resume requires best.pt alongside last.pt")
    else:
        model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    criterion = CrossEntropyDiceLoss(config.dice_weight)
    start_epoch, best_iou, stale, history = 0, -1.0, 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou, stale, history = checkpoint["best_iou"], checkpoint["stale_epochs"], checkpoint["history"]
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_info(device)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "splits.json", manifest)
    write_json(output / "environment.json", environment)
    print(
        f"{device}: {len(train)} train / {len(val)} validation; {sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )
    if stale >= config.patience:
        print("This run already met the early-stopping condition.")
    for epoch in range(start_epoch, config.epochs):
        if stale >= config.patience:
            break
        # Epoch-derived RNG streams make restart at epoch boundaries reproducible,
        # including augmentation, shuffling, dropout and worker process seeds.
        seed_everything(config.seed + epoch, config.deterministic)
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_result = run_epoch(
            model,
            make_loader(train, config, workers, True, epoch, device),
            criterion,
            device,
            config,
            optimizer,
            scaler,
        )
        val_result = run_epoch(
            model, make_loader(val, config, workers, device=device), criterion, device, config
        )
        scheduler.step()
        score = val_result["mean_iou"]
        improved = score > best_iou
        significant = score > best_iou + config.min_delta
        stale = 0 if significant else stale + 1
        if improved:
            best_iou = score
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "seconds": time.perf_counter() - started,
                "train": train_result,
                "val": val_result,
            }
        )
        payload = {
            "format_version": 1,
            "config": config.to_dict(),
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_iou": best_iou,
            "stale_epochs": stale,
            "history": history,
            "splits": manifest,
            "environment": environment,
            "workers": workers,
        }
        if improved:
            save_checkpoint(output / "best.pt", payload)
            write_json(output / "best_metrics.json", {"epoch": epoch + 1, **val_result})
        save_checkpoint(output / "last.pt", payload)
        write_history(output, history)
        print(
            f"Epoch {epoch + 1}/{config.epochs} loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} mIoU={score:.4f}",
            flush=True,
        )
    write_history(output, history)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.config:
        config = TrainConfig.load(args.config)
    elif args.resume:
        _, checkpoint = load_checkpoint(args.resume)
        config = TrainConfig.from_dict(checkpoint["config"])
    else:
        config = TrainConfig()
    training(args.dataset, config, args.output, args.device, args.workers, args.resume)


if __name__ == "__main__":
    main()
