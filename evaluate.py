"""Evaluate saved validation samples or an explicitly named labeled split."""

import argparse
from pathlib import Path

from config import TrainConfig
from dataset import discover_samples
from losses import CrossEntropyDiceLoss
from training import make_loader, prepare_splits, run_epoch
from utils import load_checkpoint, resolve_device, sample_manifest, write_json


def evaluate(dataset, checkpoint_path, output="evaluation.json", split=None, device="auto", workers=0):
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = TrainConfig.from_dict(checkpoint["config"])
    if split:
        samples = discover_samples(dataset, config.encoding, split)
        manifest = sample_manifest(samples)
    else:
        _, samples, splits = prepare_splits(dataset, config)
        if splits != checkpoint["splits"]:
            raise ValueError(
                "Dataset differs from the checkpoint; use --split to evaluate a new labeled split"
            )
        manifest = splits["val"]
    metrics = run_epoch(
        model,
        make_loader(samples, config, workers, device=device),
        CrossEntropyDiceLoss(config.dice_weight),
        device,
        config,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "split": split or "saved_validation",
        "image_size": list(config.image_size),
        "encoding": config.encoding,
        "ignore_void": config.ignore_void,
        "samples": manifest,
        "metrics": metrics,
    }
    write_json(output, report)
    print(f"mIoU={metrics['mean_iou']:.4f} Dice={metrics['mean_dice']:.4f}; report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument(
        "--split", help="Explicit labeled split, e.g. val or holdout (otherwise use saved validation)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.dataset, args.checkpoint, args.output, args.split, args.device, args.workers)


if __name__ == "__main__":
    main()
