"""Predict a file or image tree without requiring ground-truth masks."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from config import TrainConfig
from dataset import PALETTE, image_tensor
from utils import load_checkpoint, resolve_device, write_json


@torch.inference_mode()
def predict_image(model, image, size, device):
    tensor = image_tensor(image, size).unsqueeze(0).to(device)
    logits = model(tensor)
    # Resize logits before argmax to retain a smooth decision boundary.
    logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def inference(dataset_path, model_path, n_samples=0, output="output_predictions", device="auto"):
    device = resolve_device(device)
    model, checkpoint = load_checkpoint(model_path, device)
    config = TrainConfig.from_dict(checkpoint["config"])
    model.eval()
    source, output = Path(dataset_path), Path(output)
    if source.is_file():
        images, root = [source], source.parent
    elif source.is_dir():
        root = source
        for folder in ("data_seg_public/img", "img", "leftImg8bit"):
            if (source / folder).is_dir():
                root = source / folder
                break
        images = sorted(
            p
            for p in root.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and not p.resolve().is_relative_to(output.resolve())
        )
    else:
        raise FileNotFoundError(source)
    if n_samples > 0:
        images = images[:n_samples]
    if not images:
        raise ValueError(f"No images found in {source}")
    records = []
    for path in images:
        with Image.open(path) as stream:
            image = stream.convert("RGB")
        prediction = predict_image(model, image, config.image_size, device)
        relative = path.relative_to(root)
        target = output / relative.parent
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(target / f"{relative.name}.ids.png")
        colored = Image.fromarray(PALETTE[prediction])
        colored.save(target / f"{relative.name}.color.png")
        Image.blend(image, colored, 0.45).save(target / f"{relative.name}.overlay.png")
        records.append({"image": relative.as_posix(), "width": image.width, "height": image.height})
    write_json(
        output / "predictions.json", {"checkpoint": str(model_path), "count": len(records), "images": records}
    )
    print(f"Saved {len(records)} predictions to {output}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("n_samples", type=int, nargs="?", default=0)
    parser.add_argument("--output", default="output_predictions", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    inference(args.input, args.checkpoint, args.n_samples, args.output, args.device)


if __name__ == "__main__":
    main()
