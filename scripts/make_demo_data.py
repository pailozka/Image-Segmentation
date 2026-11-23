"""Create synthetic category masks for pipeline checks, never for benchmark claims."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_demo_data(root, count=8, size=(32, 64), seed=42):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {root}")
    if count < 2 or min(size) < 8:
        raise ValueError("Use at least two samples and sides >= 8")
    rng = np.random.default_rng(seed)
    palette = np.array(
        [
            (0, 0, 0),
            (128, 64, 128),
            (70, 70, 70),
            (153, 153, 153),
            (107, 142, 35),
            (70, 130, 180),
            (220, 20, 60),
            (0, 0, 142),
        ],
        dtype=np.uint8,
    )
    for split, number in (("train", count), ("val", max(2, count // 4))):
        for directory in ("img", "mask"):
            (root / directory / split).mkdir(parents=True, exist_ok=True)
        for index in range(number):
            y, x = np.indices(size)
            labels = ((x // max(1, size[1] // 7) + index + y // 8) % 7 + 1).astype(np.uint8)
            rgb = np.clip(
                palette[labels].astype(np.int16) + rng.integers(-15, 16, (*size, 3)), 0, 255
            ).astype(np.uint8)
            filename = f"{split}_{index:06d}_000000.png"
            Image.fromarray(rgb).save(root / "img" / split / filename)
            Image.fromarray(palette[labels]).save(root / "mask" / split / filename)
    (root / "SYNTHETIC.txt").write_text(
        "Generated fixture data. Not Cityscapes. Not a performance benchmark.\n"
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    make_demo_data(args.output, args.count)
