"""Paired images and masks with explicit label encodings and shared preprocessing."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

CLASS_NAMES = ("void", "flat", "construction", "object", "nature", "sky", "human", "vehicle")
PALETTE = np.array(
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
IGNORE_INDEX = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ENCODINGS = ("category_rgb", "category_ids", "cityscapes_ids")

# Original IDs collapsed to seven categories; ignoreInEval IDs stay ignored.
# Source: official cityscapesscripts/helpers/labels.py, linked in README.
CITYSCAPES_TO_CATEGORY = {
    7: 1,
    8: 1,
    11: 2,
    12: 2,
    13: 2,
    17: 3,
    19: 3,
    20: 3,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 7,
    27: 7,
    28: 7,
    31: 7,
    32: 7,
    33: 7,
}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    key: str
    group: str


def sequence_group(key):
    parts = Path(key).stem.split("_")
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        return "_".join(parts[:2])
    return None


def discover_samples(root, encoding="category_rgb", split=None):
    """Read img/mask or official leftImg8bit/gtFine trees; never download."""
    root = Path(root).resolve()
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding: {encoding}")
    if (root / "data_seg_public").is_dir():
        root = root / "data_seg_public"
    if (root / "leftImg8bit").is_dir():
        images, masks = root / "leftImg8bit", root / "gtFine"
    else:
        images, masks = root / "img", root / "mask"
    if split:
        images, masks = images / split, masks / split
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Expected paired directories: {images}, {masks}")
    samples = []
    for image in sorted(images.rglob("*.png")):
        rel = image.relative_to(images)
        if image.stem.endswith("_leftImg8bit"):
            suffixes = {"cityscapes_ids": "labelIds", "category_ids": "categoryIds", "category_rgb": "color"}
            name = image.stem.removesuffix("_leftImg8bit") + f"_gtFine_{suffixes[encoding]}.png"
            mask = masks / rel.parent / name
        else:
            mask = masks / rel
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask for {image}: {mask}")
        group = sequence_group(rel) or str(rel.parent)
        samples.append(Sample(image, mask, rel.as_posix(), group))
    if not samples:
        raise ValueError(f"No PNG image pairs found in {images}")
    return samples


def split_samples(samples, val_fraction=0.2, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = sorted({s.group for s in samples})
    if len(groups) < 2:
        raise ValueError("Need two sequence/directory groups; alternatively provide train/val folders")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    validation = set(groups[:count])
    return ([s for s in samples if s.group not in validation], [s for s in samples if s.group in validation])


def decode_mask(mask, encoding, ignore_void=True):
    """Return HxW class IDs. Unknown input is an error, never silently void."""
    array = np.asarray(mask)
    if encoding == "category_rgb":
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("category_rgb requires an RGB/RGBA mask")
        rgb = array[..., :3]
        labels = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        known = np.zeros(rgb.shape[:2], dtype=bool)
        for index, color in enumerate(PALETTE):
            match = (rgb == color).all(axis=-1)
            labels[match] = index
            known |= match
        if not known.all():
            colors = np.unique(rgb[~known], axis=0)[:5].tolist()
            raise ValueError(f"Unknown category colors {colors}; original Cityscapes needs cityscapes_ids")
    elif encoding in ("category_ids", "cityscapes_ids"):
        if array.ndim != 2:
            raise ValueError(f"{encoding} requires a single-channel ID mask")
        if encoding == "category_ids":
            if not np.isin(array, [*range(8), IGNORE_INDEX]).all():
                raise ValueError("Category IDs must be 0..7 or 255 (ignore)")
            labels = array.astype(np.uint8).copy()
        else:
            if not np.isin(array, [*range(34), IGNORE_INDEX]).all():
                raise ValueError("Expected original Cityscapes label IDs 0..33 or 255")
            labels = np.full(array.shape, IGNORE_INDEX, dtype=np.uint8)
            for label_id, category in CITYSCAPES_TO_CATEGORY.items():
                labels[array == label_id] = category
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    if ignore_void:
        labels[labels == 0] = IGNORE_INDEX
    return labels


def image_tensor(image, size):
    """size is (height, width); inference and training share normalization."""
    image = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class SegmentationDataset(Dataset):
    def __init__(self, samples, size=(256, 512), encoding="category_rgb", augment=False, ignore_void=True):
        self.samples = list(samples)
        self.size = tuple(size)
        self.encoding = encoding
        self.augment = augment
        self.ignore_void = ignore_void

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            mask = source.convert("RGB") if self.encoding == "category_rgb" else source.copy()
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch for {sample.key}")
        # Decode before resize so invalid colors cannot disappear through downsampling.
        mask = Image.fromarray(decode_mask(mask, self.encoding, self.ignore_void))
        if self.augment:
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        mask = mask.resize((self.size[1], self.size[0]), Image.Resampling.NEAREST)
        return image_tensor(image, self.size), torch.from_numpy(np.array(mask, dtype=np.int64))
