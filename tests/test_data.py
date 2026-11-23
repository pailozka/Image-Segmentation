from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from config import TrainConfig
from dataset import (
    IGNORE_INDEX,
    PALETTE,
    Sample,
    SegmentationDataset,
    decode_mask,
    discover_samples,
    split_samples,
)
from training import prepare_splits


def test_category_palette_and_void_policy():
    mask = PALETTE[np.arange(8).reshape(2, 4)]
    assert np.array_equal(decode_mask(mask, "category_rgb", False), np.arange(8).reshape(2, 4))
    assert decode_mask(mask, "category_rgb")[0, 0] == IGNORE_INDEX


def test_unknown_color_is_not_silently_void():
    with pytest.raises(ValueError, match="Unknown category colors"):
        decode_mask(np.array([[[1, 2, 3]]], dtype=np.uint8), "category_rgb")


def test_original_cityscapes_ids_are_collapsed_and_ignored():
    raw = np.array([[0, 7, 8, 11, 17, 18, 21, 23, 24, 26, 29, 33, 255]], dtype=np.uint8)
    expected = [[255, 1, 1, 2, 3, 255, 4, 5, 6, 7, 255, 7, 255]]
    assert decode_mask(raw, "cityscapes_ids").tolist() == expected
    with pytest.raises(ValueError, match="original Cityscapes"):
        decode_mask(np.array([[34]], dtype=np.uint8), "cityscapes_ids")


def test_category_ids_and_invalid_ids():
    assert decode_mask(np.array([[0, 1, 255]], dtype=np.uint8), "category_ids").tolist() == [[255, 1, 255]]
    with pytest.raises(ValueError, match="0..7"):
        decode_mask(np.array([[8]], dtype=np.uint8), "category_ids")


def test_sequence_split_is_disjoint_and_reproducible():
    samples = [
        Sample(Path(f"{g}_{i}.png"), Path("mask.png"), f"{g}/{i}", str(g)) for g in range(5) for i in range(3)
    ]
    train, val = split_samples(samples, seed=42)
    assert not {s.group for s in train} & {s.group for s in val}
    assert len(train) + len(val) == len(samples)
    assert (train, val) == split_samples(samples, seed=42)
    with pytest.raises(ValueError, match="two sequence"):
        split_samples(samples[:3])


def test_discovery_and_nearest_mask_resize(demo_data):
    samples = discover_samples(demo_data, split="train")
    image, mask = SegmentationDataset(samples, size=(19, 37))[0]
    assert image.shape == (3, 19, 37)
    assert mask.shape == (19, 37)
    assert image.dtype == torch.float32 and mask.dtype == torch.int64
    assert set(mask.unique().tolist()) <= set(range(1, 8))
    samples[0].mask.unlink()
    with pytest.raises(FileNotFoundError, match="Missing mask"):
        discover_samples(demo_data, split="train")


def test_mismatched_image_mask_size_fails(demo_data):
    samples = discover_samples(demo_data, split="train")
    Image.new("RGB", (3, 3)).save(samples[0].image)
    with pytest.raises(ValueError, match="size mismatch"):
        SegmentationDataset(samples)[0]


def test_duplicate_content_cannot_leak_into_validation(demo_data):
    train = discover_samples(demo_data, split="train")
    val = discover_samples(demo_data, split="val")
    val[0].image.write_bytes(train[0].image.read_bytes())
    with pytest.raises(ValueError, match="Identical image content"):
        prepare_splits(demo_data, TrainConfig())


def test_original_cityscapes_directory_pairing(tmp_path):
    image = tmp_path / "leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png"
    mask = tmp_path / "gtFine/train/aachen/aachen_000001_000019_gtFine_labelIds.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(image)
    Image.fromarray(np.full((16, 16), 7, dtype=np.uint8)).save(mask)
    samples = discover_samples(tmp_path, "cityscapes_ids", "train")
    assert samples[0].mask == mask
    assert samples[0].group == "aachen_000001"
    assert SegmentationDataset(samples, (16, 16), "cityscapes_ids")[0][1].unique().tolist() == [1]


def test_explicit_flat_splits_do_not_need_sequence_names(demo_data):
    for split in ("train", "val"):
        for directory in ("img", "mask"):
            for index, path in enumerate(sorted((demo_data / directory / split).glob("*.png"))):
                path.rename(path.with_name(f"image{index}.png"))
    train, val, _ = prepare_splits(demo_data, TrainConfig())
    assert len(train) == 4 and len(val) == 2
