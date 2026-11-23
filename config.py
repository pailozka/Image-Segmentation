"""Serializable experiment configuration, with validation before allocating a model."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dataset import ENCODINGS
from network import ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    image_size: tuple[int, int] = (256, 512)
    encoding: str = "category_rgb"
    ignore_void: bool = True
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    dice_weight: float = 0.5
    grad_clip: float = 1.0
    patience: int = 10
    min_delta: float = 0.0001
    val_fraction: float = 0.2
    seed: int = 42
    augment: bool = True
    amp: bool = True
    deterministic: bool = True

    def __post_init__(self):
        integers = (self.epochs, self.batch_size, self.patience, self.seed, *self.image_size)
        if any(type(value) is not int for value in integers):
            raise ValueError("Epochs, batch size, patience, seed and image dimensions must be integers")
        scalars = (
            self.learning_rate,
            self.weight_decay,
            self.dice_weight,
            self.grad_clip,
            self.min_delta,
            self.val_fraction,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Configuration values must be finite")
        if self.model.in_channels != 3 or self.model.num_classes != 8:
            raise ValueError("This dataset pipeline requires RGB input and eight category outputs")
        if len(self.image_size) != 2 or min(self.image_size) < 2**self.model.depth:
            raise ValueError("image_size must contain H,W >= 2**depth")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.dice_weight <= 1 or not 0 < self.val_fraction < 1 or self.min_delta < 0:
            raise ValueError("Invalid loss/split/early stopping settings")
        if self.seed < 0 or self.seed + self.epochs >= 2**32:
            raise ValueError("seed + epochs must be in [0, 2**32)")
        if self.encoding == "cityscapes_ids" and not self.ignore_void:
            raise ValueError("cityscapes_ids always ignores official void labels; use ignore_void=true")

    def to_dict(self):
        # Normalize tuples into JSON arrays for checkpoint/config equality.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["model"] = ModelConfig(**values.get("model", {}))
        values["image_size"] = tuple(values.get("image_size", (256, 512)))
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
