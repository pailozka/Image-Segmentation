"""Configurable U-Net family returning full-resolution raw logits."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ("unet", "residual_unet", "attention_unet", "residual_attention_unet")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "residual_attention_unet"
    in_channels: int = 3
    num_classes: int = 8
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.in_channels, self.num_classes, self.base_channels, self.depth)
        ):
            raise ValueError("Channel counts and depth must be integers")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if self.in_channels < 1 or self.num_classes < 2 or self.base_channels < 2:
            raise ValueError("Invalid input, class, or base channel count")
        if not 1 <= self.depth <= 5 or not 0 <= self.dropout < 1:
            raise ValueError("depth must be 1..5 and dropout must be in [0, 1)")


def normalization(channels):
    # At least two channels per group, including at a 1x1 bottleneck.
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0 and channels // g >= 2)
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            normalization(out_channels),
        )
        self.residual = residual
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        out = self.layers(x)
        return F.relu(out + self.shortcut(x) if self.residual else out)


class AttentionGate(nn.Module):
    """Use decoder context to learn spatial weights for skip features."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.skip = nn.Conv2d(channels, hidden, 1)
        self.gate = nn.Conv2d(channels, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, context):
        weights = torch.sigmoid(self.score(F.relu(self.skip(skip) + self.gate(context))))
        return skip * weights


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual, attention):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.attention = AttentionGate(out_channels) if attention else None
        self.fuse = ConvBlock(2 * out_channels, out_channels, residual)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.project(x)
        if self.attention is not None:
            skip = self.attention(skip, x)
        return self.fuse(torch.cat((x, skip), dim=1))


class UNet(nn.Module):
    """Normalized baseline or residual/attention ablation; supports odd H/W."""

    def __init__(self, config=None, *, num_class=None):
        super().__init__()
        self.config = config or ModelConfig(architecture="unet", num_classes=num_class or 8)
        if config is not None and num_class is not None:
            raise ValueError("Pass config or num_class, not both")
        c = self.config
        residual, attention = "residual" in c.architecture, "attention" in c.architecture
        widths = [c.base_channels * 2**i for i in range(c.depth + 1)]
        self.encoder = nn.ModuleList(
            ConvBlock(a, b, residual) for a, b in zip([c.in_channels, *widths[:-1]], widths)
        )
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(c.dropout)
        self.decoder = nn.ModuleList(
            UpBlock(widths[i], widths[i - 1], residual, attention) for i in range(c.depth, 0, -1)
        )
        self.head = nn.Conv2d(widths[0], c.num_classes, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected N,{self.config.in_channels},H,W input")
        if min(x.shape[-2:]) < 2**self.config.depth:
            raise ValueError(f"Input sides must be at least {2**self.config.depth} pixels")
        skips = []
        for block in self.encoder[:-1]:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.dropout(self.encoder[-1](x))
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        return self.head(x)


def build_model(config):
    return UNet(ModelConfig(**config) if isinstance(config, dict) else config)


if __name__ == "__main__":
    for name in ARCHITECTURES:
        model = build_model(ModelConfig(architecture=name))
        print(name, sum(p.numel() for p in model.parameters()), "parameters")
