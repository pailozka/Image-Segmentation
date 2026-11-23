"""Multiclass cross entropy plus soft Dice, with consistent ignored pixels."""

from torch import nn
from torch.nn import functional as F

from dataset import IGNORE_INDEX


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ignore_index=IGNORE_INDEX):
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must be in [0, 1]")
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits = logits.float()
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        if self.dice_weight == 0:
            return ce
        truth = F.one_hot(target.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2).float()
        truth = truth * valid.unsqueeze(1)
        probability = logits.softmax(dim=1) * valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (truth * probability).sum(dims)
        denominator = truth.sum(dims) + probability.sum(dims)
        # Absent classes are handled by CE; Dice averages target-present classes.
        present = truth.sum(dims) > 0
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        return (1 - self.dice_weight) * ce + self.dice_weight * (1 - dice[present].mean())
