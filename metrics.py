"""Dataset-level confusion matrix (rows=true, columns=predicted)."""

import torch

from dataset import CLASS_NAMES, IGNORE_INDEX


class SegmentationMetrics:
    def __init__(self, num_classes=8, excluded_classes=(0,), device="cpu"):
        self.num_classes = num_classes
        self.excluded_classes = set(excluded_classes)
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        if prediction.ndim == target.ndim + 1:
            prediction = prediction.argmax(dim=1)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        valid = target != IGNORE_INDEX
        true, pred = target[valid].long(), prediction[valid].long()
        if ((true < 0) | (true >= self.num_classes) | (pred < 0) | (pred >= self.num_classes)).any():
            raise ValueError("Out-of-range class ID in metrics")
        counts = torch.bincount(self.num_classes * true + pred, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes).to(self.matrix.device)

    def compute(self):
        matrix = self.matrix.cpu().double()
        tp = matrix.diag()
        support, predicted = matrix.sum(1), matrix.sum(0)
        union = support + predicted - tp
        included = [i for i in range(self.num_classes) if i not in self.excluded_classes and union[i] > 0]
        iou = tp / union.clamp_min(1)
        dice = 2 * tp / (support + predicted).clamp_min(1)
        per_class = {}
        for i in range(self.num_classes):
            name = CLASS_NAMES[i] if self.num_classes == len(CLASS_NAMES) else str(i)
            per_class[name] = {
                "iou": float(iou[i]) if union[i] > 0 else None,
                "dice": float(dice[i]) if union[i] > 0 else None,
                "support": int(support[i]),
                "included_in_mean": i in included,
            }
        total = int(matrix.sum())
        return {
            "mean_iou": float(iou[included].mean()) if included else None,
            "mean_dice": float(dice[included].mean()) if included else None,
            "pixel_accuracy": float(tp.sum() / total) if total else None,
            "valid_pixels": total,
            "per_class": per_class,
            "confusion_matrix": self.matrix.cpu().tolist(),
        }
