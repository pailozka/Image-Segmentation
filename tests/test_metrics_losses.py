import pytest
import torch

from losses import CrossEntropyDiceLoss
from metrics import SegmentationMetrics


def test_dataset_level_metrics_against_hand_computation():
    metric = SegmentationMetrics(3)
    metric.update(torch.tensor([1, 2]), torch.tensor([1, 1]))
    metric.update(torch.tensor([2, 0, 1]), torch.tensor([2, 2, 255]))
    result = metric.compute()
    assert result["mean_iou"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert result["mean_dice"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert result["pixel_accuracy"] == 0.5
    assert result["valid_pixels"] == 4


def test_false_positive_class_is_included_but_absent_class_is_not():
    metric = SegmentationMetrics(4)
    metric.update(torch.tensor([1, 2]), torch.tensor([1, 1]))
    result = metric.compute()
    assert result["mean_iou"] == 0.25
    assert result["per_class"]["3"]["iou"] is None
    assert result["per_class"]["2"]["included_in_mean"]


def test_empty_metric_has_no_fabricated_score():
    metric = SegmentationMetrics()
    metric.update(torch.zeros(2, dtype=torch.long), torch.full((2,), 255))
    assert metric.compute()["mean_iou"] is None


def test_ignored_pixels_have_no_loss_gradient():
    logits = torch.randn(1, 8, 2, 2, requires_grad=True)
    target = torch.tensor([[[1, 2], [255, 255]]])
    loss = CrossEntropyDiceLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, :, 1]) == 0


def test_all_ignored_loss_is_finite_differentiable_zero():
    logits = torch.randn(1, 8, 2, 2, requires_grad=True)
    loss = CrossEntropyDiceLoss()(logits, torch.full((1, 2, 2), 255))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_ce_only_matches_pytorch():
    logits = torch.randn(2, 8, 4, 4)
    target = torch.randint(0, 8, (2, 4, 4))
    assert torch.equal(
        CrossEntropyDiceLoss(0)(logits, target), torch.nn.functional.cross_entropy(logits, target)
    )
