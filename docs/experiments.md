# Experiment protocol and portfolio presentation

## Current evidence

The repository provides implemented architectures and automated CPU engineering checks. It does not yet contain a real-data experiment establishing whether residual blocks or attention improve segmentation. The parameter counts in the README are measured model properties; accuracy, runtime, and memory claims need separate measurements.

## Run a controlled comparison

1. Obtain and document the dataset version, label encoding, and evaluation taxonomy. Inspect several image/mask pairs before a long training run.
2. Use fixed train/validation groups. Reserve an independent labeled holdout before repeated tuning when available. Use validation for architecture/checkpoint selection and evaluate the holdout only after those choices are frozen.
3. Train `unet`, `residual_unet`, `attention_unet`, and `residual_attention_unet` with identical base width, depth, dropout, resolution, augmentation, loss, optimizer, epoch budget, split and seed. Change only `model.architecture` in copies of the same JSON config. For official Cityscapes start from `configs/cityscapes.json`.
4. Use at least three seeds for a final comparison if compute allows. With explicit train/val folders, the split remains fixed as seeds vary. With an automatic grouped split, changing `seed` changes the split too; establish explicit folders first for a fixed-split, multi-seed experiment.
5. Record all runs, including worse variants. Report mean and standard deviation of category mIoU and Dice, per-class IoU, parameter count, training time, and the actual hardware. Do not infer speed from parameter count.
6. Compare input, target, prediction and overlay on the same predetermined examples. Include failures such as small humans, thin poles, boundary errors, and confusing vegetation/construction regions when present in your data.

The normalized baseline and enhanced model have different parameter counts. The comparison isolates the added components at fixed stage widths, not at precisely matched parameter budgets. An additional width-matched experiment can examine that tradeoff.

For loss ablation, keep the selected architecture fixed and set `dice_weight` to `0` for cross entropy only; compare with `0.5`. For augmentation ablation, toggle `augment`. Change one factor at a time.

## Result table to fill after real training

| Model | Dataset/split | Seeds | Category mIoU (mean ± SD) | Dice (mean ± SD) | Parameters |
| --- | --- | --- | --- | --- | ---: |
| U-Net | Not run | — | — | — | 7,240,552 |
| Residual U-Net | Not run | — | — | — | 7,588,808 |
| Attention U-Net | Not run | — | — | — | 7,328,316 |
| Residual attention U-Net | Not run | — | — | — | 7,676,572 |

Scores are fractions in JSON; convert consistently if displaying percentages. State the configured evaluation resolution. These seven-category scores are not interchangeable with the official 19-class Cityscapes leaderboard.

## What to explain in an interview

- Why pixel accuracy alone can hide poor performance on rare/small classes, and why the project accumulates one confusion matrix across the dataset.
- Why target masks need nearest-neighbor interpolation, synchronized geometric transforms, and strict palette validation.
- How GroupNorm supports small batches, residual shortcuts affect gradient paths, and attention gates weight skip features using decoder context.
- How the baseline and component ablations separate hypotheses from measured improvements.
- How split manifests, content hashes, configuration snapshots, and resumable checkpoints support reproducibility.
- Which tradeoffs remain: resized inputs lose fine detail, full-frame resizing can distort other aspect ratios, training starts from random weights, and GPU behavior needs hardware testing.

## Accurate CV wording before a real benchmark

> Built a reproducible PyTorch pipeline for urban-scene semantic segmentation, implementing U-Net, residual and attention variants with IoU/Dice evaluation, resumable training, original-resolution prediction exports, and automated CPU tests.

After real experiments, add the measured dataset, resolution, seed count and improvement over the baseline. For example, replace the placeholders in “Improved category mIoU from [baseline] to [result] on [held-out split], averaged over [N] seeds.” Do not use that sentence until the saved reports support it.
