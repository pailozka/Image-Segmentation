"""Render a run's training curves from its recorded history."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(run):
    run = Path(run)
    history = json.loads((run / "history.json").read_text())
    if not history:
        raise ValueError("No epochs in history")
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], label=split)
    axes[0].set(title="Cross entropy + Dice loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for metric, label in (("mean_iou", "mIoU"), ("mean_dice", "Dice")):
        axes[1].plot(epochs, [row["val"][metric] for row in history], label=label)
    axes[1].set(title="Validation segmentation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    fig.savefig(run / "learning_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot_history(parser.parse_args().run)
