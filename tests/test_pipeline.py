import json
from dataclasses import replace

import pytest
import torch
from PIL import Image

import training as train_module
from config import TrainConfig
from evaluate import evaluate
from inference import inference
from network import ModelConfig
from utils import load_checkpoint


def tiny_config():
    return TrainConfig(
        model=ModelConfig(base_channels=4, depth=2, dropout=0.1),
        image_size=(16, 32),
        epochs=2,
        batch_size=2,
        amp=False,
    )


def test_train_checkpoint_evaluate_and_unlabeled_inference(demo_data, tmp_path):
    output = tmp_path / "run"
    history = train_module.training(demo_data, tiny_config(), output, "cpu")
    assert len(history) == 2
    model, checkpoint = load_checkpoint(output / "best.pt")
    assert checkpoint["best_iou"] == max(row["val"]["mean_iou"] for row in history)
    report = evaluate(demo_data, output / "best.pt", tmp_path / "metrics.json", device="cpu")
    assert report["metrics"]["mean_iou"] == checkpoint["best_iou"]
    photo = tmp_path / "unlabeled.png"
    Image.new("RGB", (45, 27), (34, 56, 78)).save(photo)
    records = inference(photo, output / "best.pt", 999, tmp_path / "predictions", "cpu")
    assert len(records) == 1
    with Image.open(tmp_path / "predictions/unlabeled.png.ids.png") as mask:
        assert mask.size == (45, 27) and mask.mode == "L"
    assert json.loads((output / "environment.json").read_text())["device"] == "cpu"
    with pytest.raises(FileExistsError):
        train_module.training(demo_data, tiny_config(), output, "cpu")


def test_resume_matches_uninterrupted_training(demo_data, tmp_path, monkeypatch):
    config = tiny_config()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_module.training(demo_data, config, full, "cpu")
    original_save = train_module.save_checkpoint

    class SimulatedInterruption(Exception):
        pass

    def interrupted_save(path, payload):
        original_save(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 0:
            raise SimulatedInterruption

    monkeypatch.setattr(train_module, "save_checkpoint", interrupted_save)
    with pytest.raises(SimulatedInterruption):
        train_module.training(demo_data, config, resumed, "cpu")
    monkeypatch.setattr(train_module, "save_checkpoint", original_save)
    train_module.training(demo_data, config, resumed, "cpu", resume=resumed / "last.pt")
    _, expected = load_checkpoint(full / "last.pt")
    _, actual = load_checkpoint(resumed / "last.pt")
    for name in expected["model_state"]:
        assert torch.equal(actual["model_state"][name], expected["model_state"][name]), name
    assert actual["history"][-1]["val"] == expected["history"][-1]["val"]
    assert actual["scheduler_state"] == expected["scheduler_state"]
    with pytest.raises(ValueError, match="identical configuration"):
        train_module.training(
            demo_data, replace(config, learning_rate=0.01), resumed, "cpu", resume=resumed / "last.pt"
        )


def test_config_validation_and_roundtrip():
    config = tiny_config()
    assert TrainConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        replace(config, image_size=(1, 1))
    with pytest.raises(ValueError):
        replace(config, encoding="unknown")
    with pytest.raises(ValueError):
        replace(config, encoding="cityscapes_ids", ignore_void=False)
    with pytest.raises(ValueError, match="finite"):
        replace(config, learning_rate=float("nan"))
