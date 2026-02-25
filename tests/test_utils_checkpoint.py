"""Tests for checkpoint and model load/save (utils.io)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils import load_checkpoint, load_model_weights, save_checkpoint, save_epoch_checkpoint


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_save_checkpoint_creates_file(tmp_path: Path) -> None:
    state = {"epoch": 1, "model_state_dict": _TinyModel().state_dict()}
    path = tmp_path / "sub" / "ckpt.pt"
    save_checkpoint(state, path)
    assert path.exists()


def test_save_epoch_checkpoint_creates_last_and_best(tmp_path: Path, device: torch.device) -> None:
    model = _TinyModel()
    save_epoch_checkpoint(model, epoch=1, metric_value=0.9, metric_key="val_acc", save_dir=tmp_path, run_id="2025_01_01_00_00_00", is_best=True)
    last_path = tmp_path / "last_2025_01_01_00_00_00.pt"
    best_path = tmp_path / "best_2025_01_01_00_00_00.pt"
    assert last_path.exists()
    assert best_path.exists()
    state = load_checkpoint(last_path, device)
    assert state["epoch"] == 1
    assert state["val_acc"] == 0.9
    assert "model_state_dict" in state


def test_save_checkpoint_is_best_creates_best_path(tmp_path: Path) -> None:
    state = {"epoch": 1}
    path = tmp_path / "last.pt"
    best_path = tmp_path / "best.pt"
    save_checkpoint(state, path, is_best=True, best_path=best_path)
    assert path.exists()
    assert best_path.exists()


def test_load_checkpoint_returns_state(tmp_path: Path) -> None:
    state = {"epoch": 5, "val_acc": 0.9}
    path = tmp_path / "ckpt.pt"
    torch.save(state, path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded["epoch"] == 5
    assert loaded["val_acc"] == 0.9


def test_load_checkpoint_raises_if_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "nonexistent.pt", torch.device("cpu"))


def test_load_model_weights_from_model_state_dict(tmp_path: Path, device: torch.device) -> None:
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    model2 = _TinyModel()
    load_model_weights(model2, path, device)
    x = torch.randn(1, 2)
    assert torch.allclose(model(x), model2(x))


def test_load_model_weights_from_raw_state_dict(tmp_path: Path, device: torch.device) -> None:
    model = _TinyModel()
    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)
    model2 = _TinyModel()
    load_model_weights(model2, path, device)
    x = torch.randn(1, 2)
    assert torch.allclose(model(x), model2(x))
