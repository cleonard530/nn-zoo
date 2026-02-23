"""Checkpoint save/load helpers."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
    is_best: bool = False,
    best_path: str | Path | None = None,
) -> None:
    """Save training state to path. If is_best, also save to best_path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best and best_path is not None:
        best_path = Path(best_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoint from path and return state dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_model_weights(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load only model state dict from checkpoint (e.g. for inference)."""
    state = load_checkpoint(path, device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return model.to(device)
