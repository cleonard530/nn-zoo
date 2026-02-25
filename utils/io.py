"""Checkpoint save/load and training run metadata."""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# --- Checkpoint ---


def save_epoch_checkpoint(
    model: nn.Module,
    epoch: int,
    metric_value: float,
    metric_key: str,
    save_dir: str | Path,
    run_id: str,
    is_best: bool,
) -> None:
    """Save epoch checkpoint with standard keys (model_state_dict, epoch, metric)."""
    save_dir = Path(save_dir)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        metric_key: metric_value,
    }
    save_checkpoint(
        state,
        save_dir / f"last_{run_id}.pt",
        is_best=is_best,
        best_path=save_dir / f"best_{run_id}.pt",
    )


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


# --- Training metadata ---


def get_run_id() -> str:
    """Return current date and time as Y_M_D_H_M_S for use in filenames."""
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def save_training_metadata(
    save_dir: str | Path,
    run_id: str,
    args: Any,
    metrics: dict[str, Any],
    model_hp: dict[str, Any] | None = None,
) -> None:
    """Write a .txt file with run_id, hyperparameters, and final/best metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"meta_{run_id}.txt"
    lines = [
        f"run_id: {run_id}",
        f"timestamp: {datetime.now().isoformat()}",
        "",
        "=== Metrics ===",
    ]
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.extend(["", "=== Hyperparameters (args) ==="])
    args_dict = vars(args) if hasattr(args, "__dict__") else args
    for k, v in sorted(args_dict.items()):
        lines.append(f"  {k}: {v}")
    if model_hp:
        lines.extend(["", "=== Model hyperparameters ==="])
        for k, v in sorted(model_hp.items()):
            lines.append(f"  {k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Metadata saved to {path}")
