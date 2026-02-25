"""Device, CLI args, logging, validation metrics, and train loop."""

import argparse
import csv
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


# --- Device ---


def get_device(use_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- CLI ---


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    include_data_dir: bool = True,
    default_save_dir: str = "./weights",
    default_epochs: int = 10,
    default_batch_size: int = 128,
    default_lr: float = 1e-3,
) -> None:
    """Add common training arguments: data_dir (optional), epochs, batch_size, lr, save_dir, use_cuda."""
    if include_data_dir:
        parser.add_argument("--data_dir", type=str, default="./data", help="Data root")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA if available",
    )


# --- Logging ---


def log_epoch(epoch: int, metrics: dict[str, float], log_file: Path | None = None) -> None:
    """Print metrics to console and optionally append to CSV."""
    parts = [f"epoch {epoch}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    print("  ".join(parts))
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_file.exists()
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            row: dict[str, Any] = {"epoch": epoch, **metrics}
            writer.writerow(row)


# --- Validation metrics ---


def validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    batch_loss_fn: Callable[[torch.nn.Module, tuple, torch.device], torch.Tensor],
) -> float:
    """
    Run the model in eval mode on the validation loader and return mean loss.
    batch_loss_fn(model, batch, device) should return a scalar loss tensor for that batch.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = batch_loss_fn(model, batch, device)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def validation_accuracy(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Run the model in eval mode on the validation loader and return
    accuracy (fraction of samples where argmax(logits) == target).
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# --- Train loop ---


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.nn.Module, tuple, torch.device], torch.Tensor],
) -> float:
    """
    Run one training epoch: model.train(), loop over batches, backward, step.

    loss_fn(model, batch, device) must return a scalar loss tensor for that batch.
    The batch is passed as-is; move inputs to device inside loss_fn.
    """
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model, batch, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
