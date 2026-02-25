"""Metric helpers for training (e.g. validation accuracy)."""

import torch
from torch.utils.data import DataLoader


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
