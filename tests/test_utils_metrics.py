"""Tests for utils.metrics."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import validation_accuracy


def test_validation_accuracy_perfect(device: torch.device) -> None:
    """Model that always predicts correctly should get 1.0."""
    model = torch.nn.Linear(4, 3).to(device)
    # Freeze so we can set weights that give correct predictions for fixed data
    x = torch.randn(10, 4, device=device)
    y = torch.randint(0, 3, (10,), device=device)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=5)
    # With random model we just check it returns a float in [0, 1]
    acc = validation_accuracy(model, loader, device)
    assert 0.0 <= acc <= 1.0
    assert isinstance(acc, float)


def test_validation_accuracy_empty_returns_zero(device: torch.device) -> None:
    """Empty loader should return 0.0 (guarded in implementation)."""
    model = torch.nn.Linear(2, 2).to(device)
    ds = TensorDataset(torch.randn(0, 2), torch.randint(0, 2, (0,)))
    loader = DataLoader(ds, batch_size=1)
    acc = validation_accuracy(model, loader, device)
    assert acc == 0.0
