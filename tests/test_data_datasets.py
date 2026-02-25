"""Tests for data.datasets (synthetic and dataloader only to avoid download)."""

import torch

from data.datasets import (
    SyntheticSequenceDataset,
    get_dataloader,
    get_synthetic_sequence,
)


def test_synthetic_sequence_dataset_len() -> None:
    ds = SyntheticSequenceDataset(num_samples=100, seq_len=16, vocab_size=64, seed=42)
    assert len(ds) == 100


def test_synthetic_sequence_dataset_next_token_shape() -> None:
    ds = SyntheticSequenceDataset(
        num_samples=10, seq_len=8, vocab_size=32, target_mode="next_token", seed=42
    )
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_synthetic_sequence_dataset_class_shape() -> None:
    ds = SyntheticSequenceDataset(
        num_samples=10, seq_len=8, vocab_size=32, target_mode="class", num_classes=5, seed=42
    )
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.dim() == 0 or y.shape == ()
    assert y.dtype == torch.long


def test_get_synthetic_sequence_returns_dataset() -> None:
    ds = get_synthetic_sequence(num_samples=50, seq_len=16, vocab_size=64)
    assert isinstance(ds, SyntheticSequenceDataset)
    assert len(ds) == 50


def test_get_dataloader_batch_shape() -> None:
    ds = get_synthetic_sequence(num_samples=100, seq_len=16, vocab_size=64, target_mode="class", num_classes=10)
    loader = get_dataloader(ds, batch_size=8, shuffle=True)
    x, y = next(iter(loader))
    assert x.shape == (8, 16)
    assert y.shape == (8,)
