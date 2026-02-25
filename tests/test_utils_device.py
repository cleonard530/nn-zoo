"""Tests for utils.device."""

import torch

import pytest

# conftest adds repo root to path
from utils.device import get_device


def test_get_device_returns_device_type() -> None:
    out = get_device(use_cuda=True)
    assert isinstance(out, torch.device)


def test_get_device_cpu_when_disabled() -> None:
    out = get_device(use_cuda=False)
    assert out.type == "cpu"


def test_get_device_cpu_or_cuda_when_enabled() -> None:
    out = get_device(use_cuda=True)
    assert out.type in ("cpu", "cuda")
