"""Pytest fixtures and repo root path for imports."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure repo root is on path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def device() -> torch.device:
    """Use CPU for tests to avoid CUDA dependency."""
    return torch.device("cpu")


@pytest.fixture
def tmp_plot_dir(tmp_path: Path) -> Path:
    """Temporary directory for saving plots."""
    d = tmp_path / "plots"
    d.mkdir()
    return d
