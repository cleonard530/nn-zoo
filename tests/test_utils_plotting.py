"""Tests for inference plotting (utils.plotting)."""

from pathlib import Path
from types import SimpleNamespace

import torch

from utils import plot_results


def test_plot_results_saves_file(tmp_plot_dir: Path) -> None:
    n = 4
    # (img, y_true, pred): grayscale 1x8x8
    results = [
        (torch.rand(1, 8, 8), 0, 0),
        (torch.rand(1, 8, 8), 1, 1),
        (torch.rand(1, 8, 8), 2, 2),
        (torch.rand(1, 8, 8), 3, 3),
    ]
    args = SimpleNamespace(cifar=False, plot_dir=str(tmp_plot_dir))
    plot_results(n, results, args, "test")
    out_path = tmp_plot_dir / "test_inference_mnist_n4.png"
    assert out_path.exists()


def test_plot_results_cifar_saves_file(tmp_plot_dir: Path) -> None:
    n = 2
    results = [
        (torch.rand(3, 8, 8), 0, 0),
        (torch.rand(3, 8, 8), 1, 1),
    ]
    args = SimpleNamespace(cifar=True, plot_dir=str(tmp_plot_dir))
    plot_results(n, results, args, "cnn")
    out_path = tmp_plot_dir / "cnn_inference_cifar10_n2.png"
    assert out_path.exists()
