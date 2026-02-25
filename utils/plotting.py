import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# CIFAR-10 normalization (for display)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def plot_results(
    n: int,
    results: list[tuple[torch.Tensor, Any, Any]],
    args: Any,
    title_prefix: str,
) -> None:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        if idx < n:
            img, y_true, pred = results[idx]
            img = img.numpy()
            if args.cifar:
                img = img.transpose(1, 2, 0) * np.array(CIFAR_STD) + np.array(CIFAR_MEAN)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            else:
                ax.imshow(img.squeeze(0), cmap="gray")
            ax.set_title(f"true: {y_true}, pred: {pred}")
        ax.axis("off")
    plt.tight_layout()
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "cifar10" if args.cifar else "mnist"
    out_path = plot_dir / f"{title_prefix}_inference_{dataset_name}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")

