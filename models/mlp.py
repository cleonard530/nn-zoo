"""Feed-forward MLP for image classification (e.g. MNIST flattened)."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple multi-layer perceptron: flatten -> linear layers -> logits.
    Default sizes chosen for MNIST (28*28 = 784 input, 10 classes).
    """

    def __init__(
        self,
        input_size: int = 784,
        num_classes: int = 10,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)
