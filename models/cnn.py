"""Small CNN for image classification (MNIST / CIFAR-10)."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional network: conv blocks -> flatten -> linear -> logits.
    in_channels=1, side=28 for MNIST; in_channels=3, side=32 for CIFAR-10.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        side: int = 28,
        channels: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        s = side
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                ]
            )
            prev = c
            s = s // 2
        self.conv = nn.Sequential(*layers)
        self.flat_size = prev * s * s
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)
