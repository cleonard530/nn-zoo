"""Small ResNet for CIFAR-style image classification."""

import torch
import torch.nn as nn
from torch.nn import functional as F


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    ResNet with BasicBlock for small images (CIFAR: 32x32).
    num_blocks per stage, e.g. (2, 2, 2, 2) for 8 blocks total.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        num_blocks: tuple[int, ...] = (2, 2, 2, 2),
        base_channels: int = 64,
    ):
        super().__init__()
        self.in_ch = base_channels
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(base_channels * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, s))
            self.in_ch = out_ch * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.linear(x)
