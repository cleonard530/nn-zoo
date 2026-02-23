"""Small U-Net for image segmentation (e.g. binary mask on small images)."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Small U-Net: encoder path, bottleneck, decoder path with skip connections.
    in_channels=1 for grayscale, out_channels=1 for binary mask (or N for N classes).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoder.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f
        self.bottleneck = DoubleConv(prev, prev * 2)
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev = prev * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, f, 2, stride=2))
            self.decoder.append(DoubleConv(2 * f, f))
            prev = f  # next block input channels = current f
        self.final = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final(x)
