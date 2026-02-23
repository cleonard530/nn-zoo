"""Autoencoder for reconstruction (e.g. MNIST)."""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Encoder: flatten -> linear layers -> latent.
    Decoder: linear layers -> reshape to image.
    Default for MNIST: 784 -> 256 -> 64 -> latent_dim -> 64 -> 256 -> 784.
    """

    def __init__(
        self,
        input_size: int = 784,
        latent_dim: int = 32,
        hidden_sizes: tuple[int, ...] = (256, 64),
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        # Encoder
        enc: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            enc.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        # Decoder
        dec: list[nn.Module] = [nn.Linear(latent_dim, prev), nn.ReLU(inplace=True)]
        for h in reversed(hidden_sizes):
            dec.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec.append(nn.Linear(prev, input_size))
        dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
