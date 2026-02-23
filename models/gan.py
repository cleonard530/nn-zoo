"""GAN: Generator and Discriminator for image generation (e.g. MNIST)."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps latent z to image (e.g. 64 -> 784 for MNIST)."""

    def __init__(
        self,
        latent_dim: int = 64,
        out_size: int = 784,
        hidden_sizes: tuple[int, ...] = (256, 512),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)])
            prev = h
        layers.append(nn.Linear(prev, out_size))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Maps image to scalar logit (real/fake)."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)


class GAN(nn.Module):
    """Container holding generator and discriminator (for checkpointing both)."""

    def __init__(
        self,
        latent_dim: int = 64,
        image_size: int = 784,
        g_hidden: tuple[int, ...] = (256, 512),
        d_hidden: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        self.generator = Generator(latent_dim, image_size, g_hidden)
        self.discriminator = Discriminator(image_size, d_hidden)
