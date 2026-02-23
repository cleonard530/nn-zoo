"""Variational Autoencoder (VAE) for MNIST."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    VAE: encoder outputs mu, logvar; sample z; decoder reconstructs.
    Reconstruction loss (BCE) + KL divergence. Default for MNIST 784 <-> latent_dim.
    """

    def __init__(
        self,
        input_size: int = 784,
        latent_dim: int = 20,
        hidden_sizes: tuple[int, ...] = (400, 200),
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        # Encoder to mu and logvar
        enc: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            enc.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        self.encoder_fc = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        dec: list[nn.Module] = [nn.Linear(latent_dim, prev), nn.ReLU(inplace=True)]
        for h in reversed(hidden_sizes):
            dec.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec.append(nn.Linear(prev, input_size))
        dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        h = self.encoder_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
