"""Smoke tests for training: a few steps with synthetic data to verify training loops run."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import get_dataloader, get_synthetic_sequence
from models.autoencoder import Autoencoder
from models.bert_tiny import BertTiny
from models.cnn import CNN
from models.gan import GAN
from models.gpt_tiny import GPTTiny
from models.lstm import LSTM
from models.mlp import MLP
from models.resnet import ResNet
from models.unet import UNet
from models.vae import VAE


def _vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    bce = nn.BCELoss(reduction="sum")(recon, x) / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + kl


def test_mlp_training_steps(device: torch.device) -> None:
    model = MLP(input_size=784, num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()
    assert loss.dim() == 0


def test_cnn_training_steps(device: torch.device) -> None:
    model = CNN(in_channels=1, num_classes=10, side=28).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_autoencoder_training_steps(device: torch.device) -> None:
    model = Autoencoder(input_size=784, latent_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(4, 1, 28, 28, device=device)
    criterion = nn.BCELoss(reduction="sum")
    for _ in range(3):
        opt.zero_grad()
        recon = model(x)
        loss = criterion(recon, x.view(4, -1)) / 4
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_vae_training_steps(device: torch.device) -> None:
    model = VAE(input_size=784, latent_dim=20).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(4, 1, 28, 28, device=device)
    for _ in range(3):
        opt.zero_grad()
        recon, mu, logvar = model(x)
        loss = _vae_loss(recon, x.view(4, -1), mu, logvar)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_resnet_training_steps(device: torch.device) -> None:
    model = ResNet(in_channels=3, num_classes=10, num_blocks=(2, 2, 2, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    criterion = nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_bert_tiny_training_steps(device: torch.device) -> None:
    model = BertTiny(vocab_size=64, max_len=16, d_model=64, num_heads=4, num_layers=2, num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ds = get_synthetic_sequence(num_samples=32, seq_len=16, vocab_size=64, target_mode="class", num_classes=10, seed=42)
    loader = get_dataloader(ds, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= 3:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_gpt_tiny_training_steps(device: torch.device) -> None:
    model = GPTTiny(vocab_size=64, max_len=16, d_model=64, num_heads=4, num_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    ds = get_synthetic_sequence(num_samples=32, seq_len=16, vocab_size=64, target_mode="next_token", seed=42)
    loader = get_dataloader(ds, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= 3:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 64), y.view(-1))
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_lstm_training_steps(device: torch.device) -> None:
    model = LSTM(vocab_size=64, embed_dim=32, hidden_size=64, num_layers=2, num_classes=10, mode="classify").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = get_synthetic_sequence(num_samples=32, seq_len=16, vocab_size=64, target_mode="class", num_classes=10, seed=42)
    loader = get_dataloader(ds, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= 3:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()


def test_gan_training_steps(device: torch.device) -> None:
    gan = GAN(latent_dim=32, image_size=784).to(device)
    G, D = gan.generator, gan.discriminator
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4)
    real_label, fake_label = 1.0, 0.0
    for _ in range(3):
        x_real = torch.randn(4, 784, device=device)
        B = x_real.size(0)
        opt_d.zero_grad()
        pred_real = D(x_real)
        loss_d_real = nn.BCEWithLogitsLoss()(pred_real, torch.full((B,), real_label, device=device, dtype=torch.float32))
        z = torch.randn(B, 32, device=device)
        fake = G(z)
        pred_fake = D(fake.detach())
        loss_d_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.full((B,), fake_label, device=device, dtype=torch.float32))
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        opt_d.step()
        opt_g.zero_grad()
        pred_fake = D(fake)
        loss_g = nn.BCEWithLogitsLoss()(pred_fake, torch.full((B,), real_label, device=device, dtype=torch.float32))
        loss_g.backward()
        opt_g.step()
    assert torch.isfinite(loss_d).item()
    assert torch.isfinite(loss_g).item()


def test_unet_training_steps(device: torch.device) -> None:
    model = UNet(in_channels=1, out_channels=1, features=(32, 64)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(4, 1, 32, 32, device=device)
    mask = (torch.rand(4, 1, 32, 32, device=device) > 0.5).float()
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(3):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, mask)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()
