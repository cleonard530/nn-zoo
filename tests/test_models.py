"""Tests for model forward passes (shape and no crash)."""

import torch

from models.autoencoder import Autoencoder
from models.bert_tiny import BertTiny
from models.cnn import CNN
from models.gan import Discriminator, GAN, Generator
from models.gpt_tiny import GPTTiny
from models.lstm import LSTM
from models.mlp import MLP
from models.resnet import ResNet
from models.unet import UNet
from models.vae import VAE


def test_mlp_forward_shape(device: torch.device) -> None:
    model = MLP(input_size=784, num_classes=10).to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_cnn_forward_shape_mnist(device: torch.device) -> None:
    model = CNN(in_channels=1, num_classes=10, side=28).to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_cnn_forward_shape_cifar(device: torch.device) -> None:
    model = CNN(in_channels=3, num_classes=10, side=32).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_autoencoder_forward_shape(device: torch.device) -> None:
    model = Autoencoder(input_size=784, latent_dim=32).to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    out = model(x)
    assert out.shape == (4, 784)


def test_autoencoder_encode_decode_shape(device: torch.device) -> None:
    model = Autoencoder(input_size=784, latent_dim=32).to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    z = model.encode(x)
    assert z.shape == (4, 32)
    recon = model.decode(z)
    assert recon.shape == (4, 784)


def test_vae_forward_shape(device: torch.device) -> None:
    model = VAE(input_size=784, latent_dim=20).to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    recon, mu, logvar = model(x)
    assert recon.shape == (4, 784)
    assert mu.shape == (4, 20)
    assert logvar.shape == (4, 20)


def test_resnet_forward_shape(device: torch.device) -> None:
    model = ResNet(in_channels=3, num_classes=10, num_blocks=(2, 2, 2, 2)).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_bert_tiny_forward_shape(device: torch.device) -> None:
    model = BertTiny(vocab_size=64, max_len=32, d_model=64, num_heads=4, num_layers=2, num_classes=10).to(device)
    x = torch.randint(0, 64, (4, 32), device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_gpt_tiny_forward_shape(device: torch.device) -> None:
    model = GPTTiny(vocab_size=64, max_len=32, d_model=64, num_heads=4, num_layers=2).to(device)
    x = torch.randint(0, 64, (4, 32), device=device)
    out = model(x)
    assert out.shape == (4, 32, 64)


def test_gan_generator_forward_shape(device: torch.device) -> None:
    gen = Generator(latent_dim=64, out_size=784).to(device)
    z = torch.randn(4, 64, device=device)
    out = gen(z)
    assert out.shape == (4, 784)


def test_gan_discriminator_forward_shape(device: torch.device) -> None:
    disc = Discriminator(input_size=784).to(device)
    x = torch.randn(4, 784, device=device)
    out = disc(x)
    assert out.shape == (4,)


def test_gan_container_has_generator_discriminator() -> None:
    gan = GAN(latent_dim=64, image_size=784)
    assert hasattr(gan, "generator")
    assert hasattr(gan, "discriminator")


def test_lstm_forward_classify_shape(device: torch.device) -> None:
    model = LSTM(vocab_size=64, embed_dim=32, hidden_size=64, num_layers=2, num_classes=10, mode="classify").to(device)
    x = torch.randint(0, 64, (4, 16), device=device)
    out = model(x)
    assert out.shape == (4, 10)


def test_lstm_forward_predict_shape(device: torch.device) -> None:
    model = LSTM(vocab_size=64, embed_dim=32, hidden_size=64, num_layers=2, num_classes=64, mode="predict").to(device)
    x = torch.randint(0, 64, (4, 16), device=device)
    out = model(x)
    assert out.shape == (4, 16, 64)


def test_unet_forward_shape(device: torch.device) -> None:
    model = UNet(in_channels=1, out_channels=1, features=(32, 64)).to(device)
    x = torch.randn(2, 1, 32, 32, device=device)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)
