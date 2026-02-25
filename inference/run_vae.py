"""Run VAE inference: reconstruct and sample from prior."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_mnist
from models.vae import VAE
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/vae/best.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--latent_dim", type=int, default=20)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--sample_prior", type=int, default=0, help="Number of images to generate from prior")
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)

    model = VAE(input_size=784, latent_dim=args.latent_dim)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    test_ds = get_mnist(args.data_dir, train=False)
    for i in range(min(args.num_samples, len(test_ds))):
        x, _ = test_ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            recon, mu, logvar = model(x)
        mse = (recon - x.view(1, -1)).pow(2).mean().item()
        print(f"Sample {i}: reconstruction MSE = {mse:.4f}")

    if args.sample_prior > 0:
        with torch.no_grad():
            z = torch.randn(args.sample_prior, args.latent_dim, device=device)
            gen = model.decode(z)
        print(f"Generated {args.sample_prior} images from prior (shape {gen.shape})")


if __name__ == "__main__":
    main()
