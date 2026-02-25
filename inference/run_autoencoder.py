"""Run Autoencoder inference: encode and decode a few MNIST images."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_mnist
from models.autoencoder import Autoencoder
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/autoencoder/best.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)

    model = Autoencoder(input_size=784, latent_dim=args.latent_dim)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    test_ds = get_mnist(args.data_dir, train=False)
    for i in range(min(args.num_samples, len(test_ds))):
        x, _ = test_ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)
        mse = (recon - x.view(1, -1)).pow(2).mean().item()
        print(f"Sample {i}: reconstruction MSE = {mse:.4f}")


if __name__ == "__main__":
    main()
