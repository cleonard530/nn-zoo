"""Run GAN inference: generate images from random latent vectors."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.gan import GAN
from utils import get_device, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/gan/best.pt")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)

    gan = GAN(latent_dim=args.latent_dim, image_size=784).to(device)
    state = load_checkpoint(args.checkpoint, device)
    gan.generator.load_state_dict(state["generator_state_dict"], strict=True)
    gan.generator.eval()

    z = torch.randn(args.num_samples, args.latent_dim, device=device)
    with torch.no_grad():
        gen = gan.generator(z)
    print(f"Generated {args.num_samples} images, shape {gen.shape} (flattened 1x784)")


if __name__ == "__main__":
    main()
