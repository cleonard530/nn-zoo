"""Run U-Net inference on a few MNIST segmentation samples."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_mnist_segmentation
from models.unet import UNet
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/unet/best.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)

    model = UNet(in_channels=1, out_channels=1, features=(32, 64, 128))
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    test_ds = get_mnist_segmentation(args.data_dir, train=False)
    for i in range(min(args.num_samples, len(test_ds))):
        x, mask_true = test_ds[i]
        x = x.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        iou = (pred * mask_true.unsqueeze(0).unsqueeze(0).to(device)).sum() / (
            (pred + mask_true.unsqueeze(0).unsqueeze(0).to(device) > 0).float().sum() + 1e-8
        )
        print(f"Sample {i}: IoU = {iou.item():.4f}")


if __name__ == "__main__":
    main()
