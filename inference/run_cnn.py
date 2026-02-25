"""Run CNN inference on a few MNIST or CIFAR samples."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_cifar10, get_mnist
from models.cnn import CNN
from utils import get_device, load_model_weights, plot_results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/cnn/best.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cifar", action="store_true")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    p.add_argument("--plot_dir", type=str, default="plots", help="Directory to save inference plots")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)

    if args.cifar:
        test_ds = get_cifar10(args.data_dir, train=False)
        model = CNN(in_channels=3, num_classes=10, side=32)
    else:
        test_ds = get_mnist(args.data_dir, train=False)
        model = CNN(in_channels=1, num_classes=10, side=28)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    n = min(args.num_samples, len(test_ds))
    results = []
    for i in range(n):
        x, y_true = test_ds[i]
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_batch)
        pred = logits.argmax(1).item()
        print(f"Sample {i}: true={y_true}, pred={pred}")
        results.append((x, y_true, pred))

    if n < 26:
        plot_results(n, results, args, "cnn")


if __name__ == "__main__":
    main()
