"""Train U-Net on MNIST segmentation (binary mask)."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_mnist_segmentation
from models.unet import UNet
from utils import get_device, log_epoch, save_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="./weights/unet")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = get_device(use_cuda=not args.no_cuda)

    train_ds = get_mnist_segmentation(args.data_dir, train=True)
    val_ds = get_mnist_segmentation(args.data_dir, train=False)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=1, out_channels=1, features=(32, 64, 128)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, mask)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(device), mask.to(device)
                logits = model(x)
                val_loss += criterion(logits, mask).item()
        val_loss /= len(val_loader)
        log_epoch(epoch, {"train_loss": train_loss, "val_loss": val_loss})

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
            Path(args.save_dir) / "last.pt",
            is_best=is_best,
            best_path=Path(args.save_dir) / "best.pt",
        )


if __name__ == "__main__":
    main()
