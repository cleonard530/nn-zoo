"""Train ResNet on CIFAR-10."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_cifar10, get_dataloader
from models.resnet import ResNet
from utils import get_device, get_run_id, log_epoch, save_checkpoint, save_training_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="./weights/resnet")
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_cifar10(args.data_dir, train=True)
    val_ds = get_cifar10(args.data_dir, train=False)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResNet(in_channels=3, num_classes=10, num_blocks=(2, 2, 2, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        log_epoch(epoch, {"train_loss": train_loss, "val_acc": val_acc})

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc},
            Path(args.save_dir) / f"last_{run_id}.pt",
            is_best=is_best,
            best_path=Path(args.save_dir) / f"best_{run_id}.pt",
        )

    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {"epochs": args.epochs, "best_val_acc": best_acc, "final_val_acc": val_acc},
        model_hp={"in_channels": 3, "num_classes": 10, "num_blocks": (2, 2, 2, 2)},
    )


if __name__ == "__main__":
    main()
