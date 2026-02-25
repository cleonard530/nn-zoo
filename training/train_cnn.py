"""Train CNN on MNIST (or CIFAR-10 with --cifar)."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_cifar10, get_dataloader, get_mnist
from models.cnn import CNN
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_epoch_checkpoint, save_training_metadata, train_epoch, validation_accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, default_save_dir="./weights/cnn", default_epochs=15)
    p.add_argument("--cifar", action="store_true", help="Use CIFAR-10 instead of MNIST")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    if args.cifar:
        train_ds = get_cifar10(args.data_dir, train=True)
        val_ds = get_cifar10(args.data_dir, train=False)
        model = CNN(in_channels=3, num_classes=10, side=32)
    else:
        train_ds = get_mnist(args.data_dir, train=True)
        val_ds = get_mnist(args.data_dir, train=False)
        model = CNN(in_channels=1, num_classes=10, side=28)
    model = model.to(device)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(m, batch, dev):
        x, y = batch[0].to(dev), batch[1].to(dev)
        return criterion(m(x), y)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, opt, loss_fn)
        val_acc = validation_accuracy(model, val_loader, device)
        log_epoch(epoch, {"train_loss": train_loss, "val_acc": val_acc})

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        save_epoch_checkpoint(model, epoch, val_acc, "val_acc", args.save_dir, run_id, is_best)

    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {"epochs": args.epochs, "best_val_acc": best_acc, "final_val_acc": val_acc},
        model_hp={"in_channels": 3 if args.cifar else 1, "num_classes": 10, "side": 32 if args.cifar else 28},
    )


if __name__ == "__main__":
    main()
