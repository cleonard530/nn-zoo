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
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_epoch_checkpoint, save_training_metadata, train_epoch, validation_loss


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, default_save_dir="./weights/unet", default_epochs=15, default_batch_size=64)
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_mnist_segmentation(args.data_dir, train=True)
    val_ds = get_mnist_segmentation(args.data_dir, train=False)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=1, out_channels=1, features=(32, 64, 128)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    def loss_fn(m, batch, dev):
        x, mask = batch[0].to(dev), batch[1].to(dev)
        return criterion(m(x), mask)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, opt, loss_fn)
        def batch_loss(m, batch, dev):
            x, mask = batch[0].to(dev), batch[1].to(dev)
            return criterion(m(x), mask)
        val_loss = validation_loss(model, val_loader, device, batch_loss)
        log_epoch(epoch, {"train_loss": train_loss, "val_loss": val_loss})

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_epoch_checkpoint(model, epoch, val_loss, "val_loss", args.save_dir, run_id, is_best)

    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {"epochs": args.epochs, "best_val_loss": best_loss, "final_val_loss": val_loss},
        model_hp={"in_channels": 1, "out_channels": 1, "features": (32, 64, 128)},
    )


if __name__ == "__main__":
    main()
