"""Train Autoencoder on MNIST (reconstruction)."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_mnist
from models.autoencoder import Autoencoder
from utils import get_device, get_run_id, log_epoch, save_checkpoint, save_training_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--save_dir", type=str, default="./weights/autoencoder")
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_mnist(args.data_dir, train=True)
    val_ds = get_mnist(args.data_dir, train=False)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = Autoencoder(input_size=784, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction="sum")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = criterion(recon, x.view(x.size(0), -1)) / x.size(0)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon = model(x)
                val_loss += criterion(recon, x.view(x.size(0), -1)).item() / x.size(0)
        val_loss /= len(val_loader)
        log_epoch(epoch, {"train_loss": train_loss, "val_loss": val_loss})

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
            Path(args.save_dir) / f"last_{run_id}.pt",
            is_best=is_best,
            best_path=Path(args.save_dir) / f"best_{run_id}.pt",
        )

    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {"epochs": args.epochs, "best_val_loss": best_loss, "final_val_loss": val_loss},
        model_hp={"input_size": 784, "latent_dim": args.latent_dim},
    )


if __name__ == "__main__":
    main()
