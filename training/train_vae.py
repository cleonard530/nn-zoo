"""Train VAE on MNIST."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_mnist
from models.vae import VAE
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_epoch_checkpoint, save_training_metadata, train_epoch, validation_loss


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    bce = nn.BCELoss(reduction="sum")(recon, x) / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + kl


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, default_save_dir="./weights/vae", default_epochs=20)
    p.add_argument("--latent_dim", type=int, default=20)
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_mnist(args.data_dir, train=True)
    val_ds = get_mnist(args.data_dir, train=False)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = VAE(input_size=784, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(m, batch, dev):
        x = batch[0].to(dev).view(batch[0].size(0), -1)
        recon, mu, logvar = m(x.view(x.size(0), 1, 28, 28))
        return vae_loss(recon, x, mu, logvar)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, opt, loss_fn)
        def batch_loss(m, batch, dev):
            x = batch[0].to(dev).view(batch[0].size(0), -1)
            recon, mu, logvar = m(x.view(x.size(0), 1, 28, 28))
            return vae_loss(recon, x, mu, logvar)
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
        model_hp={"input_size": 784, "latent_dim": args.latent_dim},
    )


if __name__ == "__main__":
    main()
