"""Train GAN on MNIST."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_mnist
from models.gan import GAN
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_checkpoint, save_training_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, default_save_dir="./weights/gan", default_epochs=30)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--latent_dim", type=int, default=64)
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_mnist(args.data_dir, train=True)
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)

    gan = GAN(latent_dim=args.latent_dim, image_size=784).to(device)
    G, D = gan.generator, gan.discriminator
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d)
    real_label, fake_label = 1.0, 0.0

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        loss_d_sum, loss_g_sum = 0.0, 0.0
        n_batches = 0
        for x, _ in train_loader:
            x = x.to(device).view(x.size(0), -1)
            B = x.size(0)
            # Train D: real
            opt_d.zero_grad()
            pred_real = D(x)
            loss_d_real = nn.BCEWithLogitsLoss()(pred_real, torch.full((B,), real_label, device=device, dtype=x.dtype))
            # Train D: fake
            z = torch.randn(B, args.latent_dim, device=device)
            fake = G(z)
            pred_fake = D(fake.detach())
            loss_d_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.full((B,), fake_label, device=device, dtype=x.dtype))
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_d.step()
            loss_d_sum += loss_d.item()

            # Train G
            opt_g.zero_grad()
            pred_fake = D(fake)
            loss_g = nn.BCEWithLogitsLoss()(pred_fake, torch.full((B,), real_label, device=device, dtype=x.dtype))
            loss_g.backward()
            opt_g.step()
            loss_g_sum += loss_g.item()
            n_batches += 1

        loss_d_avg = loss_d_sum / n_batches
        loss_g_avg = loss_g_sum / n_batches
        log_epoch(epoch, {"loss_d": loss_d_avg, "loss_g": loss_g_avg})
        save_checkpoint(
            {
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
            },
            Path(args.save_dir) / f"last_{run_id}.pt",
            is_best=False,
            best_path=None,
        )
        if epoch == 1 or epoch % 10 == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                },
                Path(args.save_dir) / f"best_{run_id}.pt",
                is_best=False,
                best_path=None,
            )

    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {
            "epochs": args.epochs,
            "final_loss_d": loss_d_avg,
            "final_loss_g": loss_g_avg,
        },
        model_hp={"latent_dim": args.latent_dim, "image_size": 784},
    )


if __name__ == "__main__":
    main()
