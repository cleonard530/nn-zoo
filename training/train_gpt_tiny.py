"""Train tiny GPT on synthetic next-token prediction."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_synthetic_sequence
from models.gpt_tiny import GPTTiny
from utils import get_device, get_run_id, log_epoch, save_checkpoint, save_training_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--save_dir", type=str, default="./weights/gpt_tiny")
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_synthetic_sequence(
        num_samples=8000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode="next_token", seed=42,
    )
    val_ds = get_synthetic_sequence(
        num_samples=2000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode="next_token", seed=43,
    )
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GPTTiny(
        vocab_size=args.vocab_size,
        max_len=args.seq_len,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits.view(-1, args.vocab_size), y.view(-1)).item()
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
        model_hp={"vocab_size": args.vocab_size, "max_len": args.seq_len, "d_model": 128, "num_heads": 4, "num_layers": 2},
    )


if __name__ == "__main__":
    main()
