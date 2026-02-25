"""Train LSTM on synthetic sequence (classification or next-token)."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_synthetic_sequence
from models.lstm import LSTM
from utils import get_device, get_run_id, log_epoch, save_checkpoint, save_training_metadata, validation_accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--mode", type=str, default="classify", choices=["classify", "predict"])
    p.add_argument("--save_dir", type=str, default="./weights/lstm")
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA if available")
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    target_mode = "class" if args.mode == "classify" else "next_token"
    train_ds = get_synthetic_sequence(
        num_samples=8000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode=target_mode, num_classes=args.num_classes, seed=42,
    )
    val_ds = get_synthetic_sequence(
        num_samples=2000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode=target_mode, num_classes=args.num_classes, seed=43,
    )
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTM(
        vocab_size=args.vocab_size,
        embed_dim=64,
        hidden_size=128,
        num_layers=2,
        num_classes=args.num_classes,
        mode=args.mode,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.mode == "classify":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_metric = float("inf") if args.mode == "predict" else 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            if args.mode == "classify":
                loss = criterion(out, y)
            else:
                loss = criterion(out.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        if args.mode == "classify":
            val_metric = validation_accuracy(model, val_loader, device)
            log_epoch(epoch, {"train_loss": train_loss, "val_acc": val_metric})
            is_best = val_metric > best_metric
            if is_best:
                best_metric = val_metric
        else:
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    val_loss += criterion(out.view(-1, args.vocab_size), y.view(-1)).item()
            val_metric = val_loss / len(val_loader)
            log_epoch(epoch, {"train_loss": train_loss, "val_loss": val_metric})
            is_best = val_metric < best_metric
            if is_best:
                best_metric = val_metric

        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metric": val_metric},
            Path(args.save_dir) / f"last_{run_id}.pt",
            is_best=is_best,
            best_path=Path(args.save_dir) / f"best_{run_id}.pt",
        )

    metric_name = "val_acc" if args.mode == "classify" else "val_loss"
    save_training_metadata(
        args.save_dir,
        run_id,
        args,
        {"epochs": args.epochs, f"best_{metric_name}": best_metric, f"final_{metric_name}": val_metric},
        model_hp={"vocab_size": args.vocab_size, "embed_dim": 64, "hidden_size": 128, "num_layers": 2, "num_classes": args.num_classes, "mode": args.mode},
    )


if __name__ == "__main__":
    main()
