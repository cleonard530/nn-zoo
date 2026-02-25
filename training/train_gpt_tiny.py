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
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_epoch_checkpoint, save_training_metadata, train_epoch, validation_loss


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, include_data_dir=False, default_save_dir="./weights/gpt_tiny", default_epochs=10, default_batch_size=64, default_lr=1e-4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
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

    def loss_fn(m, batch, dev):
        x, y = batch[0].to(dev), batch[1].to(dev)
        logits = m(x)
        return criterion(logits.view(-1, args.vocab_size), y.view(-1))

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, opt, loss_fn)
        def batch_loss(m, batch, dev):
            x, y = batch[0].to(dev), batch[1].to(dev)
            logits = m(x)
            return criterion(logits.view(-1, args.vocab_size), y.view(-1))
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
        model_hp={"vocab_size": args.vocab_size, "max_len": args.seq_len, "d_model": 128, "num_heads": 4, "num_layers": 2},
    )


if __name__ == "__main__":
    main()
