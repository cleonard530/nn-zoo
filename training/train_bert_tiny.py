"""Train tiny BERT on synthetic sequence classification."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_dataloader, get_synthetic_sequence
from models.bert_tiny import BertTiny
from utils import add_common_train_args, get_device, get_run_id, log_epoch, save_epoch_checkpoint, save_training_metadata, train_epoch, validation_accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    add_common_train_args(p, include_data_dir=False, default_save_dir="./weights/bert_tiny", default_epochs=10, default_batch_size=64, default_lr=1e-4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    args = p.parse_args()
    device = get_device(use_cuda=args.use_cuda)
    run_id = get_run_id()

    train_ds = get_synthetic_sequence(
        num_samples=8000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode="class", num_classes=args.num_classes, seed=42,
    )
    val_ds = get_synthetic_sequence(
        num_samples=2000, seq_len=args.seq_len, vocab_size=args.vocab_size,
        target_mode="class", num_classes=args.num_classes, seed=43,
    )
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BertTiny(
        vocab_size=args.vocab_size,
        max_len=args.seq_len,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=args.num_classes,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        model_hp={"vocab_size": args.vocab_size, "max_len": args.seq_len, "d_model": 128, "num_heads": 4, "num_layers": 2, "num_classes": args.num_classes},
    )


if __name__ == "__main__":
    main()
