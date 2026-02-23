"""Run LSTM inference on a few synthetic sequences."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_synthetic_sequence
from models.lstm import LSTM
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/lstm/best.pt")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--mode", type=str, default="classify", choices=["classify", "predict"])
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = get_device(use_cuda=not args.no_cuda)

    model = LSTM(
        vocab_size=args.vocab_size,
        embed_dim=64,
        hidden_size=128,
        num_layers=2,
        num_classes=args.num_classes,
        mode=args.mode,
    )
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    target_mode = "class" if args.mode == "classify" else "next_token"
    ds = get_synthetic_sequence(num_samples=args.num_samples, seq_len=args.seq_len, vocab_size=args.vocab_size, target_mode=target_mode, num_classes=args.num_classes, seed=99)
    for i in range(args.num_samples):
        x, y_true = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        if args.mode == "classify":
            pred = out.argmax(1).item()
            print(f"Sample {i}: true={y_true.item()}, pred={pred}")
        else:
            pred_next = out[0, -1].argmax().item()
            print(f"Sample {i}: last target={y_true[-1].item()}, predicted next={pred_next}")


if __name__ == "__main__":
    main()
