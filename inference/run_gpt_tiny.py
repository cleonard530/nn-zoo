"""Run GPT tiny inference: next-token prediction on a short sequence."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_synthetic_sequence
from models.gpt_tiny import GPTTiny
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/gpt_tiny/best.pt")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = get_device(use_cuda=not args.no_cuda)

    model = GPTTiny(
        vocab_size=args.vocab_size,
        max_len=args.seq_len,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
    )
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    ds = get_synthetic_sequence(num_samples=args.num_samples, seq_len=args.seq_len, vocab_size=args.vocab_size, target_mode="next_token", seed=99)
    for i in range(args.num_samples):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        pred_next = logits[0, -1].argmax().item()
        print(f"Sample {i}: last target={y[-1].item()}, predicted next={pred_next}")


if __name__ == "__main__":
    main()
