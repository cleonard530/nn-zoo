"""Run BERT tiny inference on a few synthetic sequences."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import get_synthetic_sequence
from models.bert_tiny import BertTiny
from utils import get_device, load_model_weights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/bert_tiny/best.pt")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = get_device(use_cuda=not args.no_cuda)

    model = BertTiny(
        vocab_size=args.vocab_size,
        max_len=args.seq_len,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        num_classes=args.num_classes,
    )
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    ds = get_synthetic_sequence(num_samples=args.num_samples, seq_len=args.seq_len, vocab_size=args.vocab_size, target_mode="class", num_classes=args.num_classes, seed=99)
    for i in range(args.num_samples):
        x, y_true = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = logits.argmax(1).item()
        print(f"Sample {i}: true={y_true.item()}, pred={pred}")


if __name__ == "__main__":
    main()
