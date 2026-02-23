"""LSTM for sequence classification or next-step prediction."""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM with optional classification head (last hidden or mean pooling).
    For classification: output (B, num_classes). For next-token: output (B, seq_len, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
        mode: str = "classify",  # "classify" or "predict"
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        if mode == "classify":
            self.head = nn.Linear(hidden_size, num_classes)
        else:
            self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        x = self.embed(x)
        out, (h_n, _) = self.lstm(x)
        out = self.drop(out)
        if self.mode == "classify":
            pooled = h_n[-1]
            return self.head(pooled)
        return self.head(out)
