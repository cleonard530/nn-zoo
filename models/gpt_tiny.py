"""Tiny GPT-style decoder-only transformer for next-token prediction."""

import math
from typing import Optional

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTTiny(nn.Module):
    """
    Tiny decoder-only transformer (GPT-style). Next-token prediction.
    Input: (B, seq_len) token ids. Output: (B, seq_len, vocab_size) logits.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        max_len: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, max_len, d_ff, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: output projection shares embed weights
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)
