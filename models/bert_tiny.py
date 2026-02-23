"""Tiny BERT-style encoder (encoder-only transformer) for classification/embedding."""

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.ff(self.ln2(x))
        return x


class BertTiny(nn.Module):
    """
    Tiny encoder-only transformer (BERT-style). Uses [CLS] or mean pooling for classification.
    Input: (B, seq_len) token ids. Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        max_len: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
        pool: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.max_len = max_len

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = x.shape
        x = self.embed(x)
        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            T = T + 1
            if mask is not None:
                mask = torch.cat([torch.ones(B, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)
        x = x + self.pos_embed[:, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        if self.pool == "cls":
            pooled = x[:, 0]
        else:
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                pooled = x.sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            else:
                pooled = x.mean(1)
        return self.head(pooled)
