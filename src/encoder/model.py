"""Trajectory encoder: a small transformer over a target LLM's per-layer
residual-stream trajectory.

Input:  (B, L, d_in)  — B activation trajectories, each with L layer slots
                         and d_in dims matching the target LLM's hidden size.
Output: (B, L, d_model) — per-slot encoded latents.

Architecture:
  Linear(d_in -> d_model) + learned per-layer position embedding +
  TransformerEncoder (pre-LN) over the L-slot sequence.

No masking, no EMA target, no reconstruction loss — the full training signal
comes from a pair-aware contrastive objective over pooled outputs (see
src/encoder/train.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_L: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.max_L = max_L

        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_in) -> (B, L, d_model)."""
        B, L, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:L]
        return self.encoder(h)


def pool_latents(
    z: torch.Tensor,
    mode: str = "mean",
    layer_idx: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> torch.Tensor:
    """(B, L, d_model) -> (B, d_model). Supports mean / last / mid / layer / window."""
    if mode == "mean":
        return z.mean(dim=1)
    if mode == "last":
        return z[:, -1, :]
    if mode == "mid":
        return z[:, z.size(1) // 2, :]
    if mode == "layer":
        if layer_idx is None:
            raise ValueError("pool='layer' requires layer_idx")
        return z[:, layer_idx, :]
    if mode == "window":
        if window_start is None or window_end is None:
            raise ValueError("pool='window' requires window_start and window_end")
        return z[:, window_start:window_end, :].mean(dim=1)
    raise ValueError(f"unknown pool mode: {mode}")
