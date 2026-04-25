"""Baseline encoder architectures for the fair-contrastive comparison.

All three output a single (B, d_model) latent vector per input, matching the
TrajectoryEncoder's pooled output shape. They are trained with the same
SupCon-style InfoNCE loss used in src/encoder/train.py; the only difference
is what architecture produces the latent.

Architectures:
  LinearSingleLayerEncoder(d_in, d_model)
      Input:  (B, d_in)    — a SINGLE layer's activation (e.g. layer 24).
      Output: (B, d_model) — Linear projection.
      Tests: does restricting to one layer already have the win?

  LinearConcatEncoder(d_in_flat, d_model)
      Input:  (B, d_in_flat) — the flattened (L*d) trajectory.
      Output: (B, d_model)   — Linear projection.
      Tests: does a single learned linear compression of the full
             trajectory suffice?

  MLPConcatEncoder(d_in_flat, d_hidden, d_model)
      Input:  (B, d_in_flat) — the flattened trajectory.
      Output: (B, d_model)   — two-layer MLP with GELU nonlinearity.
      Tests: does nonlinearity on the flat trajectory help?
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearSingleLayerEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.input_layer = None  # set by the training loop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_in) — we slice to layer `input_layer` before projection."""
        if self.input_layer is None:
            raise RuntimeError("LinearSingleLayerEncoder.input_layer must be set")
        return self.proj(x[:, self.input_layer, :])


class LinearConcatEncoder(nn.Module):
    def __init__(self, d_in_flat: int, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_in_flat, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_in) — flatten to (B, L*d_in) then project."""
        B = x.size(0)
        return self.proj(x.reshape(B, -1))


class MLPConcatEncoder(nn.Module):
    def __init__(self, d_in_flat: int, d_hidden: int = 256, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in_flat, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.mlp(x.reshape(B, -1))
