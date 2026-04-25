"""Probe families for misalignment classification from transformer activation
trajectories.

Common interface:
  fit(traj_train, y_train) -> None
  predict_score(traj) -> np.ndarray of shape (N,)  (higher = more "misaligned")

traj is torch.Tensor of shape (N, L, d); y is torch.Tensor of shape (N,) with {0, 1}.

Three baselines operate on raw activations. ContrastiveEncoderProbe uses a
frozen pretrained trajectory encoder (src/encoder/model.py) as a feature
extractor and fits a logistic probe on the pooled latent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class SingleLayerProbe:
    """Logistic regression on one chosen layer's activation vector."""
    def __init__(self, layer: int, C: float = 0.1, max_iter: int = 2000):
        self.layer = layer
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = _as_np(traj[:, self.layer, :])
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(_as_np(traj[:, self.layer, :]))
        return self.clf.decision_function(X)


class AllLayersConcatProbe:
    """Logistic regression on flattened (L*d,) per-sample concatenation."""
    def __init__(self, C: float = 0.01, max_iter: int = 5000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = _as_np(traj.flatten(1))
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(_as_np(traj.flatten(1)))
        return self.clf.decision_function(X)


class TransformerOverLayersProbe:
    """Small transformer over the L-layer sequence + classification head.

    Supervised, trained with BCE. Layer inputs are z-normalized per layer.
    Serves as a "what if you trained a supervised transformer directly on
    trajectories" baseline.
    """
    def __init__(self, d_in: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 max_L: int = 64, epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
                 wd: float = 0.05, device: str = "cpu", seed: int = 42):
        self.device = device
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        torch.manual_seed(seed)
        self.model = _TransformerClassifier(d_in, d_model, n_heads, n_layers, max_L).to(device)
        self.mean = None
        self.std = None

    def fit(self, traj, y, traj_val=None, y_val=None):
        self.mean = traj.mean(dim=0, keepdim=True)
        self.std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
        X = ((traj - self.mean) / self.std).to(self.device)
        yt = y.float().to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        N = X.size(0)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.model(X[idx])
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

    def predict_score(self, traj):
        self.model.eval()
        X = ((traj - self.mean) / self.std).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
        return logits.detach().cpu().numpy()


class _TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_model, n_heads, n_layers, max_L):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(max_L, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.proj(x) + self.pos[:L]
        z = self.enc(h)
        return self.head(z.mean(dim=1)).squeeze(-1)


class ContrastiveEncoderProbe:
    """Frozen pretrained TrajectoryEncoder + L2-logistic probe on pooled latents."""
    def __init__(self, ckpt_path: str, device: str = "cpu", C: float = 0.1,
                 max_iter: int = 2000, pool: str = "layer", layer_idx: int | None = 24,
                 window_start: int | None = None, window_end: int | None = None):
        from src.encoder.model import TrajectoryEncoder, pool_latents
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.device = device
        self.encoder = TrajectoryEncoder(
            d_in=ckpt["d_in"],
            d_model=cfg["d_model"],
            num_layers=cfg["enc_layers"],
            num_heads=cfg["heads"],
            max_L=max(ckpt["L"], 64),
        ).to(device)
        self.encoder.load_state_dict(ckpt["state_dict"])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.norm_mean = ckpt["norm_mean"].to("cpu")
        self.norm_std = ckpt["norm_std"].to("cpu")
        self.pool = pool
        self.layer_idx = layer_idx
        self.window_start = window_start
        self.window_end = window_end
        self._pool_fn = pool_latents
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    @torch.no_grad()
    def _featurize(self, traj):
        x = ((traj.cpu() - self.norm_mean) / self.norm_std).to(self.device)
        z = self.encoder(x)
        feat = self._pool_fn(
            z, mode=self.pool, layer_idx=self.layer_idx,
            window_start=self.window_start, window_end=self.window_end,
        )
        return feat.cpu().numpy()

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = self._featurize(traj)
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(self._featurize(traj))
        return self.clf.decision_function(X)
