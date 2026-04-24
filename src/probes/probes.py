"""Three baseline probe families for misalignment classification from raw
transformer activation trajectories.

Common interface:
  fit(traj_train, y_train) -> None
  predict_score(traj) -> np.ndarray of shape (N,)  (higher = more "misaligned")

traj is torch.Tensor of shape (N, L, d); y is torch.Tensor of shape (N,) with {0, 1}.

(The ContrastiveEncoderProbe that uses a frozen pretrained trajectory encoder
is added separately in src/encoder/, so baselines stand alone here.)
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
