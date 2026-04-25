"""Fair contrastive baselines: ablate architecture while keeping SupCon loss fixed.

Four encoder variants, all trained with identical SupCon/InfoNCE pretraining
on the IP-true fact-grouped train split, then probed with the same linear
classifier:

  (a) linear_single   — Linear(d_in, d_model) on a single chosen layer
  (b) linear_concat   — Linear(L*d_in, d_model) on flattened trajectory
  (c) mlp_concat      — two-layer MLP on flattened trajectory
  (d) transformer     — our TrajectoryEncoder with attention over layer slots

All share d_model=256, 40 epochs, T=0.07, AdamW lr=1e-3, wd=0.05.

Reports:
  - Parameter count per variant.
  - AUROC + recall@1%FPR on IP test.
  - Few-shot sweep (5 seeds per N_train).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.encoder.model import TrajectoryEncoder, pool_latents
from src.encoder.baselines import (
    LinearSingleLayerEncoder, LinearConcatEncoder, MLPConcatEncoder,
)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def build_encoder(kind: str, d_in: int, L: int, d_model: int, layer_idx: int, device: str):
    if kind == "linear_single":
        enc = LinearSingleLayerEncoder(d_in=d_in, d_model=d_model)
        enc.input_layer = layer_idx
        def featurize(x: torch.Tensor) -> torch.Tensor:
            return enc(x)
        return enc, featurize
    if kind == "linear_concat":
        enc = LinearConcatEncoder(d_in_flat=L * d_in, d_model=d_model)
        def featurize(x: torch.Tensor) -> torch.Tensor:
            return enc(x)
        return enc, featurize
    if kind == "mlp_concat":
        enc = MLPConcatEncoder(d_in_flat=L * d_in, d_hidden=d_model, d_model=d_model)
        def featurize(x: torch.Tensor) -> torch.Tensor:
            return enc(x)
        return enc, featurize
    if kind == "transformer":
        enc = TrajectoryEncoder(d_in=d_in, d_model=d_model, num_layers=4,
                                num_heads=4, max_L=max(L, 64))
        def featurize(x: torch.Tensor) -> torch.Tensor:
            z = enc(x)
            return pool_latents(z, mode="layer", layer_idx=layer_idx)
        return enc, featurize
    raise ValueError(kind)


def supcon_loss(pooled: torch.Tensor, labels: torch.Tensor, temperature: float):
    pooled = F.normalize(pooled, dim=-1)
    B = pooled.size(0)
    sim = pooled @ pooled.T / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(self_mask, -1e9)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_label & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / n_pos
    valid = pos_mask.any(dim=1)
    return -mean_log_prob_pos[valid].mean() if valid.any() else torch.zeros((), device=sim.device)


def train_one(kind: str, Xtr: torch.Tensor, ytr: torch.Tensor,
              d_in: int, L: int, d_model: int, layer_idx: int, device: str,
              epochs: int, lr: float, wd: float, temperature: float, seed: int):
    torch.manual_seed(seed)
    enc, featurize = build_encoder(kind, d_in, L, d_model, layer_idx, device)
    enc = enc.to(device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  [{kind}] params={n_params:,}")

    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X = Xtr.to(device)
    y = ytr.to(device)
    for ep in range(epochs):
        enc.train()
        pooled = featurize(X)
        loss = supcon_loss(pooled, y, temperature=temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()

    return enc, featurize, n_params


@torch.no_grad()
def encode(featurize, X: torch.Tensor, device: str) -> np.ndarray:
    X = X.to(device)
    return featurize(X).cpu().numpy()


def probe_eval(enc_tr_feat, ytr_np, enc_te_feat, yte_np):
    scaler = StandardScaler().fit(enc_tr_feat)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(enc_tr_feat), ytr_np)
    s = clf.decision_function(scaler.transform(enc_te_feat))
    return float(roc_auc_score(yte_np, s)), recall_at_fpr(yte_np, s, 0.01)


def few_shot_curve(featurize, Xtr_pool: torch.Tensor, ytr_pool: torch.Tensor,
                   Xte: torch.Tensor, yte: torch.Tensor, device: str,
                   n_trains: list[int], n_seeds: int):
    """Encode train/test once; then sample N_train indices from the encoded train pool."""
    enc_tr = encode(featurize, Xtr_pool, device)
    enc_te = encode(featurize, Xte, device)
    y_pool = ytr_pool.numpy()
    yte_np = yte.numpy()

    out = {n: [] for n in n_trains}
    pool_idx = np.arange(len(enc_tr))
    for n in n_trains:
        if n > len(pool_idx):
            continue
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            pos = pool_idx[y_pool == 1]
            neg = pool_idx[y_pool == 0]
            n_pos = n // 2
            n_neg = n - n_pos
            pick = np.concatenate([
                rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
            ])
            scaler = StandardScaler().fit(enc_tr[pick])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(
                scaler.transform(enc_tr[pick]), y_pool[pick]
            )
            s = clf.decision_function(scaler.transform(enc_te))
            out[n].append(float(roc_auc_score(yte_np, s)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/instructed_pairs.pt")
    p.add_argument("--output", default="results/fair_contrastive_baselines.json")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layer_idx", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--kinds", nargs="+",
                   default=["linear_single", "linear_concat", "mlp_concat", "transformer"])
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_trains", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80, 160, 320])
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    fact_ids = blob["fact_ids"].numpy()
    N, L, d = traj.shape
    print(f"data: N={N} L={L} d={d}  pos={int(labels.sum())}  neg={int((1 - labels).sum())}")

    # z-normalize per layer across the dataset (same as src/encoder/train.py)
    mean = traj.mean(dim=0, keepdim=True)
    std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
    traj_norm = (traj - mean) / std

    idx = np.arange(N)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, labels.numpy(), fact_ids))
    print(f"train={len(tr_idx)}  test={len(te_idx)}")
    Xtr = traj_norm[torch.tensor(tr_idx)]
    ytr = labels[torch.tensor(tr_idx)]
    Xte = traj_norm[torch.tensor(te_idx)]
    yte = labels[torch.tensor(te_idx)]

    results = {}
    for kind in args.kinds:
        print(f"\n=== {kind} ===")
        t0 = time.time()
        enc, featurize, n_params = train_one(
            kind, Xtr, ytr, d_in=d, L=L, d_model=args.d_model,
            layer_idx=args.layer_idx, device=device,
            epochs=args.epochs, lr=args.lr, wd=args.wd,
            temperature=args.temperature, seed=args.seed,
        )
        t_train = time.time() - t0

        enc.eval()
        enc_tr_feat = encode(featurize, Xtr, device)
        enc_te_feat = encode(featurize, Xte, device)
        auc, rec = probe_eval(enc_tr_feat, ytr.numpy(), enc_te_feat, yte.numpy())
        print(f"  full-train probe: AUROC={auc:.4f}  recall@1%FPR={rec:.4f}  ({t_train:.1f}s train)")

        t0 = time.time()
        fs = few_shot_curve(
            featurize, Xtr, ytr, Xte, yte, device,
            args.n_trains, args.n_seeds,
        )
        print(f"  few-shot sweep ({time.time() - t0:.1f}s):")
        row = ""
        for n in args.n_trains:
            xs = fs[n]
            if xs:
                row += f"  N={n}: {np.mean(xs):.2f}\u00b1{np.std(xs):.2f}"
        print("   " + row)

        results[kind] = {
            "params": n_params,
            "full_train_auroc": auc,
            "full_train_recall_at_1pct_fpr": rec,
            "train_time_sec": t_train,
            "few_shot_auroc": {str(n): fs[n] for n in args.n_trains},
        }

    print("\n" + "=" * 80)
    print(f"{'kind':<18s} {'params':>12s} {'AUROC':>8s} {'recall@1%':>10s}  few-shot AUROC by N_train")
    print("-" * 80)
    for kind, r in results.items():
        row = f"{kind:<18s} {r['params']:>12,d} {r['full_train_auroc']:>8.4f} {r['full_train_recall_at_1pct_fpr']:>10.4f} "
        for n in args.n_trains:
            xs = r["few_shot_auroc"][str(n)]
            row += f" {np.mean(xs):.2f}"
        print(row)
    print("=" * 80)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data,
            "config": vars(args),
            "results": results,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
