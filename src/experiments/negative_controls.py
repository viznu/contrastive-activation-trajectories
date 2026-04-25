"""Tier-1 negative controls for the contrastive-pretraining premise.

Premise being tested: "Contrastive pretraining on behavior-paired activations
produces label-efficient, transferable behavioral-state representations."

Five conditions, all evaluated with the same fact-grouped split and the same
downstream linear probe:

  reference        : our contrastive encoder trained with the real
                     honest/deceptive pair labels (full method).
  pca              : passive compression — PCA(256) on flattened trajectory,
                     no training, then L2 logistic on the projection.
                     If this matches reference, contrastive is not doing
                     anything special — any compression works.
  random_encoder   : initialize the TrajectoryEncoder, do NOT train it,
                     then probe its random-init pooled output.
                     Establishes the floor without any learned representation.
  shuffled_labels  : same encoder + same InfoNCE training loop, but training
                     labels shuffled within the train set. Pair structure
                     destroyed; the encoder is now trained against noise.
                     Should collapse to chance if the pair structure is
                     load-bearing.
  wrong_attribute  : pretrain InfoNCE with a pseudo-label orthogonal to
                     honest/deceptive (we use length-above-median of the fact
                     text). Probe with TRUE honest/deceptive labels at
                     downstream. Tests whether the contrastive objective is
                     finding behavior specifically or any binary structure.

Output: results/negative_controls.json with full-N AUROC + recall@1%FPR per
condition. The reference number should already be the headline (~0.99). The
controls should be much lower; how much lower is the actual measurement.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.encoder.model import TrajectoryEncoder, pool_latents


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


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


def train_encoder(Xtr: torch.Tensor, ytr_for_supcon: torch.Tensor,
                  d_in: int, d_model: int, max_L: int, layer_idx: int,
                  device: str, epochs: int, lr: float, wd: float,
                  temperature: float, seed: int):
    torch.manual_seed(seed)
    enc = TrajectoryEncoder(d_in=d_in, d_model=d_model, num_layers=4,
                            num_heads=4, max_L=max_L).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X = Xtr.to(device)
    y = ytr_for_supcon.to(device)
    for ep in range(epochs):
        enc.train()
        z = enc(X)
        pooled = pool_latents(z, mode="layer", layer_idx=layer_idx)
        loss = supcon_loss(pooled, y, temperature=temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
    enc.eval()
    return enc


@torch.no_grad()
def probe_features_from_encoder(enc: TrajectoryEncoder, X: torch.Tensor,
                                layer_idx: int, device: str) -> np.ndarray:
    z = enc(X.to(device))
    return pool_latents(z, mode="layer", layer_idx=layer_idx).cpu().numpy()


def fit_logistic_and_score(Xtr_feat: np.ndarray, ytr: np.ndarray,
                           Xte_feat: np.ndarray, yte: np.ndarray,
                           C: float = 0.1, max_iter: int = 2000):
    scaler = StandardScaler().fit(Xtr_feat)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(scaler.transform(Xtr_feat), ytr)
    scores = clf.decision_function(scaler.transform(Xte_feat))
    auc = float(roc_auc_score(yte, scores))
    rec = recall_at_fpr(yte, scores, 0.01)
    return auc, rec, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/instructed_pairs.pt")
    p.add_argument("--output", default="results/negative_controls.json")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layer_idx", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--shuffle_seed", type=int, default=42)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    fact_ids = blob["fact_ids"].numpy()
    facts = blob["facts"]
    N, L, d = traj.shape
    print(f"data: N={N} L={L} d={d}  pos={int(labels.sum())} neg={int((1-labels).sum())}")

    mean_per_layer = traj.mean(dim=0, keepdim=True)
    std_per_layer = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
    traj_norm = (traj - mean_per_layer) / std_per_layer

    idx = np.arange(N)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.split_seed)
    tr_idx, te_idx = next(splitter.split(idx, labels.numpy(), fact_ids))
    Xtr = traj_norm[torch.tensor(tr_idx)]
    ytr = labels[torch.tensor(tr_idx)]
    Xte = traj_norm[torch.tensor(te_idx)]
    yte = labels[torch.tensor(te_idx)]
    print(f"train={len(tr_idx)}  test={len(te_idx)}")

    results = {}

    # 1) reference: real labels
    print("\n[reference] real-label contrastive (our headline method)")
    enc_ref = train_encoder(
        Xtr, ytr, d_in=d, d_model=args.d_model, max_L=max(L, 64),
        layer_idx=args.layer_idx, device=device, epochs=args.epochs,
        lr=args.lr, wd=args.wd, temperature=args.temperature, seed=args.split_seed,
    )
    feat_tr = probe_features_from_encoder(enc_ref, Xtr, args.layer_idx, device)
    feat_te = probe_features_from_encoder(enc_ref, Xte, args.layer_idx, device)
    auc, rec, _ = fit_logistic_and_score(feat_tr, ytr.numpy(), feat_te, yte.numpy())
    results["reference"] = {"auroc": auc, "recall_at_1pct_fpr": rec,
                            "description": "real-label contrastive"}
    print(f"  AUROC={auc:.4f}  recall@1%={rec:.4f}")

    # 2) PCA passive compression
    print("\n[pca] PCA(256) on flattened trajectory + L2 logistic, no training")
    Xtr_flat = Xtr.reshape(len(Xtr), -1).numpy()
    Xte_flat = Xte.reshape(len(Xte), -1).numpy()
    pca = PCA(n_components=args.d_model).fit(Xtr_flat)
    auc, rec, _ = fit_logistic_and_score(pca.transform(Xtr_flat), ytr.numpy(),
                                          pca.transform(Xte_flat), yte.numpy())
    results["pca"] = {"auroc": auc, "recall_at_1pct_fpr": rec,
                      "description": "PCA(256) on flat trajectory, no training"}
    print(f"  AUROC={auc:.4f}  recall@1%={rec:.4f}")

    # 3) random encoder
    print("\n[random_encoder] random-init TrajectoryEncoder, no training")
    torch.manual_seed(args.split_seed + 100)
    enc_rand = TrajectoryEncoder(d_in=d, d_model=args.d_model, num_layers=4,
                                 num_heads=4, max_L=max(L, 64)).to(device)
    enc_rand.eval()
    feat_tr = probe_features_from_encoder(enc_rand, Xtr, args.layer_idx, device)
    feat_te = probe_features_from_encoder(enc_rand, Xte, args.layer_idx, device)
    auc, rec, _ = fit_logistic_and_score(feat_tr, ytr.numpy(), feat_te, yte.numpy())
    results["random_encoder"] = {"auroc": auc, "recall_at_1pct_fpr": rec,
                                  "description": "random-init encoder, no training"}
    print(f"  AUROC={auc:.4f}  recall@1%={rec:.4f}")

    # 4) shuffled labels
    print("\n[shuffled_labels] same encoder + same loop, training labels shuffled")
    rng = np.random.default_rng(args.shuffle_seed)
    ytr_shuffled_np = rng.permutation(ytr.numpy())
    ytr_shuffled = torch.tensor(ytr_shuffled_np)
    enc_shuf = train_encoder(
        Xtr, ytr_shuffled, d_in=d, d_model=args.d_model, max_L=max(L, 64),
        layer_idx=args.layer_idx, device=device, epochs=args.epochs,
        lr=args.lr, wd=args.wd, temperature=args.temperature, seed=args.split_seed,
    )
    feat_tr = probe_features_from_encoder(enc_shuf, Xtr, args.layer_idx, device)
    feat_te = probe_features_from_encoder(enc_shuf, Xte, args.layer_idx, device)
    auc, rec, _ = fit_logistic_and_score(feat_tr, ytr.numpy(), feat_te, yte.numpy())
    results["shuffled_labels"] = {"auroc": auc, "recall_at_1pct_fpr": rec,
                                   "description": "contrastive on shuffled labels"}
    print(f"  AUROC={auc:.4f}  recall@1%={rec:.4f}")

    # 5) wrong-attribute contrastive: pseudo-label = (fact length > median)
    print("\n[wrong_attribute] contrastive on length-based pseudo-labels (orthogonal to behavior)")
    fact_len = np.array([len(facts[fid]) for fid in fact_ids])
    median_len = np.median(fact_len)
    length_label = (fact_len > median_len).astype(np.int64)
    ytr_wrong = torch.tensor(length_label[tr_idx], dtype=torch.long)
    enc_wrong = train_encoder(
        Xtr, ytr_wrong, d_in=d, d_model=args.d_model, max_L=max(L, 64),
        layer_idx=args.layer_idx, device=device, epochs=args.epochs,
        lr=args.lr, wd=args.wd, temperature=args.temperature, seed=args.split_seed,
    )
    feat_tr = probe_features_from_encoder(enc_wrong, Xtr, args.layer_idx, device)
    feat_te = probe_features_from_encoder(enc_wrong, Xte, args.layer_idx, device)
    auc, rec, _ = fit_logistic_and_score(feat_tr, ytr.numpy(), feat_te, yte.numpy())
    # Verify the wrong-attribute label was actually orthogonal (~0.5 correlation)
    behavior_vs_length_corr = np.corrcoef(ytr.numpy(), ytr_wrong.numpy())[0, 1]
    results["wrong_attribute"] = {"auroc": auc, "recall_at_1pct_fpr": rec,
                                   "description": "contrastive on fact-length pseudo-labels",
                                   "behavior_vs_pseudo_correlation": float(behavior_vs_length_corr)}
    print(f"  AUROC={auc:.4f}  recall@1%={rec:.4f}  "
          f"(behavior vs length corr = {behavior_vs_length_corr:.3f})")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'condition':<22s} {'AUROC':>8s}  {'recall@1%FPR':>14s}  description")
    print("-" * 70)
    for k in ["reference", "pca", "random_encoder", "shuffled_labels", "wrong_attribute"]:
        r = results[k]
        print(f"{k:<22s} {r['auroc']:>8.4f}  {r['recall_at_1pct_fpr']:>14.4f}  {r['description']}")
    print("=" * 70)

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
