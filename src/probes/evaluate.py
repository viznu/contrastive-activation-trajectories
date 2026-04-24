"""Train three baseline probes on labeled trajectories and compare.

The single-layer probe sweeps all layers internally and picks the best by
cross-validation on the train set.

The contrastive-encoder probe is added separately once the encoder is
available; see src/encoder/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from src.probes.probes import (
    SingleLayerProbe, AllLayersConcatProbe, TransformerOverLayersProbe,
)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def pick_best_layer(traj_train, y_train, L: int, cv: int = 5) -> tuple[int, list[float]]:
    """Cross-validated AUROC sweep over layers on the train set."""
    scores_per_layer = []
    for layer in range(L):
        X = traj_train[:, layer, :].cpu().numpy()
        y = y_train.cpu().numpy()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            p = SingleLayerProbe(layer=layer)
            p.fit(traj_train[tr_idx], y_train[tr_idx])
            s = p.predict_score(traj_train[va_idx])
            aucs.append(roc_auc_score(y_train[va_idx].cpu().numpy(), s))
        scores_per_layer.append(float(np.mean(aucs)))
    best = int(np.argmax(scores_per_layer))
    return best, scores_per_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--transformer_epochs", type=int, default=30)
    p.add_argument("--group_field", default=None,
                   help="If set (e.g. 'fact_ids'), split by group so paired examples stay together.")
    args = p.parse_args()

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    N, L, d = traj.shape
    print(f"loaded {args.data}: N={N}  L={L}  d={d}  "
          f"pos={int(labels.sum())} neg={int((1-labels).sum())}")

    idx = np.arange(N)
    if args.group_field and args.group_field in blob:
        groups = blob[args.group_field].numpy() if torch.is_tensor(blob[args.group_field]) else np.asarray(blob[args.group_field])
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
        tr_idx, te_idx = next(splitter.split(idx, labels.numpy(), groups))
        print(f"group-aware split by '{args.group_field}' ({len(np.unique(groups))} groups)")
    else:
        tr_idx, te_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=labels.numpy(), random_state=args.seed,
        )
    Xtr, ytr = traj[torch.tensor(tr_idx)], labels[torch.tensor(tr_idx)]
    Xte, yte = traj[torch.tensor(te_idx)], labels[torch.tensor(te_idx)]
    print(f"train={len(tr_idx)}  test={len(te_idx)}")

    results = {}

    print("\n[1/3] single-layer probe — CV layer sweep on train")
    best_layer, per_layer_auc = pick_best_layer(Xtr, ytr, L)
    print(f"  best layer={best_layer}  cv_auc={per_layer_auc[best_layer]:.4f}")
    probe1 = SingleLayerProbe(layer=best_layer)
    probe1.fit(Xtr, ytr)
    s1 = probe1.predict_score(Xte)
    results["single_layer"] = {
        "best_layer": best_layer,
        "per_layer_cv_auc": per_layer_auc,
        "test_auroc": float(roc_auc_score(yte.numpy(), s1)),
        "test_recall_at_1pct_fpr": recall_at_fpr(yte.numpy(), s1, 0.01),
    }

    print("\n[2/3] all-layers-concat probe")
    probe2 = AllLayersConcatProbe()
    probe2.fit(Xtr, ytr)
    s2 = probe2.predict_score(Xte)
    results["all_layers_concat"] = {
        "test_auroc": float(roc_auc_score(yte.numpy(), s2)),
        "test_recall_at_1pct_fpr": recall_at_fpr(yte.numpy(), s2, 0.01),
    }

    print("\n[3/3] supervised transformer-over-layers")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    probe3 = TransformerOverLayersProbe(d_in=d, epochs=args.transformer_epochs, device=device)
    probe3.fit(Xtr, ytr)
    s3 = probe3.predict_score(Xte)
    results["transformer_over_layers"] = {
        "test_auroc": float(roc_auc_score(yte.numpy(), s3)),
        "test_recall_at_1pct_fpr": recall_at_fpr(yte.numpy(), s3, 0.01),
    }

    print("\n" + "=" * 60)
    print(f"{'probe':<30s} {'AUROC':>8s} {'recall@1%FPR':>14s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<30s} {r['test_auroc']:>8.4f} {r['test_recall_at_1pct_fpr']:>14.4f}")
    print("=" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data,
            "N_train": len(tr_idx),
            "N_test": len(te_idx),
            "L": L, "d": d,
            "results": results,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
