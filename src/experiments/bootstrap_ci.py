"""Bootstrap 95% CIs for the headline AUROC + recall@1%FPR numbers.

For each of the four probes evaluated in src/probes/evaluate.py, we re-run
the probe inference on the test set, then bootstrap-resample the test set
1000 times to compute confidence intervals on AUROC and recall@1%FPR.

Test sets in this project are small (124 paired examples for IP, 200 for
the refusal datasets); CIs are essential for honest reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from src.probes.probes import (
    SingleLayerProbe, AllLayersConcatProbe, TransformerOverLayersProbe,
    ContrastiveEncoderProbe,
)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def bootstrap_metric(y_true: np.ndarray, scores: np.ndarray, metric_fn,
                     n_bootstrap: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Reject draws where one class is missing — AUROC is undefined.
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], scores[idx]))
    arr = np.asarray(vals)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/instructed_pairs.pt")
    p.add_argument("--output", default="results/bootstrap_cis.json")
    p.add_argument("--encoder_ckpt", default="results/encoder_ip_infonce.pt")
    p.add_argument("--encoder_pool", default="layer")
    p.add_argument("--encoder_layer_idx", type=int, default=24)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single_layer", type=int, default=24,
                   help="Layer for SingleLayerProbe (no CV at probe-time here).")
    p.add_argument("--transformer_epochs", type=int, default=15)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    fact_ids = blob["fact_ids"].numpy()
    N, L, d = traj.shape

    idx = np.arange(N)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, labels.numpy(), fact_ids))
    Xtr = traj[torch.tensor(tr_idx)]
    ytr = labels[torch.tensor(tr_idx)]
    Xte = traj[torch.tensor(te_idx)]
    yte = labels[torch.tensor(te_idx)].numpy()
    print(f"train={len(tr_idx)}  test={len(te_idx)}  bootstrap_n={args.n_bootstrap}")

    probes = [
        ("single_layer", SingleLayerProbe(layer=args.single_layer)),
        ("all_layers_concat", AllLayersConcatProbe()),
        ("transformer_over_layers",
         TransformerOverLayersProbe(d_in=d, epochs=args.transformer_epochs, device=device)),
        ("contrastive_encoder",
         ContrastiveEncoderProbe(args.encoder_ckpt, device=device,
                                 pool=args.encoder_pool, layer_idx=args.encoder_layer_idx)),
    ]

    out = {"data": args.data, "n_bootstrap": args.n_bootstrap, "results": {}}

    print(f"\n{'probe':<28s}  {'AUROC':>22s}  {'recall@1%FPR':>22s}")
    print("-" * 76)
    for name, probe in probes:
        probe.fit(Xtr, ytr)
        scores = probe.predict_score(Xte)

        auc_pt = float(roc_auc_score(yte, scores))
        rec_pt = recall_at_fpr(yte, scores, 0.01)

        auc_mean, auc_lo, auc_hi = bootstrap_metric(yte, scores, roc_auc_score,
                                                     args.n_bootstrap, seed=args.seed)
        rec_mean, rec_lo, rec_hi = bootstrap_metric(
            yte, scores, lambda y, s: recall_at_fpr(y, s, 0.01),
            args.n_bootstrap, seed=args.seed,
        )
        out["results"][name] = {
            "test_auroc": auc_pt,
            "test_recall_at_1pct_fpr": rec_pt,
            "auroc_bootstrap_mean": auc_mean,
            "auroc_bootstrap_95ci_low": auc_lo,
            "auroc_bootstrap_95ci_high": auc_hi,
            "recall_bootstrap_mean": rec_mean,
            "recall_bootstrap_95ci_low": rec_lo,
            "recall_bootstrap_95ci_high": rec_hi,
        }
        print(f"{name:<28s}  {auc_pt:.4f} [{auc_lo:.4f},{auc_hi:.4f}]  "
              f"{rec_pt:.4f} [{rec_lo:.4f},{rec_hi:.4f}]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
