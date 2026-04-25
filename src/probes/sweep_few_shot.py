"""Few-label AUROC sweep: how many labeled examples does each probe need?

For each (N_train, probe, seed): subsample N_train labeled examples stratified
by class, fit probe, score fixed test set. Report mean +/- std across seeds.

Headline comparison: the frozen ContrastiveEncoderProbe (encoder pretrained
on labeled pairs, pool='layer') should reach near-max AUROC at very small
N_train, while raw-activation probes require many more labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.probes.probes import (
    SingleLayerProbe, AllLayersConcatProbe, TransformerOverLayersProbe,
    ContrastiveEncoderProbe,
)


def stratified_subsample(idx_pool: np.ndarray, labels: np.ndarray,
                         n: int, rng: np.random.Generator) -> np.ndarray:
    pos = idx_pool[labels[idx_pool] == 1]
    neg = idx_pool[labels[idx_pool] == 0]
    n_pos = n // 2
    n_neg = n - n_pos
    pick_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    pick_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return np.concatenate([pick_pos, pick_neg])


def build_probes(d: int, L: int, encoder_ckpt: str, single_layer: int,
                 device: str, transformer_epochs: int,
                 encoder_pool: str, encoder_layer_idx: int | None,
                 encoder_window_start: int | None,
                 encoder_window_end: int | None) -> dict:
    probes = {
        "single_layer": lambda: SingleLayerProbe(layer=single_layer),
        "all_layers_concat": lambda: AllLayersConcatProbe(),
        "transformer_over_layers": lambda: TransformerOverLayersProbe(
            d_in=d, epochs=transformer_epochs, device=device),
    }
    if encoder_ckpt:
        probes["contrastive_encoder"] = lambda: ContrastiveEncoderProbe(
            encoder_ckpt, device=device, pool=encoder_pool,
            layer_idx=encoder_layer_idx,
            window_start=encoder_window_start, window_end=encoder_window_end,
        )
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder_ckpt", default=None)
    p.add_argument("--encoder_pool", default="layer",
                   choices=["mean", "last", "mid", "layer", "window"])
    p.add_argument("--encoder_layer_idx", type=int, default=24)
    p.add_argument("--encoder_window_start", type=int, default=None)
    p.add_argument("--encoder_window_end", type=int, default=None)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--single_layer", type=int, default=24,
                   help="Fixed layer for SingleLayerProbe (no CV at each N_train).")
    p.add_argument("--transformer_epochs", type=int, default=20)
    p.add_argument("--n_trains", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80, 160, 320])
    p.add_argument("--group_field", default=None)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    N, L, d = traj.shape
    y_np = labels.numpy()
    print(f"N={N} L={L} d={d}  pos={int(labels.sum())}  neg={int((1-labels).sum())}")

    idx = np.arange(N)
    if args.group_field and args.group_field in blob:
        groups = blob[args.group_field].numpy() if torch.is_tensor(blob[args.group_field]) else np.asarray(blob[args.group_field])
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=42)
        tr_pool, te_idx = next(splitter.split(idx, y_np, groups))
        print(f"group-aware split by '{args.group_field}' ({len(np.unique(groups))} groups)")
    else:
        tr_pool, te_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=y_np, random_state=42,
        )
    Xte = traj[torch.tensor(te_idx)]
    yte = labels[torch.tensor(te_idx)]
    print(f"train_pool={len(tr_pool)}  test={len(te_idx)}")

    probe_factories = build_probes(
        d, L, args.encoder_ckpt, args.single_layer, device, args.transformer_epochs,
        args.encoder_pool, args.encoder_layer_idx,
        args.encoder_window_start, args.encoder_window_end,
    )

    results: dict = {p: {n: [] for n in args.n_trains} for p in probe_factories}

    for n_train in args.n_trains:
        if n_train > len(tr_pool):
            print(f"  (skipping n_train={n_train}: only {len(tr_pool)} in pool)")
            continue
        print(f"\nN_train={n_train}")
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            sub = stratified_subsample(tr_pool, y_np, n_train, rng)
            Xtr = traj[torch.tensor(sub)]
            ytr = labels[torch.tensor(sub)]
            for pname, pfactory in probe_factories.items():
                probe = pfactory()
                probe.fit(Xtr, ytr)
                s = probe.predict_score(Xte)
                auc = float(roc_auc_score(yte.numpy(), s))
                results[pname][n_train].append(auc)
            print(f"  seed={seed}:  "
                  + "  ".join(f"{p}={results[p][n_train][-1]:.3f}" for p in probe_factories))

    print("\n" + "=" * 78)
    header = f"{'probe':<28s}" + "".join(f"{n:>8d}" for n in args.n_trains)
    print(header)
    print("-" * 78)
    for pname in probe_factories:
        row = f"{pname:<28s}"
        for n in args.n_trains:
            xs = results[pname][n]
            row += f" {np.mean(xs):.2f}\u00b1{np.std(xs):.2f}" if xs else f"{'--':>8s}"
        print(row)
    print("=" * 78)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data,
            "encoder_ckpt": args.encoder_ckpt,
            "encoder_pool": args.encoder_pool,
            "encoder_layer_idx": args.encoder_layer_idx,
            "n_trains": args.n_trains, "n_seeds": args.n_seeds,
            "single_layer": args.single_layer,
            "N_test": len(te_idx),
            "results": results,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
