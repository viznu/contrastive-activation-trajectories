"""Train a TrajectoryEncoder with a SupCon-style InfoNCE contrastive loss
over paired activation trajectories.

Loss:
  For each anchor, positives are all same-label samples across the training
  set (other pairs of the same label), negatives are different-label samples
  (the within-pair partner plus other-label samples across pairs).

  L = - mean_i [ (1/|P(i)|) * sum_{p in P(i)} log exp(sim(z_i, z_p)/T) /
                                               sum_{a != i} exp(sim(z_i, z_a)/T) ]

  pooled latents are L2-normalized before computing sim. Temperature T = 0.07
  by default (SimCLR convention).

One gradient step per epoch over the full training set gives the
InfoNCE loss a large number of negatives per anchor, which matters more
for contrastive objectives than batch-wise noise. Dataset sizes in the
paired-prompt regime (hundreds to low thousands of trajectories) fit
comfortably in one similarity matrix.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.encoder.model import TrajectoryEncoder, pool_latents


def compute_infonce(
    encoder: TrajectoryEncoder,
    x: torch.Tensor,
    labels: torch.Tensor,
    pool: str,
    layer_idx: int | None,
    window_start: int | None,
    window_end: int | None,
    temperature: float,
) -> torch.Tensor:
    z = encoder(x)  # (B, L, d_model)
    pooled = pool_latents(z, mode=pool, layer_idx=layer_idx,
                          window_start=window_start, window_end=window_end)
    pooled = F.normalize(pooled, dim=-1)

    B = pooled.size(0)
    sim = pooled @ pooled.T / temperature  # (B, B)

    self_mask = torch.eye(B, dtype=torch.bool, device=sim.device)
    # -1e9 (not -inf) to avoid -inf * 0 = NaN in the masked multiply below.
    sim = sim.masked_fill(self_mask, -1e9)

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_label & ~self_mask

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / n_pos

    valid = pos_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=sim.device)
    return -mean_log_prob_pos[valid].mean()


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seed={args.seed}  pool={args.pool}  layer_idx={args.layer_idx}  T={args.temperature}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.mps.manual_seed(args.seed)
    except (AttributeError, RuntimeError):
        pass

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    N, L, d = traj.shape
    print(f"loaded {args.data}: N={N} L={L} d={d}")

    # Z-normalize per layer across the dataset (layers 0..L differ by ~30x
    # in norm for a typical residual-stream trajectory; normalization
    # stabilizes the InfoNCE geometry).
    mean = traj.mean(dim=0, keepdim=True)
    std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
    traj_norm = (traj - mean) / std

    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = traj_norm[tr_idx].to(device)
    tr_labels = labels[tr_idx].to(device)
    vl = traj_norm[val_idx].to(device)
    vl_labels = labels[val_idx].to(device)
    print(f"train={len(tr)} val={len(vl)}")

    encoder = TrajectoryEncoder(
        d_in=d,
        d_model=args.d_model,
        num_layers=args.enc_layers,
        num_heads=args.heads,
        max_L=max(L, 64),
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"params={n_params:,}  d_model={args.d_model}  enc_layers={args.enc_layers}")

    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = []
    for epoch in range(args.epochs):
        t0 = time.time()
        encoder.train()
        loss = compute_infonce(
            encoder, tr, tr_labels,
            pool=args.pool, layer_idx=args.layer_idx,
            window_start=args.window_start, window_end=args.window_end,
            temperature=args.temperature,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        sched.step()

        encoder.eval()
        with torch.no_grad():
            val_loss = compute_infonce(
                encoder, vl, vl_labels,
                pool=args.pool, layer_idx=args.layer_idx,
                window_start=args.window_start, window_end=args.window_end,
                temperature=args.temperature,
            ).item()

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": float(loss.item()),
               "val_loss": val_loss, "lr": lr_now, "time": dt}
        log.append(rec)
        print(f"epoch {epoch:3d}  train={loss.item():.4f}  val={val_loss:.4f}  lr={lr_now:.2e}  ({dt:.1f}s)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": encoder.state_dict(),
        "config": vars(args),
        "log": log,
        "norm_mean": mean,
        "norm_std": std,
        "d_in": d,
        "L": L,
    }, out)
    with open(out.with_suffix(".log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=0,
                   help="Unused — InfoNCE runs on the full train set per epoch.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--pool", default="layer",
                   choices=["mean", "last", "mid", "layer", "window"])
    p.add_argument("--layer_idx", type=int, default=24)
    p.add_argument("--window_start", type=int, default=None)
    p.add_argument("--window_end", type=int, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
