"""Generate figures from result JSONs in results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"


def _probe_order():
    return [
        ("single_layer", "Single-layer\n(raw layer 24)"),
        ("all_layers_concat", "All-layers-concat\n(flattened L·d)"),
        ("transformer_over_layers", "Supervised\ntransformer-over-layers"),
        ("contrastive_encoder", "Contrastive-pretrained\nencoder"),
    ]


def fig_main_comparison(src_json: Path, out_path: Path):
    """Bar chart: AUROC and recall@1%FPR for all four probes on IP."""
    with open(src_json) as f:
        blob = json.load(f)
    results = blob["results"]

    probes = _probe_order()
    aurocs = [results[k]["test_auroc"] for k, _ in probes]
    recalls = [results[k]["test_recall_at_1pct_fpr"] for k, _ in probes]
    labels = [lbl for _, lbl in probes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(probes))

    colors = ["#8aa0c7", "#8aa0c7", "#8aa0c7", "#d97b4f"]
    axes[0].bar(x, aurocs, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.9, 1.002)
    axes[0].set_title("AUROC on Instructed-Pairs test (fact-grouped split)")
    for xi, v in zip(x, aurocs):
        axes[0].text(xi, v + 0.0015, f"{v:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, recalls, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("recall @ 1% FPR")
    axes[1].set_ylim(0.8, 1.01)
    axes[1].set_title("Recall at 1% false-positive rate")
    for xi, v in zip(x, recalls):
        axes[1].text(xi, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle("Main comparison — contrastive-pretrained encoder vs raw-activation baselines",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def fig_few_shot(src_json: Path, out_path: Path):
    """AUROC vs N_train per probe, with std bands from multiple seeds."""
    with open(src_json) as f:
        blob = json.load(f)
    n_trains = blob["n_trains"]
    results = blob["results"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    probes = _probe_order()
    colors = ["#4c72b0", "#55a868", "#c44e52", "#d97b4f"]
    markers = ["o", "s", "^", "D"]

    for (key, label), color, marker in zip(probes, colors, markers):
        per_n = results[key]
        means = [np.mean(per_n[str(n) if str(n) in per_n else n]) for n in n_trains]
        stds = [np.std(per_n[str(n) if str(n) in per_n else n]) for n in n_trains]
        ax.errorbar(n_trains, means, yerr=stds, label=label.replace("\n", " "),
                    color=color, marker=marker, capsize=3, linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_xticks(n_trains)
    ax.set_xticklabels([str(n) for n in n_trains])
    ax.set_xlabel("Labeled training examples (N_train)")
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title("Label efficiency on Instructed-Pairs (5 seeds per point)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def fig_transfer(src_jsons: dict, out_path: Path):
    """Bar chart: in-dist vs transfer AUROC across three target settings."""
    settings = []
    in_dist = []
    transfer = []
    for label, path in src_jsons.items():
        with open(path) as f:
            b = json.load(f)
        settings.append(label)
        in_dist.append(b["indist"]["auroc"])
        transfer.append(b["transfer"]["auroc"])

    x = np.arange(len(settings))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, in_dist, width, label="in-dist (IP-true test)", color="#8aa0c7")
    ax.bar(x + width / 2, transfer, width, label="zero-shot transfer", color="#d97b4f")

    for xi, v in zip(x - width / 2, in_dist):
        ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, transfer):
        ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=9)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("AUROC")
    ax.set_title("Zero-shot transfer of contrastive-pretrained encoder\n"
                 "(layer-24 pool; probe fit on IP-true-train, no refit)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def fig_architecture_ablation(src_json: Path, out_path: Path):
    """4-way architecture comparison: same SupCon objective, different encoders."""
    with open(src_json) as f:
        blob = json.load(f)
    results = blob["results"]

    kinds_order = ["linear_single", "linear_concat", "mlp_concat", "transformer"]
    labels = {
        "linear_single": "Linear\n(layer 24 only)\n0.5M params",
        "linear_concat": "Linear\n(flat trajectory)\n19M params",
        "mlp_concat": "MLP\n(flat trajectory)\n19M params",
        "transformer": "Transformer\n(over layers)\n3.7M params",
    }
    aurocs = [results[k]["full_train_auroc"] for k in kinds_order]
    recalls = [results[k]["full_train_recall_at_1pct_fpr"] for k in kinds_order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(kinds_order))
    colors = ["#8aa0c7", "#8aa0c7", "#8aa0c7", "#8aa0c7"]

    axes[0].bar(x, aurocs, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[k] for k in kinds_order], fontsize=8)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.97, 1.002)
    axes[0].set_title("AUROC on IP test (full N_train)")
    for xi, v in zip(x, aurocs):
        axes[0].text(xi, v + 0.0008, f"{v:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, recalls, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[k] for k in kinds_order], fontsize=8)
    axes[1].set_ylabel("recall @ 1% FPR")
    axes[1].set_ylim(0.85, 1.01)
    axes[1].set_title("Recall @ 1% FPR")
    for xi, v in zip(x, recalls):
        axes[1].text(xi, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle("Architecture ablation — same SupCon objective, different encoders\n"
                 "(All four converge within noise; architecture is cosmetic on this task)",
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig_main_comparison(RESULTS / "eval_ip_layer24.json",
                        FIGURES / "main_comparison.png")
    fig_few_shot(RESULTS / "few_shot_ip.json",
                 FIGURES / "few_shot.png")
    fig_transfer({
        "Content\n(IP-true → IP-false)": RESULTS / "transfer_content_layer24.json",
        "Task (distinct prefills)\n(IP → refuse/comply)": RESULTS / "transfer_task_refusal_distinct_layer24.json",
        "Task (matched prefills)\n(IP → refuse/comply)": RESULTS / "transfer_task_refusal_matched_layer24.json",
    }, FIGURES / "transfer.png")
    fig_architecture_ablation(RESULTS / "fair_contrastive_baselines.json",
                              FIGURES / "architecture_ablation.png")


if __name__ == "__main__":
    main()
