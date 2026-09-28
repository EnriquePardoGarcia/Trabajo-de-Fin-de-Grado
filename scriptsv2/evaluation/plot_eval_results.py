"""
Generates comparison plots of evaluation results for dim=128, reading the
JSON files from experiments/2_metrics/.

Usage:
    python evaluation/plot_eval_results.py --code_dim 128
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import configuration as cfg

MODES   = ["mse", "mae", "ssim"]
CKPTS   = ["best", "last"]
COLORS  = {"mse": "#2196F3", "mae": "#4CAF50", "ssim": "#FF9800"}
MARKERS = {"best": "o", "last": "s"}
ALPHA   = {"best": 1.0, "last": 0.5}


def load(rec_mode: str, code_dim: int, ckpt: str) -> dict:
    path = (cfg.exp_metrics_dir(rec_mode, code_dim, "reconstruction", "per_class")
            / f"eval_test_{ckpt}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get(d, *keys, default=None):
    for k in keys:
        if d is None or not isinstance(d, dict):
            return default
        d = d.get(k, None)
    return d if d is not None else default


def plot_all(code_dim: int, out_dir: Path):
    data = {}
    for mode in MODES:
        data[mode] = {}
        for ck in CKPTS:
            data[mode][ck] = load(mode, code_dim, ck)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle(f"Evaluación test — dim={code_dim}  (best vs last)", fontsize=14)

    x     = np.arange(len(MODES))
    width = 0.35

    def bar_group(ax, values_best, values_last, title, ylabel, higher_is_better=True):
        bars_b = ax.bar(x - width/2, values_best, width, label="best",
                        color=[COLORS[m] for m in MODES], alpha=0.9)
        bars_l = ax.bar(x + width/2, values_last, width, label="last",
                        color=[COLORS[m] for m in MODES], alpha=0.45)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(MODES)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        arrow = "↑" if higher_is_better else "↓"
        ax.set_xlabel(f"rec_mode  ({arrow} mejor)", fontsize=7)

    # ── Row 0: reconstruction metrics ──
    bar_group(axes[0,0],
        [get(data[m]["best"], "psnr_db", "mean", default=0) for m in MODES],
        [get(data[m]["last"], "psnr_db", "mean", default=0) for m in MODES],
        "PSNR (dB)", "dB", higher_is_better=True)

    bar_group(axes[0,1],
        [get(data[m]["best"], "ssim_gaussiano", "mean", default=0) for m in MODES],
        [get(data[m]["last"], "ssim_gaussiano", "mean", default=0) for m in MODES],
        "SSIM gaussiano", "SSIM", higher_is_better=True)

    bar_group(axes[0,2],
        [get(data[m]["best"], "mae_gaussiana", "mean", default=0) for m in MODES],
        [get(data[m]["last"], "mae_gaussiana", "mean", default=0) for m in MODES],
        "MAE gaussiana", "MAE", higher_is_better=False)

    # ── Row 1: sparsity and atom health ──
    bar_group(axes[1,0],
        [get(data[m]["best"], "sparsity", "l0_activos", "mean", default=0) for m in MODES],
        [get(data[m]["last"], "sparsity", "l0_activos", "mean", default=0) for m in MODES],
        "L0 activos / muestra", "átomos", higher_is_better=True)

    bar_group(axes[1,1],
        [get(data[m]["best"], "atom_health", "n_alive", default=0) for m in MODES],
        [get(data[m]["last"], "atom_health", "n_alive", default=0) for m in MODES],
        f"Átomos vivos / {code_dim}", "átomos", higher_is_better=True)

    bar_group(axes[1,2],
        [get(data[m]["best"], "atom_health", "n_dead", default=0) for m in MODES],
        [get(data[m]["last"], "atom_health", "n_dead", default=0) for m in MODES],
        f"Átomos muertos / {code_dim}", "átomos", higher_is_better=False)

    # ── Row 2: per-class and coherence ──
    bar_group(axes[2,0],
        [get(data[m]["best"], "clase_mitotic_figure", "l0_mean", default=0) for m in MODES],
        [get(data[m]["last"], "clase_mitotic_figure", "l0_mean", default=0) for m in MODES],
        "L0 — mitóticas", "átomos", higher_is_better=True)

    bar_group(axes[2,1],
        [get(data[m]["best"], "clase_not_mitotic_figure", "l0_mean", default=0) for m in MODES],
        [get(data[m]["last"], "clase_not_mitotic_figure", "l0_mean", default=0) for m in MODES],
        "L0 — no-mitóticas", "átomos", higher_is_better=True)

    bar_group(axes[2,2],
        [get(data[m]["best"], "dictionary_coherence", "coherence_mean", default=0) for m in MODES],
        [get(data[m]["last"], "dictionary_coherence", "coherence_mean", default=0) for m in MODES],
        "Coherencia diccionario (media)", "coherencia", higher_is_better=False)

    plt.tight_layout()
    out = out_dir / f"eval_comparison_dim{code_dim}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")

    # ── Per-atom activation rate plot ──
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    fig2.suptitle(f"Tasa de activación por átomo — dim={code_dim}", fontsize=13)
    for col, mode in enumerate(MODES):
        for row, ck in enumerate(CKPTS):
            ax = axes2[row, col]
            d  = data[mode][ck]
            if d is None:
                ax.set_visible(False); continue
            rate = get(d, "atom_health", "activation_rate", default=[])
            if rate:
                rate = np.array(rate)
                ax.bar(np.arange(len(rate)), np.sort(rate)[::-1],
                       color=COLORS[mode], alpha=0.8, width=1.0)
                thr = get(d, "atom_health", "dead_threshold", default=0.1)
                ax.axhline(thr, color="red", linestyle="--", linewidth=1, label=f"thr={thr}")
                ax.set_title(f"{mode} — {ck} (ep.{d['epoch']})", fontsize=9)
                ax.set_xlabel("átomo (ordenado)", fontsize=7)
                ax.set_ylabel("tasa activación", fontsize=7)
                ax.legend(fontsize=7)
                ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = out_dir / f"activation_rate_dim{code_dim}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Guardado: {out2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_dim", type=int, default=128)
    args = p.parse_args()

    out_dir = cfg.exp_comparativa_dir("7_eval_plots")
    print(f"Generando gráficas para dim={args.code_dim}...")
    plot_all(args.code_dim, out_dir)
    print("Listo.")


if __name__ == "__main__":
    main()
