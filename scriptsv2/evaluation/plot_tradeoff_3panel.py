"""
5.4.2 (sec:exp-tradeoff) — Reconstruction-sparsity tradeoff, 3-panel version.

Same as plot_tradeoff() in sparsity_analysis.py, but evaluating reconstruction
with all three metrics (MSE, MAE, SSIM) instead of just MSE and SSIM, using the
chapter's visual encoding: color distinguishes the code dimension and marker
shape distinguishes the training loss.

Requires neither checkpoints nor a GPU: every value (frac_zeros and the three
metrics for best/last, plus the background trajectory) comes from the
trajectory.json files already computed by trajectory_eval.py. Verified to
match the sparsity_per_sample.json cache up to the fourth decimal place.

Usage:
    python evaluation/plot_tradeoff_3panel.py
    python evaluation/plot_tradeoff_3panel.py --out path/figure.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR  = Path(__file__).resolve().parent.parent / "experiments"
LOSSES   = ["mse", "mae", "ssim"]
DIMS     = [128, 1024]
CKPTS    = ["best", "last"]

# Color = code dimension. The 1024 model trained with SSIM is split out because
# it's the only one that doesn't collapse; grouping it with the other two of
# its dimension would be confusing.
C_128    = "#808080"
C_1024   = "#F05AAA"
C_1024_S = "#3C6EB4"

MARKERS  = {"mse": "o", "mae": "^", "ssim": "D"}

# Panels: (key in trajectory.json, y-axis label, scale)
PANELS = [
    ("mse_mean",  "MSE gaussiana  (↓ mejor)",  "log"),
    ("mae_mean",  "MAE gaussiana  (↓ mejor)",  "log"),
    ("ssim_mean", "SSIM gaussiano  (↑ mejor)", "linear"),
]

# Label offset, in points, per (panel, loss, dim). Hand-tuned: the six points
# cluster differently in each panel.
LABEL_OFFSETS = {
    0: {("mse", 128): (12, -16), ("mae", 128): (14, 6), ("ssim", 128): (-6, 16),
        ("mse", 1024): (12, 10), ("mae", 1024): (10, -16), ("ssim", 1024): (12, -14)},
    1: {("mse", 128): (12, -16), ("mae", 128): (14, 6), ("ssim", 128): (-6, 16),
        ("mse", 1024): (12, 10), ("mae", 1024): (10, -16), ("ssim", 1024): (12, -14)},
    2: {("mse", 128): (12, -16), ("mae", 128): (14, 6), ("ssim", 128): (-6, 14),
        ("mse", 1024): (12, -16), ("mae", 1024): (10, 8), ("ssim", 1024): (12, -6)},
}


def color_of(rec_mode, code_dim):
    if code_dim == 128:
        return C_128
    return C_1024_S if rec_mode == "ssim" else C_1024


def load_trajectory(rec_mode, code_dim):
    p = (EXP_DIR / "2_metrics" / rec_mode / f"dim_{code_dim}"
         / "reconstruction" / "checkpoint_evolution" / "trajectory.json")
    if not p.exists():
        print(f"  [Aviso] falta {p}")
        return []
    with open(p) as f:
        return json.load(f)


def plot(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Compromiso reconstrucción – dispersión (test)", fontsize=16)

    for pi, (ax, (ykey, ylabel, yscale)) in enumerate(zip(axes, PANELS)):
        for rec_mode in LOSSES:
            for code_dim in DIMS:
                traj  = load_trajectory(rec_mode, code_dim)
                if not traj:
                    continue
                color = color_of(rec_mode, code_dim)

                # Background training trajectory.
                xs = np.asarray([t["frac_zeros"] for t in traj])
                ys = np.asarray([t[ykey] for t in traj])
                order = np.argsort(xs)
                ax.plot(xs[order], ys[order], color=color, alpha=0.18, lw=1.0, zorder=1)

                by_label = {t.get("label"): t for t in traj}
                pts = {}
                for ck in CKPTS:
                    t = by_label.get(ck)
                    if t is None:
                        continue
                    pts[ck] = (t["frac_zeros"], t[ykey])
                    ax.scatter(*pts[ck], s=180 if ck == "best" else 110,
                               marker=MARKERS[rec_mode], color=color,
                               edgecolor="black", linewidth=1.2,
                               alpha=1.0 if ck == "best" else 0.5, zorder=3)

                # Arrow best → last: shows which way training moves the tradeoff.
                if "best" in pts and "last" in pts:
                    ax.annotate("", xy=pts["last"], xytext=pts["best"], zorder=2,
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                                                alpha=0.7, shrinkA=7, shrinkB=7))
                if "best" in pts:
                    ax.annotate(f"{rec_mode}·{code_dim}", pts["best"],
                                textcoords="offset points",
                                xytext=LABEL_OFFSETS[pi][(rec_mode, code_dim)],
                                fontsize=9, fontweight="bold", color=color)

        ax.set_xlabel("grado de dispersión —  frac_zeros  (↑ más disperso)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        # Collapse spikes MSE and MAE (and the trajectory even more): on a linear
        # scale they flatten the whole useful range against the axis.
        ax.set_yscale(yscale)
        if ykey == "mae_mean":
            # The whole useful MAE range falls within one decade: without labeling
            # the minor ticks, the axis ends up showing only a single visible value.
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="y", which="minor", labelsize=8)

    handles = [
        plt.Line2D([], [], color=C_128,    lw=6, label="dim 128"),
        plt.Line2D([], [], color=C_1024,   lw=6, label="dim 1024 (mse, mae)"),
        plt.Line2D([], [], color=C_1024_S, lw=6, label="dim 1024 (ssim)"),
    ]
    handles += [plt.Line2D([], [], color="gray", marker=mk, ls="", markersize=10,
                           markerfacecolor="none", label=m)
                for m, mk in MARKERS.items()]
    handles += [plt.Line2D([], [], color="gray", lw=1.2, alpha=0.35,
                           label="trayectoria de entrenamiento")]
    axes[0].legend(handles=handles, fontsize=9, loc="upper center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Figura 5.4.2 con paneles MSE, MAE y SSIM.")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent
                   / "imaxes" / "tradeoff_rec_sparsity_mse_mae_ssim.png")
    plot(p.parse_args().out)


if __name__ == "__main__":
    main()
