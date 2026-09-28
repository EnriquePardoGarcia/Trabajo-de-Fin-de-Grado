"""Comparison of evaluation metrics across rec_modes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from vis_utils import load_eval_json


def vis_eval_comparison(code_dim: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    MODES  = ["mse", "mae", "ssim"]
    COLORS = {"mse": "#2196F3", "mae": "#4CAF50", "ssim": "#FF9800"}
    x, width = np.arange(len(MODES)), 0.35

    data = {m: {c: load_eval_json(m, code_dim, c) for c in ["best", "last"]} for m in MODES}

    def gv(d, *keys):
        for k in keys:
            if not isinstance(d, dict): return 0
            d = d.get(k, {})
        return float(d) if isinstance(d, (int, float)) else 0

    def bar_group(ax, vb, vl, title, ylabel, higher_is_better=True):
        ax.bar(x - width/2, vb, width, label="best", color=[COLORS[m] for m in MODES], alpha=0.9)
        ax.bar(x + width/2, vl, width, label="last",  color=[COLORS[m] for m in MODES], alpha=0.45)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(MODES)
        ax.legend(fontsize=7); ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel(f"({'↑' if higher_is_better else '↓'} mejor)", fontsize=7)

    fig, axes = plt.subplots(4, 3, figsize=(16, 17))
    fig.suptitle(f"Evaluación test — dim={code_dim}  (best vs last)", fontsize=14)

    bar_group(axes[0,0], [gv(data[m]["best"], "psnr_db", "mean") for m in MODES],
                         [gv(data[m]["last"], "psnr_db", "mean") for m in MODES], "PSNR (dB)", "dB", True)
    bar_group(axes[0,1], [gv(data[m]["best"], "ssim_gaussiano", "mean") for m in MODES],
                         [gv(data[m]["last"], "ssim_gaussiano", "mean") for m in MODES], "SSIM gaussiano", "SSIM", True)
    bar_group(axes[0,2], [gv(data[m]["best"], "mae_gaussiana", "mean") for m in MODES],
                         [gv(data[m]["last"], "mae_gaussiana", "mean") for m in MODES], "MAE gaussiana", "MAE", False)
    bar_group(axes[1,0], [gv(data[m]["best"], "sparsity", "l0_activos", "mean") for m in MODES],
                         [gv(data[m]["last"], "sparsity", "l0_activos", "mean") for m in MODES], "L0 activos / muestra", "átomos", True)
    bar_group(axes[1,1], [gv(data[m]["best"], "atom_health", "n_alive") for m in MODES],
                         [gv(data[m]["last"], "atom_health", "n_alive") for m in MODES], f"Átomos vivos / {code_dim}", "átomos", True)
    bar_group(axes[1,2], [gv(data[m]["best"], "atom_health", "n_dead") for m in MODES],
                         [gv(data[m]["last"], "atom_health", "n_dead") for m in MODES], f"Átomos muertos / {code_dim}", "átomos", False)
    bar_group(axes[2,0], [gv(data[m]["best"], "atom_health", "n_dying") for m in MODES],
                         [gv(data[m]["last"], "atom_health", "n_dying") for m in MODES], f"Átomos agonizantes / {code_dim}", "átomos", False)
    bar_group(axes[2,1], [gv(data[m]["best"], "clase_mitotic_figure", "l0_mean") for m in MODES],
                         [gv(data[m]["last"], "clase_mitotic_figure", "l0_mean") for m in MODES], "L0 — mitóticas", "átomos", True)
    bar_group(axes[2,2], [gv(data[m]["best"], "clase_not_mitotic_figure", "l0_mean") for m in MODES],
                         [gv(data[m]["last"], "clase_not_mitotic_figure", "l0_mean") for m in MODES], "L0 — no-mitóticas", "átomos", True)
    bar_group(axes[3,0], [gv(data[m]["best"], "dictionary_coherence", "coherence_mean") for m in MODES],
                         [gv(data[m]["last"], "dictionary_coherence", "coherence_mean") for m in MODES], "Coherencia (media)", "coherencia", False)
    bar_group(axes[3,1], [gv(data[m]["best"], "dictionary_coherence", "coherence_max") for m in MODES],
                         [gv(data[m]["last"], "dictionary_coherence", "coherence_max") for m in MODES], "Coherencia (max)", "coherencia", False)
    axes[3,2].axis("off")

    plt.tight_layout()
    out = out_dir / f"eval_comparison_dim{code_dim}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out}")
