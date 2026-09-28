"""
Top-16 Fisher summary written to fisher/feature_selection/global/.

For each feature_ranking_*.json, extracts the top-16 atoms by Fisher index along
with their Fisher, ANOVA-F, and MI values, plus the normalized ranks used by the
"Rank comparison" panel: rank_norm(v) = argsort(argsort(v)) / (K-1), reproduced
as-is from feature_eval.plot_feature_selection_comparison.

Usage:
    python comparison/fisher_summaries.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import configuration as cfg

CONFIGS = [(m, d) for m in ["mse", "mae", "ssim"] for d in [128, 1024]]
CKPTS   = ["best", "last"]
TOP_K   = 16


def rank_norm(v):
    """Identical to feature_eval: ascending rank normalized to [0, 1]."""
    r = np.argsort(np.argsort(v)).astype(float)
    return r / max(len(r) - 1, 1)


def summarize(full: dict, rec_mode: str, code_dim: int, ckpt: str) -> dict:
    atoms = full["all_atoms"]
    fi    = np.array([a["fisher"] for a in atoms], dtype=float)
    fa    = np.array([a["f_anova"] for a in atoms], dtype=float)
    mi    = np.array([a["mutual_info"] for a in atoms], dtype=float)

    fi_n, fa_n, mi_n = rank_norm(fi), rank_norm(fa), rank_norm(mi)
    top = np.argsort(fi)[::-1][:TOP_K]

    return {
        "rec_mode":   rec_mode,
        "code_dim":   code_dim,
        "ckpt":       ckpt,
        "n_atoms":    full["n_atoms"],
        "n_samples":  full["n_samples"],
        "top_k":      TOP_K,
        "rank_norm_definition": "argsort(argsort(v)) / (n_atoms - 1); 1.0 = mejor de su métrica",
        "top_by_fisher": [
            {
                "position":    int(i + 1),
                "atom":        int(k),
                "fisher":      float(fi[k]),
                "f_anova":     float(fa[k]),
                "mutual_info": float(mi[k]),
                "rank_fisher": float(fi_n[k]),
                "rank_anova":  float(fa_n[k]),
                "rank_mi":     float(mi_n[k]),
            }
            for i, k in enumerate(top)
        ],
    }


def main():
    n = 0
    for rec_mode, code_dim in CONFIGS:
        fs_dir  = cfg.exp_analysis_dir(rec_mode, code_dim, "fisher", "feature_selection")
        out_dir = fs_dir / "global"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ck in CKPTS:
            name = f"{rec_mode}_{code_dim}_feature_ranking_{ck}.json"
            src  = fs_dir / name
            if not src.exists():
                print(f"  [Aviso] no existe {src}")
                continue
            with open(src) as f:
                full = json.load(f)
            with open(out_dir / name, "w") as f:
                json.dump(summarize(full, rec_mode, code_dim, ck), f, indent=2)
            print(f"  {rec_mode}/dim_{code_dim} [{ck}] -> {out_dir / name}")
            n += 1
    print(f"\n{n} resúmenes generados")


if __name__ == "__main__":
    main()
