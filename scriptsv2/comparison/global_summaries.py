"""
Reduced summaries of the six configurations, written to 5_dictionaries/global/.

For each eval_test_*.json, keeps the identification (rec_mode, code_dim), the
sparsity and reconstruction metrics, and the atom health blocks (atom_health,
including dead_atom_indices, and atom_trajectory), under their original names.
Per-class activations and per-atom analysis are discarded.

Usage:
    python comparison/global_summaries.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg

CONFIGS = [(m, d) for m in ["mse", "mae", "ssim"] for d in [128, 1024]]
CKPTS   = ["best", "last"]

L0_KEYS = ["mean", "std", "p25", "p50", "p75"]


def reduce_summary(full: dict, rec_mode: str, code_dim: int) -> dict:
    sp  = full.get("sparsity", {})
    l0  = sp.get("l0_activos", {})
    fz  = sp.get("frac_zeros", {})
    out = {
        "rec_mode": rec_mode,
        "code_dim": code_dim,
        "sparsity": {
            "l0_activos": {k: l0[k] for k in L0_KEYS if k in l0},
            "frac_zeros": {"mean": fz.get("mean")},
        },
    }
    for key in ["mse_gaussiana", "mae_gaussiana", "ssim_gaussiano"]:
        if key in full:
            out[key] = {"mean": full[key].get("mean")}
    # atom_health already contains dead_atom_indices; atom_trajectory holds per-sample series.
    for key in ["atom_health", "atom_trajectory"]:
        if key in full:
            out[key] = full[key]
    return out


def main():
    out_dir = cfg.EXPERIMENTS_DIR / "5_dictionaries" / "global"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for rec_mode, code_dim in CONFIGS:
        for ck in CKPTS:
            name = f"{rec_mode}_{code_dim}_eval_test_{ck}.json"
            src  = cfg.exp_dicts_dir(rec_mode, code_dim) / name
            if not src.exists():
                print(f"  [Aviso] no existe {src}")
                continue
            with open(src) as f:
                full = json.load(f)
            with open(out_dir / name, "w") as f:
                json.dump(reduce_summary(full, rec_mode, code_dim), f, indent=2)
            print(f"  {name}")
            n += 1
    print(f"\n{n} resúmenes en {out_dir}")


if __name__ == "__main__":
    main()
