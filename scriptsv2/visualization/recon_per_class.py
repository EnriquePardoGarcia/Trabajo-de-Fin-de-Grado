"""
Per-class reconstructions: best/last × mitotic/non-mitotic from the TEST set.

Each configuration is measured ONLY with its own training metric
(mse→MSE_gauss, mae→MAE_gauss, ssim→SSIM_gauss), not with all three.

The displayed patch is the SAME for best and last (deterministic choice via
seed), so the two figures for a given class are comparable with each other.

Output in experiments/3_visualization/<rec_mode>/dim_<code_dim>/reconstructions/per_class/:
    <rec_mode>_<code_dim>_recon_test_<rec_mode>_<best|last>_<mitotic|non_mitotic>.png

Usage:
    python visualization/recon_per_class.py --rec_mode mse --code_dim 128
    python visualization/recon_per_class.py --all --clean
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from core.dataset import CellsDataset

from vis_utils           import load_model, get_split_indices, get_rec_metric_fn
from vis_reconstructions import vis_recon_single

# Order from core.dataset.CLASS_NAMES: 0 = mitotic_figure, 1 = not_mitotic_figure
CLASS_TAGS = {0: "mitotic", 1: "non_mitotic"}
SEED = 42


def pick_test_indices(dataset, test_indices):
    """One test index per class, deterministic, identical for best and last."""
    rng = random.Random(SEED)
    chosen = {}
    for label, tag in CLASS_TAGS.items():
        pool = [i for i in test_indices if int(dataset.labels[i]) == label]
        if not pool:
            print(f"  [Aviso] no hay muestras de test de clase {tag}")
            continue
        chosen[label] = rng.choice(pool)
    return chosen


def clean_dir(out_dir: Path):
    removed = 0
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            p.unlink()
            removed += 1
    print(f"  Vaciado {out_dir} ({removed} ficheros borrados)")


def run(rec_mode: str, code_dim: int, cells_dir: Path, device, clean: bool):
    print(f"\n{'='*60}\n{rec_mode}/dim_{code_dim}\n{'='*60}")
    out_dir = cfg.exp_vis_dir(rec_mode, code_dim, "reconstructions", "per_class")
    if clean:
        clean_dir(out_dir)

    ckpts = {
        "best": cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt",
        "last": cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt",
    }
    missing = [k for k, v in ckpts.items() if not v.exists()]
    if missing:
        print(f"  [Aviso] faltan checkpoints: {missing}")

    dataset = CellsDataset(cells_dir, augment=False, flatten=True)
    ref_ckpt = next((v for v in ckpts.values() if v.exists()), None)
    if ref_ckpt is None:
        print("  [Error] sin checkpoints, se omite")
        return 0
    _, test_indices = get_split_indices(rec_mode, code_dim, ref_ckpt, len(dataset))
    chosen = pick_test_indices(dataset, test_indices)

    n = 0
    for ckpt_type, path in ckpts.items():
        if not path.exists():
            continue
        encoder, decoder, hparams, epoch = load_model(path, device)
        metric_fn, metric_lbl = get_rec_metric_fn(
            rec_mode, hparams.get("ssim_data_range", 1.0)
        )
        for label, idx in chosen.items():
            tag = CLASS_TAGS[label]
            name = f"{rec_mode}_{code_dim}_recon_test_{rec_mode}_{ckpt_type}_{tag}.png"
            vis_recon_single(encoder, decoder, hparams, epoch, dataset, idx,
                             "TEST", out_dir / name, device, metric_fn, metric_lbl)
            print(f"  Guardado: {name}  (idx {idx}, {tag})")
            n += 1
    return n


def parse_args():
    p = argparse.ArgumentParser(
        description="4 reconstrucciones por config: best/last × mitótica/no-mitótica (test)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rec_mode", type=str, choices=["mse", "mae", "ssim"])
    g.add_argument("--all", action="store_true",
                   help="Las 6 combinaciones (mse/mae/ssim × dim 128/1024).")
    p.add_argument("--code_dim",  type=int, default=cfg.CODE_DIM)
    p.add_argument("--cells_dir", type=str, default=None)
    p.add_argument("--clean",     action="store_true",
                   help="Vacía per_class/ antes de regenerar.")
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ no encontrado: {cells_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combos = ([(m, d) for m in ["mse", "mae", "ssim"] for d in [128, 1024]]
              if args.all else [(args.rec_mode, args.code_dim)])

    total = 0
    for rec_mode, code_dim in combos:
        total += run(rec_mode, code_dim, cells_dir, device, args.clean)
    print(f"\nTotal de figuras generadas: {total}")


if __name__ == "__main__":
    main()
