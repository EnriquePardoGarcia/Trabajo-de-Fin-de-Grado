"""
Evolution of the reconstruction throughout training, in a SINGLE figure.

Shows the original TEST patch and its reconstruction at each selected
checkpoint (by default 6 evenly spaced jumps + best + last = 8 reconstructions),
ordered by epoch so the evolution reads left to right.

Each configuration is measured ONLY with its own training metric
(mse→MSE_gauss, mae→MAE_gauss, ssim→SSIM_gauss).

Output (a single figure) in
experiments/3_visualization/<rec_mode>/dim_<code_dim>/reconstructions/checkpoint_evolution/:
    <rec_mode>_<code_dim>_evolution_test_<rec_mode>_<class>.png

Usage:
    python visualization/recon_evolution.py --rec_mode mse --code_dim 128
    python visualization/recon_evolution.py --all --clean
"""

import argparse
import math
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from core.dataset import CellsDataset
from evaluation.trajectory_eval import discover_checkpoints

from vis_utils import to_img, load_model, get_split_indices, get_rec_metric_fn

# Order from core.dataset.CLASS_NAMES: 0 = mitotic_figure, 1 = not_mitotic_figure
CLASS_TAGS   = {0: "mitotic", 1: "non_mitotic"}
CLASS_TITLES = {0: "mitótica", 1: "no-mitótica"}
SEED = 42

# Reconstruction figures that this single image replaces. The plot_*/legend
# figures from trajectory_eval (trajectory metrics) are NOT touched.
OBSOLETE_PREFIXES = ("evo_", "mosaic_all_ckpts", "compare_train_", "compare_test_")


def pick_test_index(dataset, test_indices, label):
    """Same criterion as recon_per_class.py: deterministic and comparable across configs."""
    rng = random.Random(SEED)
    chosen = {}
    for lab in CLASS_TAGS:
        pool = [i for i in test_indices if int(dataset.labels[i]) == lab]
        chosen[lab] = rng.choice(pool) if pool else None
    return chosen[label]


def clean_obsolete(out_dir: Path):
    removed = 0
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name.startswith(OBSOLETE_PREFIXES):
            p.unlink()
            removed += 1
    print(f"  Borradas {removed} figuras de reconstrucción antiguas (plot_*/legend intactos)")


def run(rec_mode, code_dim, cells_dir, device, n_jumps, label, clean):
    print(f"\n{'='*60}\n{rec_mode}/dim_{code_dim}\n{'='*60}")
    out_dir = cfg.exp_vis_dir(rec_mode, code_dim, "reconstructions", "checkpoint_evolution")
    if clean:
        clean_obsolete(out_dir)

    checkpoints = discover_checkpoints(rec_mode, code_dim, n_jumps)
    if not checkpoints:
        print("  [Error] sin checkpoints, se omite")
        return 0

    dataset = CellsDataset(cells_dir, augment=False, flatten=True)
    _, test_indices = get_split_indices(rec_mode, code_dim, checkpoints[0][0], len(dataset))
    idx = pick_test_index(dataset, test_indices, label)
    if idx is None:
        print(f"  [Aviso] sin muestras de test de clase {CLASS_TAGS[label]}")
        return 0

    x_flat, _ = dataset[idx]
    x = x_flat.unsqueeze(0).to(device)

    panels = []
    for ckpt_path, ckpt_label in checkpoints:
        encoder, decoder, hparams, epoch = load_model(ckpt_path, device)
        # In jump checkpoints, the "epoch" field reflects the last time the file
        # was written, not the jump itself; the real epoch comes from the
        # checkpoint directory name (matches jump_meta["epoch_before"]).
        # best/last do carry the correct epoch.
        m = re.search(r"_(\d+)$", ckpt_label)
        if m:
            epoch = int(m.group(1))
        metric_fn, metric_lbl = get_rec_metric_fn(
            rec_mode, hparams.get("ssim_data_range", 1.0)
        )
        with torch.no_grad():
            x_hat = decoder(encoder(x))
        val = metric_fn(x[0].detach().cpu(), x_hat[0].detach().cpu())
        panels.append({
            "epoch": epoch if isinstance(epoch, int) else -1,
            "label": ckpt_label,
            "img":   to_img(x_hat[0].detach().cpu(),
                            hparams.get("dataset_mean", 0.0),
                            hparams.get("dataset_std", 1.0)),
            "val":   val,
            "mean":  hparams.get("dataset_mean", 0.0),
            "std":   hparams.get("dataset_std", 1.0),
        })

    # Readable evolution: chronological order, with best/last marked at their real epoch.
    panels.sort(key=lambda p: p["epoch"])

    ref = panels[0]
    n_total = len(panels) + 1
    ncols   = min(3 if n_total <= 9 else 5, n_total)
    nrows   = math.ceil(n_total / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows))
    axes = axes.ravel() if n_total > 1 else [axes]

    fig.suptitle(
        f"Evolución de la reconstrucción — {rec_mode}/dim_{code_dim}  "
        f"[{metric_lbl}]  TEST idx {idx} ({CLASS_TITLES[label]})",
        fontsize=15,
    )

    axes[0].imshow(to_img(x[0].detach().cpu(), ref["mean"], ref["std"]))
    axes[0].set_title("ORIGINAL", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for ax, p in zip(axes[1:], panels):
        ax.imshow(p["img"])
        mark = "★ " if p["label"] in ("best", "last") else ""
        ax.set_title(f"{mark}{p['label']}\nepoch {p['epoch']}  —  {metric_lbl}={p['val']:.4f}",
                     fontsize=10)
        ax.axis("off")
    for ax in axes[n_total:]:
        ax.axis("off")

    plt.tight_layout()
    name = f"{rec_mode}_{code_dim}_evolution_test_{rec_mode}_{CLASS_TAGS[label]}.png"
    plt.savefig(out_dir / name, dpi=150)
    plt.close()
    print(f"  Guardado: {name}  ({len(panels)} reconstrucciones + original)")
    return 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Una figura por config con la evolución de la reconstrucción."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rec_mode", type=str, choices=["mse", "mae", "ssim"])
    g.add_argument("--all", action="store_true")
    p.add_argument("--code_dim",  type=int, default=cfg.CODE_DIM)
    p.add_argument("--cells_dir", type=str, default=None)
    p.add_argument("--n_jumps",   type=int, default=6,
                   help="Jumps intermedios a mostrar (best y last se añaden aparte).")
    p.add_argument("--class_id",  type=int, default=0, choices=[0, 1],
                   help="0 = mitótica (defecto), 1 = no-mitótica.")
    p.add_argument("--clean", action="store_true",
                   help="Borra evo_*/mosaic/compare_* previos (conserva plot_* y legend).")
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
        total += run(rec_mode, code_dim, cells_dir, device,
                     args.n_jumps, args.class_id, args.clean)
    print(f"\nFiguras generadas: {total}")


if __name__ == "__main__":
    main()
