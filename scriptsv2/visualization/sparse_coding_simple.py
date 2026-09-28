"""
Reduced version of the LISTA sparse coding figure.

Shows only:  [x original] | [x̂ reconstruction] | [dictionary atoms]
No coefficient plot, no formula, and no overall title.

Usage:
    python visualization/sparse_coding_simple.py
    python visualization/sparse_coding_simple.py --index 4207 --n_atoms 6
    python visualization/sparse_coding_simple.py --ckpt /path/best.pt --out fig.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from core.dataset import CellsDataset
from vis_utils import load_model, to_img, atom_img
from sparse_coding_figure import RANK_COLORS, pick_sample

# Best reconstruction measured on the dataset (MSE 0.2076, SSIM 0.6465).
DEFAULT_CKPT = Path("/mnt/disk1/enrique.pardo.garcia/1_checkpoints/mse/dim_128/best/best.pt")


@torch.no_grad()
def build_figure(encoder, decoder, hparams, dataset, idx, out_path, device, n_atoms):
    mean = hparams.get("dataset_mean", 0.0)
    std  = hparams.get("dataset_std",  1.0)

    x, _ = dataset[idx]
    x  = x.to(device)
    z  = encoder(x.unsqueeze(0)).squeeze(0)
    xh = decoder(z.unsqueeze(0)).squeeze(0)

    z_np = z.detach().cpu().numpy()
    mse  = float((x - xh).pow(2).mean())

    active = np.where(z_np > 0)[0]
    active = active[np.argsort(z_np[active])[::-1]]
    n_show = min(n_atoms, len(active))
    top    = active[:n_show]

    W_e = encoder.W_e.weight.detach().cpu()               # (D, IN_DIM); W_d = W_e^T

    # ── Canvas: top row (x, x̂) above bottom row (atoms) ────────────────────
    # The two large images each span 2 columns, so the top block and the
    # strip of n_show atoms line up exactly in width.
    unit = 3.0
    fig  = plt.figure(figsize=(unit * n_show, unit * 3.0 + 0.9))
    gs   = fig.add_gridspec(2, n_show, height_ratios=[2.0, 1.0],
                            wspace=0.10, hspace=0.20,
                            left=0.015, right=0.985, top=0.925, bottom=0.02)

    half = n_show // 2
    ax = fig.add_subplot(gs[0, 0:half])
    ax.imshow(to_img(x.cpu(), mean, std))
    ax.set_title(r"$\mathbf{x}$ — original", fontsize=19, pad=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, half:n_show])
    ax.imshow(to_img(xh.cpu(), mean, std))
    ax.set_title(r"$\hat{\mathbf{x}} = \mathbf{W}_d\mathbf{z}$ — reconstrucción",
                 fontsize=19, pad=12)
    ax.axis("off")

    for r, k in enumerate(top):
        ax  = fig.add_subplot(gs[1, r])
        col = RANK_COLORS[r % len(RANK_COLORS)]
        ax.imshow(atom_img(W_e[k]))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.0)
        ax.set_title(rf"$\mathbf{{w}}_{{{k}}}$", fontsize=15, color=col, pad=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {out_path}")
    print(f"  Célula   : idx={idx}  {dataset.paths[idx].name}")
    print(f"  Activos  : {len(active)}/{len(z_np)}   MSE={mse:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cell",    type=str, default=None)
    ap.add_argument("--index",   type=int, default=None)
    ap.add_argument("--n_atoms", type=int, default=4)
    ap.add_argument("--pool",    type=int, default=1500)
    ap.add_argument("--out",     type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder, hparams, epoch = load_model(args.ckpt, device)

    stats   = (hparams["dataset_mean"], hparams["dataset_std"], hparams["ssim_data_range"])
    dataset = CellsDataset(cfg.CELLS_DIR, augment=False, flatten=True, stats=stats)

    if args.index is not None:
        idx = args.index
    elif args.cell is not None:
        names = {p.name: i for i, p in enumerate(dataset.paths)}
        if args.cell not in names:
            raise SystemExit(f"No existe la célula {args.cell}")
        idx = names[args.cell]
    else:
        idx = pick_sample(encoder, decoder, dataset, device, args.pool, args.n_atoms)

    rec_mode = hparams.get("rec_mode", "?")
    code_dim = hparams.get("code_dim", cfg.CODE_DIM)
    ckpt_label = args.ckpt.stem
    out = args.out or (cfg.exp_vis_dir(rec_mode, code_dim, "sparse_coding")
                       / f"{rec_mode}_{code_dim}_sparse_coding_simple_{ckpt_label}.png")

    build_figure(encoder, decoder, hparams, dataset, idx, out, device, args.n_atoms)


if __name__ == "__main__":
    main()
