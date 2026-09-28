"""
Illustrative figure of LISTA sparse coding on real cells (MIDOG).

    Top row:    [x original] | [z sparse code] = [x̂ = W_d z reconstruction]
    Bottom row: x̂ ≈ Σ_k z_k w_k  →  top-N atoms with their coefficient  →  = x̂

Usage:
    python visualization/sparse_coding_figure.py                      # automatic cell
    python visualization/sparse_coding_figure.py --cell img123_ann000456.png
    python visualization/sparse_coding_figure.py --index 4207 --n_atoms 6
    python visualization/sparse_coding_figure.py --ckpt /path/best.pt --out fig.png
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

DEFAULT_CKPT = Path("/mnt/disk1/enrique.pardo.garcia/1_checkpoints/ssim/dim_1024/best/best.pt")

# Atom color by rank (1st most active → 6th)
RANK_COLORS = ["#c0392b", "#2b6cb0", "#4a90d9", "#e8944a", "#eda45c", "#f0b070"]
BAR_COLOR   = "#8ab4de"   # remaining active coefficients
SEED = 42


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def pick_sample(encoder, decoder, dataset, device, pool_size, n_atoms):
    """Representative mitotic cell: lowest MSE among a deterministic pool."""
    mitotic = [i for i in range(len(dataset)) if int(dataset.labels[i]) == 0]
    rng     = np.random.default_rng(SEED)
    pool    = rng.choice(mitotic, size=min(pool_size, len(mitotic)), replace=False)

    best = None
    for i in pool:
        x = dataset[int(i)][0].to(device)
        z = encoder(x.unsqueeze(0)).squeeze(0)
        xh = decoder(z.unsqueeze(0)).squeeze(0)
        nnz = int((z > 0).sum())
        if nnz < n_atoms:
            continue
        mse = float((x - xh).pow(2).mean())
        if best is None or mse < best[0]:
            best = (mse, int(i))
    return best[1]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_figure(encoder, decoder, hparams, epoch, dataset, idx, out_path,
                 device, n_atoms, ckpt_label):
    mean = hparams.get("dataset_mean", 0.0)
    std  = hparams.get("dataset_std",  1.0)

    x, label = dataset[idx]
    x  = x.to(device)
    z  = encoder(x.unsqueeze(0)).squeeze(0)
    xh = decoder(z.unsqueeze(0)).squeeze(0)

    z_np = z.detach().cpu().numpy()
    D    = len(z_np)
    mse  = float((x - xh).pow(2).mean())

    active = np.where(z_np > 0)[0]
    active = active[np.argsort(z_np[active])[::-1]]        # descending order
    n_show = min(n_atoms, len(active))
    top    = active[:n_show]

    W_e = encoder.W_e.weight.detach().cpu()                # (D, IN_DIM); W_d = W_e^T

    # ── Canvas ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19.5, 10.6))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.30,
                           left=0.035, right=0.985, top=0.855, bottom=0.055)

    # Top row: original | code | reconstruction
    gs_top   = gs[0].subgridspec(1, 3, width_ratios=[1.0, 2.35, 1.0], wspace=0.26)
    ax_orig  = fig.add_subplot(gs_top[0, 0])
    ax_code  = fig.add_subplot(gs_top[0, 1])
    ax_recon = fig.add_subplot(gs_top[0, 2])

    # x — original
    ax_orig.imshow(to_img(x.cpu(), mean, std))
    ax_orig.set_title(r"$\mathbf{x}$ — original", fontsize=13, pad=10)
    ax_orig.axis("off")
    cls = "célula mitótica" if int(label) == 0 else "célula no mitótica"
    ax_orig.text(0.5, -0.06, f"{cls} ({cfg.PATCH_SIZE}×{cfg.PATCH_SIZE} px)",
                 transform=ax_orig.transAxes, ha="center", va="top",
                 fontsize=9, color="#666666")

    # z — sparse code
    colors = np.array([BAR_COLOR] * D, dtype=object)
    for r, k in enumerate(top):
        colors[k] = RANK_COLORS[r % len(RANK_COLORS)]
    ax_code.bar(np.arange(D), z_np, color=list(colors), width=1.0)
    ax_code.axhline(0, color="black", lw=0.6)

    y_top = float(z_np.max()) * 1.30
    ax_code.set_ylim(0, y_top)
    ax_code.set_xlim(-8, D + 8)

    # Labels for the 3 largest coefficients: staggered height and minimum
    # horizontal spacing, so they don't overlap when atoms are adjacent.
    lab = sorted(top[:3], key=int)
    sep = 0.17 * D
    xs  = [float(k) for k in lab]
    for a in range(1, len(xs)):
        xs[a] = max(xs[a], xs[a - 1] + sep)
    shift = max(0.0, xs[-1] - 0.92 * D)
    xs = [min(max(v - shift, 0.08 * D), 0.92 * D) for v in xs]

    for r, (k, x_lab) in enumerate(zip(lab, xs)):
        col = RANK_COLORS[list(top).index(k) % len(RANK_COLORS)]
        ax_code.vlines(k, 0, y_top, color=col, lw=0.9, alpha=0.30, zorder=0)
        ax_code.annotate(rf"$z_{{{k}}} = +{z_np[k]:.2f}$",
                         xy=(k, z_np[k]), xytext=(x_lab, y_top * (0.97 - 0.08 * r)),
                         ha="center", va="bottom", fontsize=9.5, color=col,
                         arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                         alpha=0.55, shrinkB=3))

    pct = 100.0 * len(active) / D
    ax_code.set_title(
        rf"$\mathbf{{z}} \in \mathbb{{R}}^{{{D}}}$ — código disperso" "\n"
        f"{len(active)} activos / {D} átomos  ({pct:.1f}% no nulo)",
        fontsize=13, pad=10)
    ax_code.set_xlabel(r"índice de átomo $k$", fontsize=11)
    ax_code.set_ylabel(r"coeficiente $z_k$", fontsize=11)
    ax_code.grid(True, axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax_code.spines[side].set_visible(False)

    # x̂ — reconstruction
    ax_recon.imshow(to_img(xh.cpu(), mean, std))
    ax_recon.set_title(r"$\hat{\mathbf{x}} = \mathbf{W}_d\mathbf{z}$ — reconstrucción"
                       "\n" f"MSE = {mse:.4f}", fontsize=13, pad=10)
    ax_recon.axis("off")

    # "=" between the code and the reconstruction (the "≈" next to the y-axis was removed)
    p_code, p_rec = ax_code.get_position(), ax_recon.get_position()
    fig.text((p_code.x1 + p_rec.x0) / 2.0, (p_rec.y0 + p_rec.y1) / 2.0,
             "=", fontsize=30, ha="center", va="center")

    # ── Bottom row: formula + atoms + x̂ ────────────────────────────────────
    gs_bot = gs[1].subgridspec(1, n_show + 2,
                               width_ratios=[1.85] + [1.0] * n_show + [1.0],
                               wspace=0.16)

    ax_eq = fig.add_subplot(gs_bot[0, 0])
    ax_eq.axis("off")
    ax_eq.text(0.5, 0.5, r"$\hat{\mathbf{x}} \;\approx\; \sum_k z_k\, \mathbf{w}_k$",
               transform=ax_eq.transAxes, ha="center", va="center", fontsize=46)

    for r, k in enumerate(top):
        ax  = fig.add_subplot(gs_bot[0, r + 1])
        col = RANK_COLORS[r % len(RANK_COLORS)]
        ax.imshow(atom_img(W_e[k]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.8)
        ax.set_title(rf"$\mathbf{{w}}_{{{k}}}$" "\n" rf"$\times\ +{z_np[k]:.2f}$",
                     fontsize=10.5, color=col, pad=6)

    ax_out = fig.add_subplot(gs_bot[0, n_show + 1])
    ax_out.imshow(to_img(xh.cpu(), mean, std))
    ax_out.set_title(r"$=\ \hat{\mathbf{x}}$", fontsize=13, pad=6)
    ax_out.axis("off")

    fig.suptitle(
        "Codificación dispersa LISTA sobre células histológicas reales (MIDOG) — "
        rf"$\mathbf{{W}}_d \in \mathbb{{R}}^{{n \times {D}}}$,"
        f"  checkpoint {ckpt_label} (epoch {epoch})",
        fontsize=15, y=0.955)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {out_path}")
    print(f"  Célula   : idx={idx}  {dataset.paths[idx].name}")
    print(f"  Activos  : {len(active)}/{D}   MSE={mse:.4f}")
    print(f"  Top-{n_show}   : " + ", ".join(f"w_{k}×{z_np[k]:+.2f}" for k in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cell",    type=str, default=None,
                    help="nombre del PNG en data/cells/mitotic_figure/")
    ap.add_argument("--index",   type=int, default=None, help="índice en el dataset")
    ap.add_argument("--n_atoms", type=int, default=cfg.VIS_N_TOP_ATOMS)
    ap.add_argument("--pool",    type=int, default=400,
                    help="candidatas evaluadas en la selección automática")
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
                       / f"{rec_mode}_{code_dim}_sparse_coding_{ckpt_label}.png")

    build_figure(encoder, decoder, hparams, epoch, dataset, idx, out,
                 device, args.n_atoms, ckpt_label)


if __name__ == "__main__":
    main()
