"""
Sparsity analysis for the experiments chapter.

5.4.1 (sec:exp-active-dist) — Distribution of active atoms per sample:
    a histogram of per-patch ℓ0 over the test set, not just its mean, for the
    six configurations (3 losses × dims 128/1024), with best and last overlaid.

5.4.2 (sec:exp-tradeoff) — Reconstruction-sparsity tradeoff:
    reconstruction quality versus the degree of sparsity achieved. The sparsity
    axis is frac_zeros (fraction of zero coefficients), which is comparable
    across dimensions, unlike absolute ℓ0. The training trajectory is drawn in
    the background when a trajectory.json is available.

Per-sample values aren't stored in evaluate.py's JSON output (only their
statistics), so they're recomputed by running the test set through each
encoder and then cached.

Outputs in experiments/7_eval_plots/global_comparison/:
    sparsity_l0_distribution.png     (5.4.1)
    tradeoff_rec_sparsity.png        (5.4.2)
    sparsity_per_sample.json         (cache of per-sample ℓ0/frac_zeros)

Usage:
    python evaluation/sparsity_analysis.py
    python evaluation/sparsity_analysis.py --recompute
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core import configuration as cfg
from core.dataset import CellsDataset
from core.model   import LinearLISTAEncoder, LinearLISTADecoder

CONFIGS   = [(m, d) for m in ["mse", "mae", "ssim"] for d in [128, 1024]]
CKPTS     = ["best", "last"]
COLORS    = {"mse": "#4C72B0", "mae": "#DD8452", "ssim": "#55A868"}
CK_STYLE  = {"best": dict(alpha=0.8), "last": dict(alpha=0.45)}
MARKERS   = {128: "o", 1024: "s"}


@torch.no_grad()
def per_sample_sparsity(ckpt_path: Path, dataset, test_indices, device):
    """Per-sample ℓ0 and frac_zeros, plus mean Gaussian MSE, on the test set."""
    from core.loss import gaussian_center_mask

    ckpt     = torch.load(ckpt_path, map_location=device)
    hparams  = ckpt.get("hparams", {})
    code_dim = hparams.get("code_dim", cfg.CODE_DIM)
    encoder  = LinearLISTAEncoder(cfg.IN_DIM, code_dim,
                                  hparams.get("num_iters", cfg.NUM_ITERS)).to(device)
    decoder  = LinearLISTADecoder(encoder).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval(); decoder.eval()

    loader = DataLoader(Subset(dataset, test_indices), batch_size=cfg.EVAL_BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
    H = W = 64; C = 3
    w = gaussian_center_mask(device=device).view(1, 1, H, W).expand(1, C, H, W)

    l0, fz, mse, labels = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        z  = encoder(xb)
        xh = decoder(z)
        B  = xb.size(0)
        nz = (z > 0)
        l0.extend(nz.float().sum(dim=1).cpu().numpy().tolist())
        fz.extend((~nz).float().mean(dim=1).cpu().numpy().tolist())
        diff = (xh.view(B, C, H, W) - xb.view(B, C, H, W)) ** 2
        mse.extend((w.expand(B, C, H, W) * diff).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        labels.extend(yb.numpy().tolist())

    return {"l0": l0, "frac_zeros": fz, "mse": mse, "labels": labels,
            "code_dim": int(code_dim), "epoch": ckpt.get("epoch", -1)}


def gather(cache_path: Path, recompute: bool, device):
    if cache_path.exists() and not recompute:
        print(f"Usando caché: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    dataset = CellsDataset(cfg.CELLS_DIR, augment=False, flatten=True)
    data = {}
    for rec_mode, code_dim in CONFIGS:
        split_path = cfg.exp_split_path(rec_mode, code_dim)
        if not split_path.exists():
            print(f"  [Aviso] sin split.json para {rec_mode}/dim_{code_dim}")
            continue
        with open(split_path) as f:
            test_indices = json.load(f)["test_indices"]
        for ck in CKPTS:
            path = cfg.exp_checkpoints_dir(rec_mode, code_dim, ck) / f"{ck}.pt"
            if not path.exists():
                print(f"  [Aviso] falta {path}")
                continue
            print(f"  {rec_mode}/dim_{code_dim} [{ck}] ...")
            data[f"{rec_mode}_{code_dim}_{ck}"] = per_sample_sparsity(
                path, dataset, test_indices, device)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"Caché guardada: {cache_path}")
    return data


def plot_l0_distribution(data, out_path):
    """5.4.1 — Histogram of per-sample ℓ0, one panel per configuration."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle("Distribución de átomos activos por muestra (ℓ₀) — test", fontsize=15)

    for ax, (rec_mode, code_dim) in zip(axes.ravel(), CONFIGS):
        for ck in CKPTS:
            entry = data.get(f"{rec_mode}_{code_dim}_{ck}")
            if entry is None:
                continue
            ax.hist(np.asarray(entry["l0"]), bins=40, label=ck,
                    color=COLORS[rec_mode], **CK_STYLE[ck])
        ax.set_title(f"{rec_mode} / dim_{code_dim}", fontsize=12, fontweight="bold")
        ax.set_xlabel("átomos activos por muestra (ℓ₀)")
        ax.set_ylabel("nº de parches")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado: {out_path}")


def _trajectory(rec_mode, code_dim):
    p = (cfg.exp_metrics_dir(rec_mode, code_dim, "reconstruction", "checkpoint_evolution")
         / "trajectory.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def plot_tradeoff(data, out_path):
    """5.4.2 — Reconstruction versus sparsity achieved."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Compromiso reconstrucción – dispersión (test)", fontsize=16)

    for ax, (ykey, ylabel, better) in zip(
            axes,
            [("mse_mean", "MSE gaussiana  (↓ mejor)", "min"),
             ("ssim_mean", "SSIM gaussiano  (↑ mejor)", "max")]):

        for i, (rec_mode, code_dim) in enumerate(CONFIGS):
            traj = _trajectory(rec_mode, code_dim)
            if traj:
                xs = [t["frac_zeros"] for t in traj]
                ys = [t[ykey] for t in traj]
                order = np.argsort(xs)
                ax.plot(np.asarray(xs)[order], np.asarray(ys)[order],
                        color=COLORS[rec_mode], alpha=0.18, lw=1.0, zorder=1)

            pts = {}
            for ck in CKPTS:
                entry = data.get(f"{rec_mode}_{code_dim}_{ck}")
                if entry is None:
                    continue
                x = float(np.mean(entry["frac_zeros"]))
                y = (float(np.mean(entry["mse"])) if ykey == "mse_mean"
                     else _ssim_from_traj(traj, ck))
                if y is None:
                    continue
                pts[ck] = (x, y)
                ax.scatter(x, y, s=180 if ck == "best" else 110,
                           marker=MARKERS[code_dim], color=COLORS[rec_mode],
                           edgecolor="black", linewidth=1.2,
                           alpha=1.0 if ck == "best" else 0.5, zorder=3)

            # Arrow best → last: shows which way training moves the tradeoff.
            if "best" in pts and "last" in pts:
                (x0, y0), (x1, y1) = pts["best"], pts["last"]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                            arrowprops=dict(arrowstyle="->", color=COLORS[rec_mode],
                                            lw=1.4, alpha=0.7,
                                            shrinkA=7, shrinkB=7))
            # One label per configuration, alternating sides to avoid overlap.
            if "best" in pts:
                dx, dy = (10, 10) if i % 2 == 0 else (10, -16)
                ax.annotate(f"{rec_mode}·{code_dim}", pts["best"],
                            textcoords="offset points", xytext=(dx, dy),
                            fontsize=9, fontweight="bold", color=COLORS[rec_mode])

        ax.set_xlabel("grado de dispersión —  frac_zeros  (↑ más disperso)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ykey == "mse_mean":
            # Collapse pushes MSE to ~1.3 (and the trajectory to ~9): on a linear
            # scale it flattens the whole useful range (0.06-0.18) against the axis.
            ax.set_yscale("log")

    handles = [plt.Line2D([], [], color=c, lw=6, label=m) for m, c in COLORS.items()]
    handles += [plt.Line2D([], [], color="gray", marker=mk, ls="", markersize=10,
                           label=f"dim {d}") for d, mk in MARKERS.items()]
    handles += [plt.Line2D([], [], color="gray", lw=1.2, alpha=0.35,
                           label="trayectoria de entrenamiento")]
    axes[0].legend(handles=handles, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Guardado: {out_path}")


def _ssim_from_traj(traj, ck):
    """SSIM for best/last from trajectory.json (not recomputed per sample)."""
    if not traj:
        return None
    for t in traj:
        if t.get("label") == ck:
            return t.get("ssim_mean")
    return None


def main():
    p = argparse.ArgumentParser(description="Secciones 5.4.1 y 5.4.2 (dispersión).")
    p.add_argument("--recompute", action="store_true",
                   help="Recalcula ℓ0 por muestra aunque exista la caché.")
    p.add_argument("--tradeoff", action="store_true",
                   help="Añade la figura 5.4.2 (compromiso reconstrucción-dispersión).")
    args = p.parse_args()

    out_dir = cfg.exp_comparativa_dir("7_eval_plots")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    data = gather(out_dir / "sparsity_per_sample.json", args.recompute, device)
    if not data:
        print("[Error] sin datos")
        return
    plot_l0_distribution(data, out_dir / "sparsity_l0_distribution.png")
    if args.tradeoff:
        plot_tradeoff(data, out_dir / "tradeoff_rec_sparsity.png")


if __name__ == "__main__":
    main()
