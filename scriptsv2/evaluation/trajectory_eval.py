"""
Trajectory evaluation — metrics across all checkpoints (best, jump_*, last)
for a given configuration (rec_mode, code_dim).

Outputs in experiments/2_metrics/<rec_mode>/dim_<code_dim>/reconstruction/checkpoint_evolution/
      and experiments/3_visualization/<rec_mode>/dim_<code_dim>/

Usage:
    python evaluation/trajectory_eval.py --rec_mode mse --code_dim 128
    python evaluation/trajectory_eval.py --rec_mode mae --code_dim 1024 --split full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from core import configuration as cfg
from core.dataset import CellsDataset
from core.loss    import gaussian_center_mask, sparsity_metrics, _ssim_kernel, _ssim_map
from core.model   import LinearLISTAEncoder, LinearLISTADecoder

H = W = 64
C = 3
CLASS_NAMES = ["mitotic_figure", "not_mitotic_figure"]


def discover_checkpoints(rec_mode: str, code_dim: int, max_jumps: int = 0) -> list:
    """Return [(ckpt_path, label), ...] sorted by epoch.

    Includes best.pt, last.pt, and the checkpoint.pt from each jump_*/
    subdirectory under jumps/.

    max_jumps > 0 subsamples the jumps uniformly over epoch (keeping the
    first and last). dim_1024 has hundreds of jumps at ~160 MB each, so
    evaluating all of them means tens of GB and an unreadable mosaic; best
    and last are never dropped.
    """
    jumps_dir = cfg.exp_checkpoints_dir(rec_mode, code_dim, "jumps")
    best_path = cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt"
    last_path = cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt"

    jump_entries = []
    if jumps_dir.exists():
        for d in jumps_dir.iterdir():
            if d.is_dir() and d.name.startswith("jump_"):
                ckpt = d / "checkpoint.pt"
                if ckpt.exists():
                    epoch = int(d.name.rsplit("_", 1)[-1])
                    jump_entries.append((epoch, ckpt, d.name))
    jump_entries.sort()

    if max_jumps > 0 and len(jump_entries) > max_jumps:
        idx = np.linspace(0, len(jump_entries) - 1, max_jumps).round().astype(int)
        idx = sorted(set(idx.tolist()))
        print(f"  Submuestreo de jumps: {len(jump_entries)} → {len(idx)} "
              f"(uniforme en epoch)")
        jump_entries = [jump_entries[i] for i in idx]

    entries = [(ckpt, label) for _, ckpt, label in jump_entries]
    for path, name in [(best_path, "best"), (last_path, "last")]:
        if path.exists():
            entries.append((path, name))

    return entries


@torch.no_grad()
def _load_model(ckpt_path: Path, device):
    ckpt     = torch.load(ckpt_path, map_location=device)
    hparams  = ckpt.get("hparams", {})
    code_dim = hparams.get("code_dim", cfg.CODE_DIM)
    encoder  = LinearLISTAEncoder(
        cfg.IN_DIM, code_dim, hparams.get("num_iters", cfg.NUM_ITERS)
    ).to(device)
    decoder  = LinearLISTADecoder(encoder).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, hparams, ckpt.get("epoch", 0), ckpt.get("jump_meta", {})


def _mse_gauss(x, x_hat) -> np.ndarray:
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()


def _mae_gauss(x, x_hat) -> np.ndarray:
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)).abs()).mean(dim=[1, 2, 3]).cpu().numpy()


def _ssim_gauss(x, x_hat, data_range: float) -> np.ndarray:
    B      = x.size(0)
    x4d    = x.view(B, C, H, W)
    xh4d   = x_hat.view(B, C, H, W)
    kernel = _ssim_kernel(cfg.SSIM_KERNEL_SIZE, cfg.SSIM_SIGMA, C, x.device)
    pad    = cfg.SSIM_KERNEL_SIZE // 2
    C1     = (cfg.SSIM_K1 * data_range) ** 2
    C2     = (cfg.SSIM_K2 * data_range) ** 2
    sm     = _ssim_map(x4d, xh4d, kernel, pad, C1, C2)
    w      = gaussian_center_mask(device=x.device).view(1, 1, H, W)
    return (sm * w).mean(dim=[1, 2, 3]).cpu().numpy()


def fisher_index(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.zeros(Z.shape[1])
    Za    = Z[labels == classes[0]]
    Zb    = Z[labels == classes[1]]
    denom = Za.var(axis=0) + Zb.var(axis=0) + 1e-12
    return (Za.mean(axis=0) - Zb.mean(axis=0)) ** 2 / denom


@torch.no_grad()
def compute_checkpoint_metrics(ckpt_path: Path, dataset, indices: list,
                                data_range: float, device) -> dict:
    encoder, decoder, hparams, epoch, jump_meta = _load_model(ckpt_path, device)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size  = cfg.EVAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    all_mse, all_mae, all_ssim = [], [], []
    all_l0, all_fz = [], []
    all_labels, all_Z = [], []

    for xb, yb in loader:
        xb    = xb.to(device, non_blocking=True)
        z     = encoder(xb)
        x_hat = decoder(z)

        all_mse.extend(_mse_gauss(xb, x_hat).tolist())
        all_mae.extend(_mae_gauss(xb, x_hat).tolist())
        all_ssim.extend(_ssim_gauss(xb, x_hat, data_range).tolist())
        all_l0.extend((z > 0).float().sum(dim=1).cpu().numpy().tolist())
        fz, _ = sparsity_metrics(z)
        all_fz.append(fz)
        all_labels.extend(yb.numpy().tolist())
        all_Z.append(z.cpu().numpy())

    mse_arr    = np.array(all_mse)
    mae_arr    = np.array(all_mae)
    ssim_arr   = np.array(all_ssim)
    l0_arr     = np.array(all_l0)
    labels_arr = np.array(all_labels)
    Z          = np.vstack(all_Z)

    per_class = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = labels_arr == cls_id
        if mask.sum() > 0:
            per_class[cls_name] = {
                "n":         int(mask.sum()),
                "mse_mean":  float(mse_arr[mask].mean()),
                "mae_mean":  float(mae_arr[mask].mean()),
                "ssim_mean": float(ssim_arr[mask].mean()),
                "l0_mean":   float(l0_arr[mask].mean()),
            }

    fi       = fisher_index(Z, labels_arr)
    fi_max   = float(fi.max())
    fi_mean  = float(fi.mean())
    fi_order = np.argsort(fi)[::-1]
    a1       = int(fi_order[0])
    a2       = int(fi_order[1]) if Z.shape[1] > 1 else a1

    n_comp  = min(2, Z.shape[1])
    pca     = PCA(n_components=n_comp, random_state=42)
    pca_2d  = pca.fit_transform(Z)
    pca_var = float(pca.explained_variance_ratio_.sum())

    # 2D projections for the evolution scatterplots (not serialized to JSON)
    viz = {
        "labels":       labels_arr,
        "pca_2d":       pca_2d,
        "fisher_2d":    Z[:, [a1, a2]],
        "fisher_atoms": (a1, a2),
    }

    mse_mit  = per_class.get("mitotic_figure",     {}).get("mse_mean", float("nan"))
    mse_nmit = per_class.get("not_mitotic_figure", {}).get("mse_mean", float("nan"))
    rec_bias = float(mse_mit - mse_nmit)

    return {
        "checkpoint":   str(ckpt_path),
        "label":        ckpt_path.stem,
        "epoch":        epoch,
        "jump_meta":    jump_meta,
        "code_dim":     int(encoder.code_dim),
        "mse_mean":     float(mse_arr.mean()),
        "mae_mean":     float(mae_arr.mean()),
        "ssim_mean":    float(ssim_arr.mean()),
        "l0_mean":      float(l0_arr.mean()),
        "l0_std":       float(l0_arr.std()),
        "frac_zeros":   float(np.mean(all_fz)),
        "fisher_max":   fi_max,
        "fisher_mean":  fi_mean,
        "pca_var_12":   pca_var,
        "rec_bias_mse": rec_bias,
        "per_class":    per_class,
    }, viz


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _ckpt_color(label: str) -> str:
    if label == "best":              return "gold"
    if label == "last":              return "black"
    parts = label.split("_")
    if "up"     in parts:            return "tomato"
    if "down"   in parts:            return "seagreen"
    if "sparse" in parts:            return "mediumpurple"
    return "steelblue"


def _ax_base(ax, x, labels_str, title: str, ylabel: str):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_str, rotation=45, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.12)


def _line_with_dots(ax, x, trajectory, key: str, line_color: str):
    vals = [m[key] for m in trajectory]
    ax.plot(x, vals, "-", color=line_color, linewidth=1.4, alpha=0.5)
    for xi, vi, m in zip(x, vals, trajectory):
        ax.plot(xi, vi, "o", color=_ckpt_color(m["label"]), markersize=8, zorder=5)


CLASS_SHORT  = ["Mitótica", "No-mitótica"]
CLASS_COLORS = ["crimson", "steelblue"]


def _scatter_montage(trajectory: list, viz_list: list, key: str,
                     out_path: Path, run_label: str, title: str,
                     subtitle_fn, max_points: int = 2000):
    """Grid of 2D scatterplots (one per checkpoint, ordered by epoch), colored
    by class. `key` selects the projection ('pca_2d' or 'fisher_2d')."""
    n    = len(trajectory)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))

    # Fixed subsampling (same indices across all checkpoints → comparable)
    labels = viz_list[0]["labels"]
    N      = len(labels)
    rng    = np.random.default_rng(42)
    sel    = np.sort(rng.choice(N, size=min(max_points, N), replace=False))
    lbl_s  = labels[sel]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    fig.suptitle(f"{title} — {run_label}", fontsize=13)
    axes = np.atleast_1d(axes).ravel()

    for ax, m, v in zip(axes, trajectory, viz_list):
        coords = v[key][sel]
        for cls_id in (0, 1):
            mask = lbl_s == cls_id
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLASS_COLORS[cls_id], s=4, alpha=0.4, linewidths=0)
        ax.set_title(f"{m['label']}\n{subtitle_fn(m, v)}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.15)

    for ax in axes[n:]:
        ax.axis("off")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[i],
                      markersize=8, label=CLASS_SHORT[i]) for i in (0, 1)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save(fig, out_path)


def plot_scatter_evolution(trajectory: list, viz_list: list,
                           vis_dir: Path, run_label: str = ""):
    """Two feature-scatter montages across checkpoints: code PCA-2D and the
    top-2 Fisher atoms."""
    out_dir = vis_dir / "feature_scatter" / "checkpoint_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    _scatter_montage(
        trajectory, viz_list, "pca_2d",
        out_dir / "scatter_pca_evolution.png", run_label,
        "Scatter PCA-2D del código (evolución)",
        lambda m, v: f"ep {m['epoch']}  var={m['pca_var_12']:.2f}",
    )
    _scatter_montage(
        trajectory, viz_list, "fisher_2d",
        out_dir / "scatter_fisher_evolution.png", run_label,
        "Scatter top-2 átomos Fisher (evolución)",
        lambda m, v: f"ep {m['epoch']}  átomos {v['fisher_atoms']}  F={m['fisher_max']:.3f}",
    )


def plot_trajectory(trajectory: list, vis_dir: Path, spar_dir: Path, run_label: str = ""):
    x          = np.arange(len(trajectory))
    labels_str = [m["label"] for m in trajectory]

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(trajectory) * 0.7 + 2), 11))
    fig.suptitle(f"Métricas de reconstrucción — {run_label}", fontsize=12)
    for ax, key, title in [
        (axes[0], "mse_mean",  "MSE gaussiano (↓)"),
        (axes[1], "mae_mean",  "MAE gaussiano (↓)"),
        (axes[2], "ssim_mean", "SSIM gaussiano (↑)"),
    ]:
        _line_with_dots(ax, x, trajectory, key, "steelblue")
        _ax_base(ax, x, labels_str, title, title.split()[0])
    plt.tight_layout()
    rec_dir = vis_dir / "reconstructions" / "checkpoint_evolution"
    rec_dir.mkdir(parents=True, exist_ok=True)
    spar_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, rec_dir / "plot_rec_metrics.png")

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(trajectory) * 0.7 + 2), 8))
    fig.suptitle(f"Sparsity — {run_label}", fontsize=12)
    for ax, key, title in [
        (axes[0], "l0_mean",    "L0 activos (media)"),
        (axes[1], "frac_zeros", "Fracción de ceros"),
    ]:
        _line_with_dots(ax, x, trajectory, key, "darkorange")
        _ax_base(ax, x, labels_str, title, title)
    plt.tight_layout()
    _save(fig, spar_dir / "plot_sparsity.png")

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(trajectory) * 0.7 + 2), 8))
    fig.suptitle(f"Discriminabilidad — {run_label}", fontsize=12)
    for ax, key, title in [
        (axes[0], "fisher_max", "Fisher máx (↑)"),
        (axes[1], "pca_var_12", "Varianza explicada PC1+PC2 (↑)"),
    ]:
        _line_with_dots(ax, x, trajectory, key, "mediumpurple")
        _ax_base(ax, x, labels_str, title, title)
    plt.tight_layout()
    _save(fig, rec_dir / "plot_discriminability.png")

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(trajectory) * 0.7 + 2), 8))
    fig.suptitle(f"MSE por clase — {run_label}", fontsize=12)

    ax = axes[0]
    mit_mse  = [m["per_class"].get("mitotic_figure",     {}).get("mse_mean", float("nan"))
                for m in trajectory]
    nmit_mse = [m["per_class"].get("not_mitotic_figure", {}).get("mse_mean", float("nan"))
                for m in trajectory]
    ax.plot(x, mit_mse,  "o-", color="crimson",   label="Mitótica",    linewidth=1.4)
    ax.plot(x, nmit_mse, "o-", color="steelblue", label="No mitótica", linewidth=1.4)
    ax.legend(fontsize=9)
    _ax_base(ax, x, labels_str, "MSE gaussiano por clase", "MSE")

    ax = axes[1]
    bias   = [m["rec_bias_mse"] for m in trajectory]
    colors_bar = ["tomato" if b > 0 else "seagreen" for b in bias]
    ax.bar(x, bias, color=colors_bar, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    _ax_base(ax, x, labels_str, "Sesgo MSE: mitótica − no mitótica", "Sesgo")
    plt.tight_layout()
    _save(fig, rec_dir / "plot_rec_bias.png")

    panels = [
        ("mse_mean",   "MSE gaussiano (↓)",   "steelblue"),
        ("ssim_mean",  "SSIM gaussiano (↑)",   "steelblue"),
        ("l0_mean",    "L0 activos",            "darkorange"),
        ("frac_zeros", "Fracción de ceros",     "darkorange"),
        ("fisher_max", "Fisher máx (↑)",        "mediumpurple"),
        ("pca_var_12", "Varianza PC1+PC2 (↑)",  "mediumpurple"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(max(14, len(trajectory) * 0.8 + 2), 12))
    fig.suptitle(f"Trayectoria — {run_label}", fontsize=13)
    for ax, (key, title, color) in zip(axes.flat, panels):
        _line_with_dots(ax, x, trajectory, key, color)
        _ax_base(ax, x, labels_str, title, "")
    plt.tight_layout()
    _save(fig, rec_dir / "plot_overview.png")

    _save_legend(rec_dir)


def _save_legend(out_dir: Path):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",         markersize=10, label="best"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",        markersize=10, label="last"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",       markersize=10, label="jump_up"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",     markersize=10, label="jump_down"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="mediumpurple", markersize=10, label="jump_sparse"),
    ]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.legend(handles=handles, loc="center", fontsize=10, frameon=True)
    ax.axis("off")
    _save(fig, out_dir / "legend.png")


def run_trajectory(rec_mode: str, code_dim: int, cells_dir: Path, split: str,
                   max_jumps: int = 0) -> list | None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = discover_checkpoints(rec_mode, code_dim, max_jumps)
    if not checkpoints:
        print(f"[Aviso] No se encontraron checkpoints para {rec_mode}/dim_{code_dim}")
        return None

    print(f"\n{'='*60}")
    print(f"Trajectory eval: {rec_mode}/dim_{code_dim}  ({len(checkpoints)} checkpoints)")
    print(f"{'='*60}")

    first_ckpt = checkpoints[0][0]
    first_raw  = torch.load(first_ckpt, map_location="cpu")
    hparams    = first_raw.get("hparams", {})
    data_range = hparams.get("ssim_data_range", None)

    dataset = CellsDataset(cells_dir, augment=False, flatten=True)
    if data_range is None:
        data_range = dataset.data_range

    N          = len(dataset)
    split_path = cfg.exp_split_path(rec_mode, code_dim)

    if split == "full":
        indices = list(range(N))
    elif split_path.exists():
        with open(split_path) as f:
            sdata = json.load(f)
        indices = sdata["test_indices"] if split == "test" else sdata["train_indices"]
    else:
        indices = first_raw.get("test_indices", None)
        if indices is None or split != "test":
            raise RuntimeError(
                f"No se encontraron índices para split='{split}' en {split_path}"
            )

    print(f"Dataset: {N} parches  |  split={split}  |  {len(indices)} muestras")
    print(f"data_range={data_range:.4f}  |  device={device}")

    pairs = []   # [(metrics, viz), ...]
    for ckpt_path, label in checkpoints:
        print(f"\n  [{label}]  {ckpt_path.name}")
        try:
            metrics, viz     = compute_checkpoint_metrics(
                ckpt_path, dataset, indices, data_range, device
            )
            metrics["label"] = label
            pairs.append((metrics, viz))
            print(
                f"    epoch={metrics['epoch']:>6}  "
                f"mse={metrics['mse_mean']:.5f}  "
                f"ssim={metrics['ssim_mean']:.5f}  "
                f"l0={metrics['l0_mean']:.1f}  "
                f"fisher_max={metrics['fisher_max']:.4f}  "
                f"pca_var={metrics['pca_var_12']:.3f}"
            )
        except Exception as e:
            print(f"    [ERROR] {e}")

    if not pairs:
        return None

    pairs.sort(key=lambda p: p[0]["epoch"])
    trajectory = [m for m, _ in pairs]
    viz_list   = [v for _, v in pairs]

    evo_dir = cfg.exp_metrics_dir(rec_mode, code_dim,
                                   "reconstruction", "checkpoint_evolution")
    traj_path = evo_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\n  Trajectory JSON: {traj_path}")

    vis_dir  = cfg.exp_vis_dir(rec_mode, code_dim)
    spar_dir = cfg.exp_metrics_dir(rec_mode, code_dim, "sparsity", "checkpoint_evolution")
    plot_trajectory(trajectory, vis_dir, spar_dir, f"{rec_mode}/dim_{code_dim}")
    plot_scatter_evolution(trajectory, viz_list, vis_dir, f"{rec_mode}/dim_{code_dim}")

    return trajectory


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluación de trayectoria sobre todos los checkpoints de una configuración."
    )
    p.add_argument("--rec_mode",  type=str, required=True, choices=["mse", "mae", "ssim"])
    p.add_argument("--code_dim",  type=int, default=cfg.CODE_DIM)
    p.add_argument("--cells_dir", type=str, default=None)
    p.add_argument("--split",     type=str, default="test",
                   choices=["full", "train", "test"])
    p.add_argument("--max_jumps", type=int, default=0,
                   help="Submuestrea los jumps a N puntos uniformes en epoch "
                        "(0 = todos). best y last se conservan siempre.")
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ no encontrado: {cells_dir}")

    run_trajectory(args.rec_mode, args.code_dim, cells_dir, args.split, args.max_jumps)


if __name__ == "__main__":
    main()
