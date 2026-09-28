"""
Feature evaluation — discriminability of atoms between mitotic and non-mitotic classes.

Analyses:
  - Mean activations per class with a difference plot
  - Fisher discriminant index per atom
  - Boxplots of the top-K discriminative atoms
  - Scatterplots: PCA, t-SNE, top-2 Fisher atoms
  - Per-class atom correlation matrices and their difference
  - Comparison of feature selection methods: Fisher, ANOVA F-score, MI

Outputs go to experiments/4_analysis/<rec_mode>/dim_<code_dim>/
         and experiments/3_visualization/<rec_mode>/dim_<code_dim>/tsne/ and pca/

Usage:
    python evaluation/feature_eval.py --rec_mode mse --code_dim 128
    python evaluation/feature_eval.py --checkpoint experiments/1_checkpoints/mse/dim_128/best/best.pt
    python evaluation/feature_eval.py --rec_mode ssim --code_dim 1024 --top_k 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from core import configuration as cfg
from core.dataset import CellsDataset
from core.model   import LinearLISTAEncoder, LinearLISTADecoder

CLASS_NAMES  = ["mitotic_figure", "not_mitotic_figure"]
CLASS_SHORT  = ["Mitotic", "Non-mitotic"]
CLASS_COLORS = ["crimson", "steelblue"]


def _load_encoder(ckpt_path: Path, device):
    ckpt     = torch.load(ckpt_path, map_location=device)
    hparams  = ckpt.get("hparams", {})
    code_dim = hparams.get("code_dim", cfg.CODE_DIM)
    encoder  = LinearLISTAEncoder(
        cfg.IN_DIM, code_dim, hparams.get("num_iters", cfg.NUM_ITERS)
    ).to(device)
    decoder = LinearLISTADecoder(encoder).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    return encoder, hparams, ckpt.get("epoch", "?")


def _load_indices(rec_mode: str, code_dim: int, ckpt_path: Path, split: str, N: int) -> list:
    split_path = cfg.exp_split_path(rec_mode, code_dim)
    if split == "full":
        return list(range(N))
    key = "test_indices" if split == "test" else "train_indices"
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)[key]
    if split == "test":
        ckpt_raw = torch.load(ckpt_path, map_location="cpu")
        idx = ckpt_raw.get("test_indices", None)
        if idx is not None:
            return idx
    raise RuntimeError(f"Indices for split='{split}' not found in {split_path}")


@torch.no_grad()
def encode_split(ckpt_path: Path, cells_dir: Path, split: str,
                 rec_mode: str, code_dim: int, device):
    encoder, hparams, epoch = _load_encoder(ckpt_path, device)
    dataset  = CellsDataset(cells_dir, augment=False, flatten=True)
    indices  = _load_indices(rec_mode, code_dim, ckpt_path, split, len(dataset))

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size  = cfg.EVAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    all_z, all_labels = [], []
    for xb, yb in loader:
        all_z.append(encoder(xb.to(device, non_blocking=True)).cpu().numpy())
        all_labels.append(yb.numpy())

    Z      = np.vstack(all_z)
    labels = np.concatenate(all_labels)
    print(f"  Encoded {len(Z)} patches  |  mitotic={int((labels==0).sum())}  non-mitotic={int((labels==1).sum())}")
    return Z, labels, hparams, epoch


def fisher_index(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    Za    = Z[labels == 0]
    Zb    = Z[labels == 1]
    denom = Za.var(axis=0) + Zb.var(axis=0) + 1e-12
    return (Za.mean(axis=0) - Zb.mean(axis=0)) ** 2 / denom


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mean_activations(Z, labels, epoch, config, out_path):
    Za   = Z[labels == 0]
    Zb   = Z[labels == 1]
    mu_a = Za.mean(axis=0)
    mu_b = Zb.mean(axis=0)
    diff = mu_a - mu_b
    code_dim = Z.shape[1]
    atoms    = np.arange(code_dim)
    step     = max(1, code_dim // 16)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, code_dim * 0.25), 8))
    fig.suptitle(f"Mean activation |z_k| per class  [{config}  epoch {epoch}]", fontsize=13)
    w = 0.4
    axes[0].bar(atoms - w/2, mu_a, width=w, color=CLASS_COLORS[0], alpha=0.75, label=CLASS_SHORT[0])
    axes[0].bar(atoms + w/2, mu_b, width=w, color=CLASS_COLORS[1], alpha=0.75, label=CLASS_SHORT[1])
    axes[0].set_title("Mean z_k by class")
    axes[0].set_xlabel("Atom k"); axes[0].set_ylabel("Mean z_k")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_xticks(atoms[::step])

    colors = [CLASS_COLORS[0] if d > 0 else CLASS_COLORS[1] for d in diff]
    axes[1].bar(atoms, diff, color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Difference: mitotic − non-mitotic")
    axes[1].set_xlabel("Atom k"); axes[1].set_ylabel("Δ mean z_k")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(atoms[::step])
    plt.tight_layout()
    _save(fig, out_path)


def plot_fisher(fi, epoch, config, top_k, out_path):
    code_dim = len(fi)
    atoms    = np.arange(code_dim)
    ranked   = np.argsort(fi)[::-1]
    step     = max(1, code_dim // 16)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Fisher Discriminant Index  [{config}  epoch {epoch}]", fontsize=13)
    axes[0].bar(atoms, fi, color="mediumpurple", alpha=0.8)
    axes[0].set_title("Fisher index — all atoms")
    axes[0].set_xlabel("Atom k"); axes[0].set_ylabel("Fisher index")
    axes[0].set_xticks(atoms[::step]); axes[0].grid(True, axis="y", alpha=0.3)

    top_idx  = ranked[:top_k]
    top_vals = fi[top_idx]
    axes[1].barh(range(top_k), top_vals[::-1], color="mediumpurple", alpha=0.8)
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([f"atom {i}" for i in top_idx[::-1]], fontsize=8)
    axes[1].set_title(f"Top-{top_k} atoms by Fisher index")
    axes[1].set_xlabel("Fisher index"); axes[1].grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_boxplots(Z, labels, fi, epoch, config, top_k, out_path):
    ranked = np.argsort(fi)[::-1][:top_k]
    ncols  = min(4, top_k)
    nrows  = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), squeeze=False)
    fig.suptitle(f"Activation distribution — top-{top_k} Fisher atoms  [{config}  epoch {epoch}]",
                 fontsize=12)
    for j, atom in enumerate(ranked):
        r, c = divmod(j, ncols)
        ax   = axes[r][c]
        bp   = ax.boxplot([Z[labels == 0, atom], Z[labels == 1, atom]],
                          labels=CLASS_SHORT, patch_artist=True,
                          medianprops=dict(color="black", linewidth=1.5))
        for box, color in zip(bp["boxes"], CLASS_COLORS):
            box.set_facecolor(color); box.set_alpha(0.65)
        ax.set_title(f"Atom {atom}  (F={fi[atom]:.3f})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(ranked), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    plt.tight_layout()
    _save(fig, out_path)


def plot_scatter_top2_fisher(Z, labels, fi, epoch, config, out_path):
    ranked = np.argsort(fi)[::-1]
    a1, a2 = int(ranked[0]), int(ranked[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Scatter — top-2 Fisher atoms  [{config}  epoch {epoch}]", fontsize=13)
    for cls_id in [0, 1]:
        mask = labels == cls_id
        ax.scatter(Z[mask, a1], Z[mask, a2], c=CLASS_COLORS[cls_id],
                   alpha=0.35, s=8, label=CLASS_SHORT[cls_id], linewidths=0)
    ax.set_xlabel(f"Atom {a1}  (F={fi[a1]:.3f})")
    ax.set_ylabel(f"Atom {a2}  (F={fi[a2]:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def _degenerate_reason(Z):
    """Return a reason if the code does not support a 2D projection, or None if it is valid.

    A collapsed dictionary produces a Z that is identically zero: the variance is zero,
    PCA divides by zero, and t-SNE's Barnes-Hut implementation receives NaN and aborts
    the process (segfault).
    """
    if not np.isfinite(Z).all():
        return "el código contiene NaN/inf"
    if Z.std(axis=0).max() <= 0:
        return "el código es constante (diccionario colapsado)"
    return None


def _plot_degenerate(reason, title, epoch, config, out_path):
    """Draw a warning figure in place of the projection, so the pipeline doesn't break."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{title}  [{config}  epoch {epoch}]", fontsize=13)
    ax.text(0.5, 0.5, f"Proyección no disponible:\n{reason}",
            ha="center", va="center", fontsize=12, color="crimson")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    _save(fig, out_path)


def plot_pca(Z, labels, epoch, config, out_path):
    reason = _degenerate_reason(Z)
    if reason:
        print(f"  [Aviso] PCA omitido: {reason}")
        _plot_degenerate(reason, "PCA — code space", epoch, config, out_path)
        return
    pca  = PCA(n_components=2, random_state=42)
    Z_2d = pca.fit_transform(Z)
    var  = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"PCA — code space  [{config}  epoch {epoch}]", fontsize=13)
    for cls_id in [0, 1]:
        mask = labels == cls_id
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=CLASS_COLORS[cls_id],
                   alpha=0.35, s=8, label=CLASS_SHORT[cls_id], linewidths=0)
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}%)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_tsne(Z, labels, epoch, config, out_path, max_samples: int = 2000):
    reason = _degenerate_reason(Z)
    if reason:
        print(f"  [Aviso] t-SNE omitido: {reason}")
        _plot_degenerate(reason, "t-SNE — code space", epoch, config, out_path)
        return
    if len(Z) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), max_samples, replace=False)
        Zs, ls = Z[idx], labels[idx]
    else:
        Zs, ls = Z, labels
    n = len(Zs)
    print(f"  t-SNE on {n} samples...")
    Z_2d = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=1000, init="pca").fit_transform(Zs)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"t-SNE — code space  [{config}  epoch {epoch}  n={n}]", fontsize=13)
    for cls_id in [0, 1]:
        mask = ls == cls_id
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=CLASS_COLORS[cls_id],
                   alpha=0.35, s=8, label=CLASS_SHORT[cls_id], linewidths=0)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def plot_correlation_matrices(Z, labels, epoch, config, out_path):
    Za     = Z[labels == 0]
    Zb     = Z[labels == 1]
    corr_a = np.corrcoef(Za.T)
    corr_b = np.corrcoef(Zb.T)
    diff   = corr_a - corr_b
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Atom correlation matrices  [{config}  epoch {epoch}]", fontsize=13)
    for ax, mat, title, cmap in zip(
        axes,
        [corr_a, corr_b, diff],
        [f"Mitotic  (n={len(Za)})", f"Non-mitotic  (n={len(Zb)})",
         "Difference (mitotic − non-mitotic)"],
        ["RdBu_r", "RdBu_r", "PiYG"],
    ):
        im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Atom j"); ax.set_ylabel("Atom i")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(fig, out_path)


def plot_feature_selection_comparison(fi, f_scores, mi_scores, epoch, config, top_k, out_path):
    code_dim = len(fi)
    atoms    = np.arange(code_dim)
    step     = max(1, code_dim // 16)

    def rank_norm(v):
        r = np.argsort(np.argsort(v)).astype(float)
        return r / max(len(r) - 1, 1)

    fi_n = rank_norm(fi)
    f_n  = rank_norm(f_scores)
    mi_n = rank_norm(mi_scores)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Feature selection methods  [{config}  epoch {epoch}]", fontsize=13)
    for ax, vals, color, title, ylabel in [
        (axes[0, 0], fi,        "mediumpurple", "Fisher Discriminant Index", "Fisher index"),
        (axes[0, 1], f_scores,  "forestgreen",  "ANOVA F-score",             "F-score"),
        (axes[1, 0], mi_scores, "steelblue",    "Mutual Information",        "MI score"),
    ]:
        ax.bar(atoms, vals, color=color, alpha=0.8)
        ax.set_title(title); ax.set_xlabel("Atom k"); ax.set_ylabel(ylabel)
        ax.set_xticks(atoms[::step]); ax.grid(True, axis="y", alpha=0.3)

    top_idx = np.argsort(fi)[::-1][:top_k]
    x_pos   = np.arange(top_k); w = 0.25
    axes[1, 1].bar(x_pos - w, fi_n[top_idx],  width=w, color="mediumpurple", alpha=0.8, label="Fisher (rank norm)")
    axes[1, 1].bar(x_pos,     f_n[top_idx],   width=w, color="forestgreen",  alpha=0.8, label="F-ANOVA (rank norm)")
    axes[1, 1].bar(x_pos + w, mi_n[top_idx],  width=w, color="steelblue",    alpha=0.8, label="MI (rank norm)")
    axes[1, 1].set_title(f"Rank comparison — top-{top_k} Fisher atoms")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f"A{i}" for i in top_idx], fontsize=7, rotation=45)
    axes[1, 1].set_ylabel("Rank (normalized)")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


def save_feature_ranking(Z, labels, fi, f_scores, mi_scores, top_k, out_path):
    code_dim = Z.shape[1]

    def top_list(ranked):
        return [{"atom": int(ranked[i]), "fisher": float(fi[ranked[i]]),
                 "f_anova": float(f_scores[ranked[i]]),
                 "mutual_info": float(mi_scores[ranked[i]])}
                for i in range(min(top_k, len(ranked)))]

    summary = {
        "n_atoms":            code_dim,
        "n_samples":          int(len(Z)),
        "n_mitotic":          int((labels == 0).sum()),
        "n_nonmitotic":       int((labels == 1).sum()),
        "top_by_fisher":      top_list(np.argsort(fi)[::-1]),
        "top_by_f_anova":     top_list(np.argsort(f_scores)[::-1]),
        "top_by_mutual_info": top_list(np.argsort(mi_scores)[::-1]),
        "all_atoms": [{"atom": k, "fisher": float(fi[k]),
                       "f_anova": float(f_scores[k]),
                       "mutual_info": float(mi_scores[k])}
                      for k in range(code_dim)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {out_path}")


def run_feature_eval(ckpt_path: Path, cells_dir: Path, split: str, top_k: int,
                     rec_mode: str, code_dim: int, dir_tag: str, device,
                     file_tag: str = None):
    """Run the full feature-evaluation pipeline for a single checkpoint.

    Args:
        dir_tag: Name of the output subdirectory (best, last, jumps).
        file_tag: Prefix for files within the subdirectory. Can be more specific,
            e.g. "jump_1000_up". Defaults to dir_tag when None.
    """
    if file_tag is None:
        file_tag = dir_tag
    print(f"\n{'='*55}")
    print(f"  Feature evaluation: {ckpt_path.name}  (split={split}  dir={dir_tag}  file={file_tag})")
    print(f"{'='*55}")

    Z, labels, hparams, epoch = encode_split(ckpt_path, cells_dir, split, rec_mode, code_dim, device)
    config = hparams.get("config", "?")

    print("  Computing feature selection scores...")
    fi                = fisher_index(Z, labels)
    f_scores, _       = f_classif(Z, labels)
    mi_scores         = mutual_info_classif(Z, labels, random_state=42)

    fisher_dir  = cfg.exp_analysis_dir(rec_mode, code_dim, "fisher")
    # Files under fisher/ carry rec_mode and code_dim in the filename: they get
    # compared across configurations, and "fisher_index_best.png" alone wouldn't
    # say which one it came from.
    tag         = f"{rec_mode}_{code_dim}_{file_tag}"
    corr_dir    = cfg.exp_analysis_dir(rec_mode, code_dim, "correlation_matrices", dir_tag)
    vis_tsne    = cfg.exp_vis_dir(rec_mode, code_dim, "tsne", dir_tag)
    vis_pca     = cfg.exp_vis_dir(rec_mode, code_dim, "pca",  dir_tag)

    plot_mean_activations(Z, labels, epoch, config,
                          fisher_dir / "feature_selection" / f"mean_activations_{tag}.png")
    plot_fisher(fi, epoch, config, top_k,
                fisher_dir / "feature_selection" / f"fisher_index_{tag}.png")
    plot_boxplots(Z, labels, fi, epoch, config, top_k,
                  fisher_dir / "boxplots" / f"boxplots_top_fisher_{tag}.png")
    plot_scatter_top2_fisher(Z, labels, fi, epoch, config,
                              fisher_dir / "scatterplots" / f"scatter_top2_fisher_{tag}.png")
    plot_pca(Z, labels, epoch, config, vis_pca / f"scatter_pca_{file_tag}.png")
    plot_tsne(Z, labels, epoch, config, vis_tsne / f"scatter_tsne_{file_tag}.png")
    plot_correlation_matrices(Z, labels, epoch, config,
                               corr_dir / f"correlation_matrices_{file_tag}.png")
    plot_feature_selection_comparison(fi, f_scores, mi_scores, epoch, config, top_k,
                                       fisher_dir / "feature_selection" / f"feature_selection_comparison_{tag}.png")
    save_feature_ranking(Z, labels, fi, f_scores, mi_scores, top_k,
                          fisher_dir / "feature_selection" / f"feature_ranking_{tag}.json")

    print(f"\n  Done. Outputs in {fisher_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Feature evaluation — discriminabilidad de átomos LISTA."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint", type=str)
    grp.add_argument("--rec_mode",   type=str, choices=["mse", "mae", "ssim"])
    p.add_argument("--code_dim",  type=int,  default=cfg.CODE_DIM)
    p.add_argument("--cells_dir", type=str,  default=None)
    p.add_argument("--split",     type=str,  default="test",
                   choices=["full", "train", "test"])
    p.add_argument("--top_k",    type=int,  default=16)
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ not found: {cells_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  split: {args.split}  |  top_k: {args.top_k}")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        rec_mode, code_dim = cfg.rec_mode_and_dim_from_ckpt(ckpt_path)
        if rec_mode is None:
            hp = torch.load(ckpt_path, map_location="cpu").get("hparams", {})
            rec_mode = hp.get("rec_mode", "mse")
            code_dim = hp.get("code_dim", cfg.CODE_DIM)
        dir_tag  = ckpt_path.parent.name   # "best", "last", "jumps"
        file_tag = ckpt_path.stem          # "best", "last", "jump_1000_up", etc.
        run_feature_eval(ckpt_path, cells_dir, args.split, args.top_k,
                         rec_mode, code_dim, dir_tag, device, file_tag=file_tag)
    else:
        rec_mode  = args.rec_mode
        code_dim  = args.code_dim
        for ckpt_path, tag in [
            (cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt", "best"),
            (cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt", "last"),
        ]:
            if ckpt_path.exists():
                run_feature_eval(ckpt_path, cells_dir, args.split, args.top_k,
                                 rec_mode, code_dim, tag, device)


if __name__ == "__main__":
    main()
