"""Plots of training curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg


def _plot_total_loss(hparams, history, epoch, out_path, semilog: bool):
    epochs  = range(1, len(history["loss"]) + 1)
    rec_lbl = {"ssim": "Rec (SSIM gauss)", "mae": "Rec (MAE gauss)",
               "mse":  "Rec (MSE gauss)"}.get(hparams.get("rec_mode", "mse"), "Rec loss")
    scale = "log" if semilog else "lineal"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"Métricas de entrenamiento — escala {scale}", fontsize=14)
    pf = (lambda ax, y, **kw: ax.semilogy(epochs, y, **kw)) if semilog \
         else (lambda ax, y, **kw: ax.plot(epochs, y, **kw))
    gk = dict(which="both", alpha=0.3) if semilog else dict(alpha=0.3)

    pf(axes[0,0], history["loss"], color="steelblue",   alpha=0.85, linewidth=0.8)
    axes[0,0].set_title("Total Loss"); axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(True, **gk)
    pf(axes[0,1], history["rec"],  color="forestgreen", alpha=0.85, linewidth=0.8)
    axes[0,1].set_title(rec_lbl);     axes[0,1].set_xlabel("Epoch"); axes[0,1].grid(True, **gk)
    pf(axes[0,2], history["l1"],   color="goldenrod",   alpha=0.85, linewidth=0.8)
    axes[0,2].set_title("Sparsity (L1)"); axes[0,2].set_xlabel("Epoch"); axes[0,2].grid(True, **gk)

    axes[1,0].plot(epochs, history["frac_zeros"], color="mediumpurple", linewidth=0.9)
    axes[1,0].set_title("Fracción de ceros"); axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylim(0, 1); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, history["active"], color="firebrick", linewidth=0.9)
    axes[1,1].set_title("Coefs activos"); axes[1,1].set_xlabel("Epoch")
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def _plot_rec_vs_sparsity(hparams, history, epoch, out_path, semilog: bool):
    epochs  = range(1, len(history["loss"]) + 1)
    rec_lbl = {"ssim": "Rec (SSIM gauss)", "mae": "Rec (MAE gauss)",
               "mse":  "Rec (MSE gauss)"}.get(hparams.get("rec_mode", "mse"), "Rec loss")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Rec vs Sparsity — {'log' if semilog else 'lineal'}  (epoch {epoch})", fontsize=13)
    pf = ax.semilogy if semilog else ax.plot
    pf(epochs, history["rec"], color="forestgreen", linewidth=0.9, label=rec_lbl)
    pf(epochs, history["l1"],  color="goldenrod",   linewidth=0.9, label="Sparsity (L1)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, **(dict(which="both", alpha=0.3) if semilog else dict(alpha=0.3)))
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def _plot_active_atoms(history, epoch, out_path):
    epochs = range(1, len(history["active"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Dispersión de los códigos  (epoch {epoch})", fontsize=12)
    axes[0].semilogy(epochs, history["active"], color="firebrick", linewidth=0.9)
    axes[0].set_title("Nº medio de coefs activos")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Activos")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(epochs, history["frac_zeros"], color="mediumpurple", linewidth=0.9)
    axes[1].set_title("Fracción de ceros (= 0)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Fracción")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def _plot_atom_health_history(history, epoch, out_path):
    """Evolution of alive/dead/dying atom counts during training."""
    alive = history.get("alive", [])
    dead  = history.get("dead",  [])
    dying = history.get("dying", [])
    if not alive:
        return
    epochs = range(1, len(alive) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Salud de átomos durante entrenamiento  (epoch {epoch})", fontsize=12)
    axes[0].plot(epochs, alive, color="steelblue", lw=0.9, label="alive")
    axes[0].plot(epochs, dead,  color="crimson",   lw=0.9, label="dead")
    axes[0].plot(epochs, dying, color="goldenrod", lw=0.9, label="dying")
    axes[0].set_title("Alive / Dead / Dying (absoluto)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Nº átomos")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].stackplot(epochs,
        [np.array(alive) - np.array(dying), np.array(dying), np.array(dead)],
        labels=["vivos", "agonizantes", "muertos"],
        colors=["steelblue", "goldenrod", "crimson"], alpha=0.75)
    axes[1].set_title("Proporción alive / dying / dead")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Nº átomos")
    axes[1].legend(fontsize=8, loc="upper right"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def vis_loss_curves(hparams, history, epoch, out_dir: Path):
    if not history or not history.get("loss"):
        print("  [Aviso] Historial vacío — saltando loss_curves.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_total_loss(hparams, history, epoch, out_dir / "total_loss_log.png",    semilog=True)
    _plot_total_loss(hparams, history, epoch, out_dir / "total_loss_linear.png", semilog=False)
    _plot_rec_vs_sparsity(hparams, history, epoch, out_dir / "rec_vs_sparsity_log.png", semilog=True)
    _plot_active_atoms(history, epoch, out_dir / "active_atoms_log.png")
    _plot_atom_health_history(history, epoch, out_dir / "atom_health_history.png")
