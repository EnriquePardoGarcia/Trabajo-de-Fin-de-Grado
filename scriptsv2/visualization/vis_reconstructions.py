"""Plots of reconstructions, error maps, and checkpoint evolution."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from vis_utils import to_img, load_model

H = W = cfg.PATCH_SIZE
C = 3


@torch.no_grad()
def vis_recon_single(encoder, decoder, hparams, epoch, dataset, idx,
                     split_label, out_path, device, metric_fn, metric_lbl):
    from vis_utils import atom_img
    mean = hparams.get("dataset_mean", 0.0)
    std  = hparams.get("dataset_std",  1.0)
    W_e  = encoder.W_e.weight.detach().cpu()

    x_flat, label = dataset[idx]
    x     = x_flat.unsqueeze(0).to(device)
    z     = encoder(x)
    x_hat = decoder(z)
    z_vec             = z[0].detach().cpu()
    top_vals, top_idx = torch.topk(z_vec.abs(), k=4)
    metric_val        = metric_fn(x[0].detach().cpu(), x_hat[0].detach().cpu())
    CLASS = ["mitótica", "no-mitótica"]
    cls   = CLASS[label.item()] if label.item() < 2 else "?"

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"[{metric_lbl}]  {split_label} (idx {idx}, {cls})  —  epoch {epoch}", fontsize=13)
    axes[0,0].imshow(to_img(x[0].detach().cpu(), mean, std))
    axes[0,0].set_title("Original"); axes[0,0].axis("off")
    axes[0,1].imshow(to_img(x_hat[0].detach().cpu(), mean, std))
    axes[0,1].set_title(f"Reconstrucción\n{metric_lbl}={metric_val:.4f}"); axes[0,1].axis("off")
    axes[0,2].axis("off"); axes[0,3].axis("off")
    for j in range(4):
        aidx = top_idx[j].item()
        axes[1,j].imshow(atom_img(W_e[aidx]))
        axes[1,j].set_title(f"Átomo {aidx}\n|z|={top_vals[j]:.3f}", fontsize=9)
        axes[1,j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


@torch.no_grad()
def vis_compare_recon(enc_b, dec_b, epoch_b, enc_l, dec_l, epoch_l,
                      hparams, dataset, idx, split_label, out_path, device,
                      metric_fn, metric_lbl):
    mean = hparams.get("dataset_mean", 0.0)
    std  = hparams.get("dataset_std",  1.0)
    x_flat, _ = dataset[idx]
    x = x_flat.unsqueeze(0).to(device)
    xhat_b = dec_b(enc_b(x))
    xhat_l = dec_l(enc_l(x))
    val_b = metric_fn(x[0].detach().cpu(), xhat_b[0].detach().cpu())
    val_l = metric_fn(x[0].detach().cpu(), xhat_l[0].detach().cpu())

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(f"[{metric_lbl}]  best vs last — {split_label} (idx {idx})", fontsize=13)
    axes[0].imshow(to_img(x[0].detach().cpu(), mean, std))
    axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(to_img(xhat_b[0].detach().cpu(), mean, std))
    axes[1].set_title(f"best  (epoch {epoch_b})\n{metric_lbl}={val_b:.4f}"); axes[1].axis("off")
    axes[2].imshow(to_img(xhat_l[0].detach().cpu(), mean, std))
    axes[2].set_title(f"last  (epoch {epoch_l})\n{metric_lbl}={val_l:.4f}"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


@torch.no_grad()
def vis_error_map(encoder, decoder, hparams, epoch, out_path,
                  dataset, device, idx, split_label, metric_fn, metric_lbl):
    mean  = hparams.get("dataset_mean", 0.0)
    std   = hparams.get("dataset_std",  1.0)
    chans = ["R", "G", "B"]
    x_flat, _ = dataset[idx]
    xb    = x_flat.unsqueeze(0).to(device)
    x_hat = decoder(encoder(xb))
    xi    = xb[0].view(C, H, W).detach().cpu()
    xhati = x_hat[0].view(C, H, W).detach().cpu()
    err   = (xi - xhati).abs()
    metric_val = metric_fn(xb[0].detach().cpu(), x_hat[0].detach().cpu())

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.suptitle(
        f"[{metric_lbl}]  Error — {split_label} (idx {idx})  "
        f"[{hparams.get('config','')}]  epoch {epoch}", fontsize=11)
    axes[0].imshow(to_img(xb[0], mean, std))
    axes[0].set_title("Original", fontsize=9); axes[0].axis("off")
    axes[1].imshow(to_img(x_hat[0], mean, std))
    axes[1].set_title(f"Recon\n{metric_lbl}={metric_val:.4f}", fontsize=8); axes[1].axis("off")
    for ch in range(3):
        e  = err[ch].numpy()
        im = axes[2+ch].imshow(e, cmap="hot", vmin=0, vmax=max(e.max(), 1e-6))
        axes[2+ch].set_title(f"|err| {chans[ch]}\nmax={e.max():.3f}", fontsize=8)
        axes[2+ch].axis("off")
        plt.colorbar(im, ax=axes[2+ch], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def collect_all_checkpoints_sorted(rec_mode: str, code_dim: int) -> list:
    jumps_dir = cfg.exp_checkpoints_dir(rec_mode, code_dim, "jumps")
    entries = []
    if jumps_dir.exists():
        for d in jumps_dir.iterdir():
            if d.is_dir() and d.name.startswith("jump_"):
                ckpt = d / "checkpoint.pt"
                if ckpt.exists():
                    ep = int(d.name.rsplit("_", 1)[-1])
                    entries.append((ep, d.name, ckpt))
    for label in ("best", "last"):
        p = cfg.exp_checkpoints_dir(rec_mode, code_dim, label) / f"{label}.pt"
        if p.exists():
            ck = torch.load(p, map_location="cpu")
            ep = int(ck.get("epoch", 0))
            entries.append((ep, label, p))
    entries.sort(key=lambda x: x[0])
    return entries


@torch.no_grad()
def vis_evo_all_checkpoints(rec_mode, code_dim, dataset, hparams_ref,
                             test_indices, device, metric_fn, metric_lbl,
                             out_dir: Path, sample_idx=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if sample_idx is None:
        sample_idx = random.choice(test_indices)
    mean = hparams_ref.get("dataset_mean", 0.0)
    std  = hparams_ref.get("dataset_std",  1.0)
    x_flat, _  = dataset[sample_idx]
    x_orig_img = to_img(x_flat, mean, std)

    all_ckpts  = collect_all_checkpoints_sorted(rec_mode, code_dim)
    print(f"  Evolución completa: {len(all_ckpts)} checkpoints  sample_idx={sample_idx}")
    recon_imgs = []

    for ep, lbl, ckpt_path in all_ckpts:
        encoder, decoder, hparams, _ = load_model(ckpt_path, device)
        x     = x_flat.unsqueeze(0).to(device)
        x_hat = decoder(encoder(x))
        metric_val = metric_fn(x[0].cpu(), x_hat[0].cpu())
        recon_img  = to_img(x_hat[0].cpu(), mean, std)

        fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
        fig.suptitle(f"ep {ep:05d}  [{lbl}]  {metric_lbl}={metric_val:.4f}", fontsize=9)
        axes[0].imshow(x_orig_img); axes[0].set_title("original", fontsize=8); axes[0].axis("off")
        axes[1].imshow(recon_img);  axes[1].set_title("recon",    fontsize=8); axes[1].axis("off")
        plt.tight_layout()
        out_png = out_dir / f"evo_{ep:05d}_{lbl}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close()
        print(f"    {out_png.name}")
        recon_imgs.append((ep, lbl, recon_img, metric_val))
        del encoder, decoder; torch.cuda.empty_cache()

    n = len(recon_imgs) + 1; ncols = 16
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.9))
    axes = axes.flatten()
    axes[0].imshow(x_orig_img)
    axes[0].set_title("ORIGINAL", fontsize=6, fontweight="bold"); axes[0].axis("off")
    for i, (ep, lbl, rimg, mval) in enumerate(recon_imgs, start=1):
        tag = "★" if lbl == "best" else ("▶" if lbl == "last" else "")
        axes[i].imshow(rimg)
        axes[i].set_title(f"{tag}ep {ep}\n{metric_lbl[:3]}={mval:.3f}", fontsize=5)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(
        f"Evolución completa — {rec_mode}/dim_{code_dim}  [{len(recon_imgs)} ckpts  idx={sample_idx}]",
        fontsize=11)
    plt.tight_layout()
    mosaic_path = out_dir / "mosaic_all_ckpts.png"
    plt.savefig(mosaic_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Mosaico guardado: {mosaic_path}")
