"""Atom plots: top atoms, per-class analysis, coherence, and health."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from vis_utils import atom_img, to_img, load_eval_json

H = W = cfg.PATCH_SIZE
C = 3


@torch.no_grad()
def vis_atom_class_analysis(encoder, out_path, dataset, device, split_indices=None):
    """Plot mean per-atom activation for each class and their difference."""
    indices = split_indices if split_indices is not None else list(range(len(dataset)))
    loader  = DataLoader(Subset(dataset, indices), batch_size=256, shuffle=False, num_workers=0)
    CODE_DIM    = encoder.code_dim
    CLASS_NAMES = ["mitotic_figure", "not_mitotic_figure"]
    act_sum   = {c: np.zeros(CODE_DIM) for c in CLASS_NAMES}
    act_count = {c: 0 for c in CLASS_NAMES}
    for xb, yb in loader:
        z = encoder(xb.to(device)).detach().cpu().numpy()
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = (yb.numpy() == cls_id)
            if mask.sum() > 0:
                act_sum[cls_name]   += np.abs(z[mask]).sum(axis=0)
                act_count[cls_name] += int(mask.sum())
    mean_mit  = act_sum["mitotic_figure"]     / max(act_count["mitotic_figure"],  1)
    mean_nmit = act_sum["not_mitotic_figure"] / max(act_count["not_mitotic_figure"], 1)
    diff  = mean_mit - mean_nmit
    atoms = np.arange(CODE_DIM)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, CODE_DIM * 0.3), 8))
    fig.suptitle(f"Activación por átomo y clase  [{len(indices)} muestras]", fontsize=13)
    w = 0.4
    axes[0].bar(atoms - w/2, mean_mit,  width=w, color="crimson",   alpha=0.75, label="Mitótica")
    axes[0].bar(atoms + w/2, mean_nmit, width=w, color="steelblue", alpha=0.75, label="No-mitótica")
    axes[0].set_title("Activación media |z_k| por clase"); axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_xticks(atoms[::max(1, CODE_DIM//16)])
    colors = ["crimson" if d > 0 else "steelblue" for d in diff]
    axes[1].bar(atoms, diff, color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Diferencia mitótica − no-mitótica")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(atoms[::max(1, CODE_DIM//16)])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def vis_dictionary_coherence(encoder, hparams, epoch, out_path):
    """Plot the dictionary's Gram matrix and its off-diagonal coherence distribution."""
    W_e    = encoder.W_e.weight.detach().cpu().numpy()
    norms  = np.linalg.norm(W_e, axis=1, keepdims=True) + 1e-8
    W_norm = W_e / norms
    gram   = np.abs(W_norm @ W_norm.T)
    CODE_DIM = W_e.shape[0]
    upper    = gram[np.triu_indices(CODE_DIM, k=1)]
    np.fill_diagonal(gram, 0.0)
    coh_mean = float(upper.mean()); coh_max = float(upper.max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Coherencia del diccionario  [{hparams.get('config','')}]  epoch {epoch}\n"
        f"media={coh_mean:.4f}  max={coh_max:.4f}", fontsize=12)
    im = axes[0].imshow(gram, cmap="hot", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("|<a_i, a_j>|"); axes[0].set_xlabel("Átomo j"); axes[0].set_ylabel("Átomo i")
    plt.colorbar(im, ax=axes[0])
    axes[1].hist(upper, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[1].axvline(coh_mean, color="crimson", linestyle="--", label=f"media={coh_mean:.3f}")
    axes[1].set_title("Distribución de correlaciones"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


def vis_top_atoms(encoder, hparams, epoch, out_path, dataset, device, n_eval=512):
    """Plot the dictionary atoms with the highest mean activation."""
    indices = random.sample(range(len(dataset)), min(n_eval, len(dataset)))
    loader  = DataLoader(Subset(dataset, indices),
                         batch_size=min(256, len(indices)), shuffle=False, num_workers=0)
    acc_z = torch.zeros(encoder.code_dim); total = 0
    for xb, _ in loader:
        z = encoder(xb.to(device)).detach().cpu()
        acc_z += z.abs().sum(dim=0); total += z.size(0)
    mean_act              = acc_z / total
    top32_vals, top32_idx = torch.topk(mean_act, k=min(32, encoder.code_dim))
    W_e = encoder.W_e.weight.detach().cpu()
    fig, axes = plt.subplots(4, 8, figsize=(8 * 1.8, 4 * 1.8))
    fig.suptitle(f"Top-32 átomos  [{hparams.get('config','')}]  epoch {epoch}", fontsize=12)
    for j, ax in enumerate(axes.flatten()):
        if j < len(top32_idx):
            idx = top32_idx[j].item()
            ax.imshow(atom_img(W_e[idx]))
            ax.set_title(f"#{idx}\nact={top32_vals[j]:.3f}", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  Guardado: {out_path}")


@torch.no_grad()
def vis_sparse_decomposition(encoder, decoder, hparams, epoch, dataset, sample_idx, out_path, device,
                              n_atoms=None, candidate_pool=None):
    """Illustrate the sparse decomposition of a patch: x ≈ Σ z_k · d_k.

    Top row: [original x] | [sparse code z (bars)] | [reconstruction x̂]
    Bottom row: top-N active atoms with their coefficients → = x̂

    If candidate_pool has more than one element, all are evaluated and the
    one with the lowest reconstruction MSE is chosen (a representative
    example, not an outlier).
    """
    n_atoms = n_atoms if n_atoms is not None else cfg.VIS_N_TOP_ATOMS
    mean    = hparams.get("dataset_mean", 0.0)
    std     = hparams.get("dataset_std",  1.0)

    # Best-candidate selection: lowest MSE among those with >=2 active atoms;
    # if none meet that criterion, simply the one with the lowest MSE.
    if candidate_pool and len(candidate_pool) > 1:
        scored = []
        for ci in candidate_pool:
            xc, _ = dataset[ci]
            zc    = encoder(xc.unsqueeze(0).to(device)).squeeze(0)
            xhc   = decoder(zc.unsqueeze(0)).squeeze(0)
            nnz   = int((zc > 0).sum())
            mse   = float((xc.to(device) - xhc).pow(2).mean())
            scored.append((mse, nnz, ci))
        rich   = [(m, n, i) for m, n, i in scored if n >= 2]
        pool_s = sorted(rich if rich else scored, key=lambda t: t[0])
        sample_idx = pool_s[0][2]

    x_flat, label = dataset[sample_idx]
    x_flat = x_flat.to(device)
    z      = encoder(x_flat.unsqueeze(0)).squeeze(0)   # (CODE_DIM,)
    x_hat  = decoder(z.unsqueeze(0)).squeeze(0)         # (IN_DIM,)
    z_np   = z.detach().cpu().numpy()

    active_idx = np.where(z_np > 0)[0]
    active_idx = active_idx[np.argsort(np.abs(z_np[active_idx]))[::-1]]  # descending order
    n_show     = min(n_atoms, len(active_idx))

    W_e = encoder.W_e.weight.detach().cpu()

    # ── Layout ──
    fig = plt.figure(figsize=(max(14, (n_show + 1) * 2.4), 8))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.50)

    # Row 0: original | code | reconstruction
    gs_top  = gs[0].subgridspec(1, 3, wspace=0.35)
    ax_orig  = fig.add_subplot(gs_top[0, 0])
    ax_code  = fig.add_subplot(gs_top[0, 1])
    ax_recon = fig.add_subplot(gs_top[0, 2])

    ax_orig.imshow(to_img(x_flat.cpu(), mean, std))
    cls = "Mitótica" if int(label) == 0 else "No mitótica"
    ax_orig.set_title(f"Original x\n({cls})", fontsize=10)
    ax_orig.axis("off")

    bar_colors = np.array(["#cccccc"] * len(z_np))
    bar_colors[active_idx[:n_show]] = "#e05c2a"   # top-N active: orange
    if len(active_idx) > n_show:
        bar_colors[active_idx[n_show:]] = "#6fa8dc"  # remaining active: blue
    ax_code.bar(np.arange(len(z_np)), z_np, color=bar_colors, alpha=0.85, width=1.0)
    ax_code.axhline(0, color="black", lw=0.5)
    sparsity_pct = 100.0 * len(active_idx) / max(len(z_np), 1)
    ax_code.set_title(
        f"Código disperso z\n"
        f"{len(active_idx)} activos / {len(z_np)} total  ({sparsity_pct:.1f}% no nulo)",
        fontsize=10)
    ax_code.set_xlabel("índice de átomo k")
    ax_code.set_ylabel("z_k")
    ax_code.grid(True, axis="y", alpha=0.3)

    ax_recon.imshow(to_img(x_hat.cpu(), mean, std))
    mse = float((x_flat.cpu() - x_hat.cpu()).pow(2).mean())
    ax_recon.set_title(f"Reconstrucción x̂\nMSE = {mse:.5f}", fontsize=10)
    ax_recon.axis("off")

    # Row 1: top-N atoms + "= x̂" panel
    n_cols  = n_show + 1
    gs_bot  = gs[1].subgridspec(1, n_cols, wspace=0.20)
    for j, atom_i in enumerate(active_idx[:n_show]):
        ax = fig.add_subplot(gs_bot[0, j])
        ax.imshow(atom_img(W_e[atom_i]))
        coef   = z_np[atom_i]
        prefix = "+" if coef >= 0 else ""
        ax.set_title(f"d_{atom_i}\n×{prefix}{coef:.3f}", fontsize=8)
        ax.axis("off")

    ax_eq = fig.add_subplot(gs_bot[0, n_show])
    ax_eq.imshow(to_img(x_hat.cpu(), mean, std))
    ax_eq.set_title("= x̂", fontsize=9)
    ax_eq.axis("off")

    fig.suptitle(
        f"Descomposición dispersa LISTA — {hparams.get('config','?')}  epoch {epoch}\n"
        f"x̂ = Σ z_k · d_k  (naranjas: top-{n_show} átomos más activos)",
        fontsize=12
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out_path}")


def vis_atom_health(rec_mode: str, code_dim: int, out_dir: Path):
    """Three atom-health plots computed on the test set."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sorted activation rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Salud de átomos (test) — {rec_mode}/dim_{code_dim}", fontsize=13)
    for col, ckpt in enumerate(["best", "last"]):
        d = load_eval_json(rec_mode, code_dim, ckpt)
        ax = axes[col]
        if d is None:
            ax.set_title(f"{ckpt} — sin datos"); ax.axis("off"); continue
        ah   = d.get("atom_health", {})
        rate = np.array(ah.get("activation_rate", []))
        if len(rate) == 0:
            ax.set_title(f"{ckpt} — sin activation_rate"); continue
        dead_thr  = ah.get("dead_threshold",  0.1)
        dying_thr = ah.get("dying_threshold", 0.25)

        def color(r):
            if r <= dead_thr:  return "crimson"
            if r <  dying_thr: return "goldenrod"
            return "steelblue"

        sorted_rate = np.sort(rate)[::-1]
        ax.bar(np.arange(len(sorted_rate)), sorted_rate,
               color=[color(r) for r in sorted_rate], width=1.0, alpha=0.85)
        ax.axhline(dead_thr,  color="crimson",   linestyle="--", lw=1, label=f"dead thr={dead_thr}")
        ax.axhline(dying_thr, color="goldenrod", linestyle="--", lw=1, label=f"dying thr={dying_thr}")
        ax.set_title(
            f"{ckpt} (ep.{d['epoch']})  "
            f"vivos={ah.get('n_alive',0)}  agonizantes={ah.get('n_dying',0)}  muertos={ah.get('n_dead',0)}",
            fontsize=9)
        ax.set_xlabel("átomo (ordenado por activación)"); ax.set_ylabel("fracción muestras test")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "atom_health.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out}")

    # 2. Stacked bars: alive / dying / dead
    ckpt_labels, n_alive_v, n_dying_v, n_dead_v, code_dim_v = [], [], [], [], []
    for ckpt in ["best", "last"]:
        d = load_eval_json(rec_mode, code_dim, ckpt)
        if d is None: continue
        ah = d.get("atom_health", {})
        ckpt_labels.append(f"{ckpt}\n(ep.{d['epoch']})")
        n_alive_v.append(ah.get("n_alive", 0))
        n_dying_v.append(ah.get("n_dying", 0))
        n_dead_v.append(ah.get("n_dead",  0))
        code_dim_v.append(d.get("code_dim", code_dim))
    if ckpt_labels:
        x = np.arange(len(ckpt_labels))
        fig2, ax = plt.subplots(figsize=(6, 5))
        fig2.suptitle(f"Salud de átomos — test set\n{rec_mode}/dim_{code_dim}", fontsize=12)
        alive_pure = [a - dy for a, dy in zip(n_alive_v, n_dying_v)]
        ax.bar(x, alive_pure, color="steelblue", label="vivos")
        ax.bar(x, n_dying_v,  bottom=alive_pure, color="goldenrod", label="agonizantes")
        ax.bar(x, n_dead_v,   bottom=[a + dy for a, dy in zip(alive_pure, n_dying_v)],
               color="crimson", label="muertos")
        ax.axhline(code_dim_v[0], color="black", linestyle="--", lw=0.8, label=f"total={code_dim_v[0]}")
        ax.set_xticks(x); ax.set_xticklabels(ckpt_labels)
        ax.set_ylabel("Nº átomos"); ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
        for i, (a, dy, de) in enumerate(zip(n_alive_v, n_dying_v, n_dead_v)):
            ax.text(i, code_dim_v[0] + 1, f"V={a} A={dy} M={de}", ha="center", fontsize=8)
        plt.tight_layout()
        out2 = out_dir / "atom_health_summary.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Guardado: {out2}")

    # 3. Activation by atom index (natural order)
    fig3, axes3 = plt.subplots(len(ckpt_labels) or 1, 1,
                                figsize=(max(14, code_dim * 0.12), 4 * max(len(ckpt_labels), 1)),
                                squeeze=False)
    fig3.suptitle(f"Activación por átomo (test) — {rec_mode}/dim_{code_dim}", fontsize=12)
    for row, ckpt in enumerate(["best", "last"]):
        d = load_eval_json(rec_mode, code_dim, ckpt)
        ax = axes3[row, 0]
        if d is None: ax.set_visible(False); continue
        ah   = d.get("atom_health", {})
        rate = np.array(ah.get("activation_rate", []))
        if len(rate) == 0: ax.set_visible(False); continue
        dead_thr  = ah.get("dead_threshold",  0.1)
        dying_thr = ah.get("dying_threshold", 0.25)
        colors = ["crimson" if r <= dead_thr else "goldenrod" if r < dying_thr else "steelblue"
                  for r in rate]
        ax.bar(np.arange(len(rate)), rate, color=colors, width=1.0, alpha=0.85)
        ax.axhline(dead_thr,  color="crimson",   linestyle="--", lw=1, label=f"dead thr={dead_thr}")
        ax.axhline(dying_thr, color="goldenrod", linestyle="--", lw=1, label=f"dying thr={dying_thr}")
        ax.set_title(
            f"{ckpt} (ep.{d['epoch']})  "
            f"vivos={ah.get('n_alive',0)}  agonizantes={ah.get('n_dying',0)}  muertos={ah.get('n_dead',0)}",
            fontsize=10)
        ax.set_xlabel("índice de átomo"); ax.set_ylabel("tasa activación")
        ax.set_xlim(-0.5, len(rate) - 0.5)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = out_dir / "atom_activation_by_index.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out3}")

    # 4. Cumulative trajectory over the test set
    vis_atom_trajectory(rec_mode, code_dim, out_dir)


def vis_atom_trajectory(rec_mode: str, code_dim: int, out_dir: Path):
    """Evolution of alive/dead/dying/seen_once/active_mean/frac_zeros over the test set."""
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for ckpt in ["best", "last"]:
        d = load_eval_json(rec_mode, code_dim, ckpt)
        if d and "atom_trajectory" in d:
            datasets[ckpt] = (d["epoch"], d["atom_trajectory"])

    if not datasets:
        print("  [Aviso] Sin atom_trajectory en los JSONs — saltando.")
        return

    COLORS = {"best": "steelblue", "last": "crimson"}
    n_ckpts = len(datasets)

    # ── Plot 1: alive / dead / dying / seen_once ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Trayectoria de átomos durante test — {rec_mode}/dim_{code_dim}", fontsize=13)

    panels = [
        (axes[0,0], "n_alive",     "Átomos vivos",          "steelblue"),
        (axes[0,1], "n_dead",      "Átomos muertos",         "crimson"),
        (axes[1,0], "n_dying",     "Átomos agonizantes",     "goldenrod"),
        (axes[1,1], "n_seen_once", "Vistos ≥1 vez (acumulado)", "mediumseagreen"),
    ]
    for ax, key, title, _ in panels:
        for ckpt, (epoch, traj) in datasets.items():
            samples = traj["samples"]
            values  = traj[key]
            ax.plot(samples, values, lw=0.9, color=COLORS[ckpt],
                    label=f"{ckpt} (ep.{epoch})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("muestras test procesadas"); ax.set_ylabel("Nº átomos")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)

    plt.tight_layout()
    out4 = out_dir / "atom_trajectory_health.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out4}")

    # ── Plot 2: alive/dead/dying stacked (area) ──
    fig2, axes2 = plt.subplots(1, n_ckpts, figsize=(8 * n_ckpts, 5), squeeze=False)
    fig2.suptitle(f"Composición de salud de átomos a lo largo del test — {rec_mode}/dim_{code_dim}", fontsize=12)
    for col, (ckpt, (epoch, traj)) in enumerate(datasets.items()):
        ax   = axes2[0, col]
        s    = np.array(traj["samples"])
        alive_pure = np.array(traj["n_alive"]) - np.array(traj["n_dying"])
        dying      = np.array(traj["n_dying"])
        dead       = np.array(traj["n_dead"])
        ax.stackplot(s, alive_pure, dying, dead,
                     labels=["vivos", "agonizantes", "muertos"],
                     colors=["steelblue", "goldenrod", "crimson"], alpha=0.8)
        ax.set_title(f"{ckpt} (ep.{epoch})", fontsize=10)
        ax.set_xlabel("muestras test procesadas"); ax.set_ylabel("Nº átomos")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, s[-1])
    plt.tight_layout()
    out5 = out_dir / "atom_trajectory_stacked.png"
    plt.savefig(out5, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out5}")

    # ── Plot 3: active_mean and frac_zeros ──
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle(f"Sparsidad acumulada durante test — {rec_mode}/dim_{code_dim}", fontsize=12)
    for ckpt, (epoch, traj) in datasets.items():
        s = traj["samples"]
        axes3[0].plot(s, traj["active_mean"], lw=0.9, color=COLORS[ckpt], label=f"{ckpt} (ep.{epoch})")
        axes3[1].plot(s, traj["frac_zeros"],  lw=0.9, color=COLORS[ckpt], label=f"{ckpt} (ep.{epoch})")
    axes3[0].set_title("L0 medio acumulado"); axes3[0].set_xlabel("muestras"); axes3[0].set_ylabel("átomos activos / muestra")
    axes3[1].set_title("Fracción de ceros acumulada"); axes3[1].set_xlabel("muestras"); axes3[1].set_ylabel("fracción")
    for ax in axes3: ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, None)
    plt.tight_layout()
    out6 = out_dir / "atom_trajectory_sparsity.png"
    plt.savefig(out6, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Guardado: {out6}")
