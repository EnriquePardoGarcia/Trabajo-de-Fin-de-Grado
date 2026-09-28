"""
Generates visualizations for LISTA checkpoints.

Usage:
    python visualization/visualize.py --rec_mode mse --code_dim 128
    python visualization/visualize.py --log experiments/6_logs/mae/dim_128/train_1_20915.log
    python visualization/visualize.py --eval_compare --code_dim 128
    python visualization/visualize.py --checkpoint experiments/1_checkpoints/mse/dim_128/best/best.pt

Structure generated under experiments/3_visualization/<rec_mode>/dim_<code_dim>/:
    loss_curves/        -> training curves
    reconstructions/    -> reconstructions, error maps, best/last comparisons
    atoms/top_atoms/    -> top atoms, per-class analysis, coherence
    atoms/atom_health/  -> atom health over the test set
"""

import argparse
import random
from pathlib import Path

import matplotlib
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from core.dataset import CellsDataset

from vis_utils         import load_model, load_history, get_split_indices, get_rec_metric_fn
from vis_loss_curves   import vis_loss_curves
from vis_reconstructions import (vis_recon_single, vis_compare_recon, vis_error_map,
                                  vis_evo_all_checkpoints)
from vis_atoms         import (vis_top_atoms, vis_atom_class_analysis, vis_dictionary_coherence,
                               vis_atom_health, vis_sparse_decomposition)
from vis_eval          import vis_eval_comparison
from vis_log           import run_log_mode


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def generate_all_vis(rec_mode, code_dim, device, vis_metric,
                     enc_b, dec_b, hparams_b, epoch_b,
                     enc_l, dec_l, hparams_l, epoch_l,
                     hparams_ref, dataset, train_indices, test_indices, history):

    base = lambda *s: cfg.exp_vis_dir(rec_mode, code_dim, *s)

    # 1. Training curves
    if history:
        epoch_ref = epoch_b if epoch_b is not None else epoch_l
        vis_loss_curves(hparams_ref, history, epoch_ref, base("loss_curves"))

    # Pre-select a representative test sample (min MSE among 50 candidates)
    def _best_test_idx(enc, dec):
        pool = random.sample(test_indices, min(50, len(test_indices)))
        best_idx, best_mse = pool[0], float("inf")
        for ci in pool:
            xc, _ = dataset[ci]
            with torch.no_grad():
                zc  = enc(xc.unsqueeze(0).to(device)).squeeze(0)
                xhc = dec(zc.unsqueeze(0)).squeeze(0)
            mse = float((xc.to(device) - xhc).pow(2).mean())
            if mse < best_mse:
                best_mse, best_idx = mse, ci
        return best_idx

    # 2. Reconstructions per metric
    metrics    = cfg.EVAL_METRICS if vis_metric == "all" else [vis_metric]
    data_range = hparams_ref.get("ssim_data_range", 1.0)
    per_class  = base("reconstructions", "per_class")
    err_maps   = base("reconstructions", "error_maps")
    evo_dir    = base("reconstructions", "checkpoint_evolution")

    for em in metrics:
        metric_fn, metric_lbl = get_rec_metric_fn(em, data_range)
        print(f"\n-- reconstructions [{em}] --")

        if enc_b:
            train_idx = random.choice(train_indices)
            test_idx  = _best_test_idx(enc_b, dec_b)
            vis_recon_single(enc_b, dec_b, hparams_b, epoch_b, dataset,
                             train_idx, "ENTRENAMIENTO",
                             per_class / f"recon_train_{em}_best.png", device, metric_fn, metric_lbl)
            vis_recon_single(enc_b, dec_b, hparams_b, epoch_b, dataset,
                             test_idx, "TEST",
                             per_class / f"recon_test_{em}_best.png", device, metric_fn, metric_lbl)
            vis_error_map(enc_b, dec_b, hparams_b, epoch_b,
                          err_maps / f"error_map_{em}_best.png",
                          dataset, device, test_idx, "TEST", metric_fn, metric_lbl)

        if enc_l:
            train_idx = random.choice(train_indices)
            test_idx  = random.choice(test_indices)
            vis_recon_single(enc_l, dec_l, hparams_l, epoch_l, dataset,
                             train_idx, "ENTRENAMIENTO",
                             per_class / f"recon_train_{em}_last.png", device, metric_fn, metric_lbl)
            vis_recon_single(enc_l, dec_l, hparams_l, epoch_l, dataset,
                             test_idx, "TEST",
                             per_class / f"recon_test_{em}_last.png", device, metric_fn, metric_lbl)
            vis_error_map(enc_l, dec_l, hparams_l, epoch_l,
                          err_maps / f"error_map_{em}_last.png",
                          dataset, device, test_idx, "TEST", metric_fn, metric_lbl)

        if enc_b and enc_l:
            train_idx = random.choice(train_indices)
            test_idx  = random.choice(test_indices)
            vis_compare_recon(enc_b, dec_b, epoch_b, enc_l, dec_l, epoch_l,
                              hparams_ref, dataset, train_idx, "ENTRENAMIENTO",
                              evo_dir / f"compare_train_{em}.png", device, metric_fn, metric_lbl)
            vis_compare_recon(enc_b, dec_b, epoch_b, enc_l, dec_l, epoch_l,
                              hparams_ref, dataset, test_idx, "TEST",
                              evo_dir / f"compare_test_{em}.png", device, metric_fn, metric_lbl)

    # 3. Atoms
    print("\n-- atoms/top_atoms/ --")
    atoms_dir = base("atoms", "top_atoms")
    for enc, hp, ep, label in [(enc_b, hparams_b, epoch_b, "best"),
                                (enc_l, hparams_l, epoch_l, "last")]:
        if enc is None: continue
        vis_top_atoms(enc, hp, ep, atoms_dir / f"top_atoms_{label}.png", dataset, device)
        vis_atom_class_analysis(enc, atoms_dir / f"atom_class_analysis_{label}.png",
                                dataset, device, split_indices=test_indices)
        vis_dictionary_coherence(enc, hp, ep, atoms_dir / f"dictionary_coherence_{label}.png")

    # 4. Sparse decomposition example (using the train split to avoid overfitting)
    print("\n-- atoms/decomposition/ --")
    decomp_dir = base("atoms", "decomposition")
    pool = random.sample(train_indices, min(30, len(train_indices)))
    for enc, dec, hp, ep, label in [(enc_b, dec_b, hparams_b, epoch_b, "best"),
                                    (enc_l, dec_l, hparams_l, epoch_l, "last")]:
        if enc is None:
            continue
        vis_sparse_decomposition(enc, dec, hp, ep, dataset, pool[0],
                                 decomp_dir / f"sparse_decomp_{label}.png", device,
                                 candidate_pool=pool)

    # 5. Atom health (test set, from JSON)
    print("\n-- atoms/atom_health/ --")
    vis_atom_health(rec_mode, code_dim, base("atoms", "atom_health"))

    print(f"\nVisualizaciones guardadas en: {cfg.exp_vis_dir(rec_mode, code_dim)}")


def run_exp_mode(rec_mode, code_dim, cells_dir, device, vis_metric, ckpt_type="both", mode="all"):
    ckpt_best = cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt"
    ckpt_last = cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt"
    ref_ckpt  = ckpt_best if ckpt_best.exists() else ckpt_last
    if not ref_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint para {rec_mode}/dim_{code_dim}")

    hparams_ref = torch.load(ref_ckpt, map_location="cpu").get("hparams", {})
    stats       = (hparams_ref["dataset_mean"], hparams_ref["dataset_std"])
    dataset     = CellsDataset(cells_dir, augment=False, flatten=True, stats=stats)
    train_indices, test_indices = get_split_indices(rec_mode, code_dim, ref_ckpt, len(dataset))

    enc_b = dec_b = hparams_b = epoch_b = None
    enc_l = dec_l = hparams_l = epoch_l = None
    if ckpt_type in ("best", "both") and ckpt_best.exists():
        enc_b, dec_b, hparams_b, epoch_b = load_model(ckpt_best, device)
    if ckpt_type in ("last", "both") and ckpt_last.exists():
        enc_l, dec_l, hparams_l, epoch_l = load_model(ckpt_last, device)

    hparams_ref = hparams_b if hparams_b else hparams_l
    history = load_history(rec_mode, code_dim, ref_ckpt)

    if mode in ("all", "evo_all"):
        em = vis_metric if vis_metric != "all" else "mse"
        data_range = hparams_ref.get("ssim_data_range", 1.0)
        metric_fn, metric_lbl = get_rec_metric_fn(em, data_range)
        out_dir = cfg.exp_vis_dir(rec_mode, code_dim, "reconstructions", "checkpoint_evolution")
        print("\n-- evo_all --")
        vis_evo_all_checkpoints(rec_mode, code_dim, dataset, hparams_ref,
                                test_indices, device, metric_fn, metric_lbl, out_dir)

    if mode != "evo_all":
        generate_all_vis(rec_mode, code_dim, device, vis_metric,
                         enc_b, dec_b, hparams_b, epoch_b,
                         enc_l, dec_l, hparams_l, epoch_l,
                         hparams_ref, dataset, train_indices, test_indices, history)


@torch.no_grad()
def run_checkpoint_mode(ckpt_path, cells_dir, device, mode, n_samples, vis_metric):
    encoder, decoder, hparams, epoch = load_model(ckpt_path, device)
    rec_mode, code_dim = cfg.rec_mode_and_dim_from_ckpt(ckpt_path)
    if rec_mode is None:
        rec_mode = hparams.get("rec_mode", "unknown")
        code_dim = hparams.get("code_dim", cfg.CODE_DIM)

    stats   = (hparams["dataset_mean"], hparams["dataset_std"])
    dataset = CellsDataset(cells_dir, augment=False, flatten=True, stats=stats)
    train_indices, test_indices = get_split_indices(rec_mode, code_dim, ckpt_path, len(dataset))
    history = load_history(rec_mode, code_dim, ckpt_path)

    metrics    = cfg.EVAL_METRICS if vis_metric == "all" else [vis_metric]
    data_range = hparams.get("ssim_data_range", 1.0)
    modes      = ["loss_curves", "recon", "top_atoms", "error_map"] if mode == "all" else [mode]
    base = lambda *s: cfg.exp_vis_dir(rec_mode, code_dim, *s)

    for m in modes:
        if m == "loss_curves":
            vis_loss_curves(hparams, history, epoch, base("loss_curves"))
        elif m == "recon":
            for em in metrics:
                metric_fn, metric_lbl = get_rec_metric_fn(em, data_range)
                for split_label, pool in [("ENTRENAMIENTO", train_indices), ("TEST", test_indices)]:
                    tag = "train" if "ENTRENAMIENTO" in split_label else "test"
                    for idx in random.sample(pool, min(n_samples // 2, len(pool))):
                        vis_recon_single(encoder, decoder, hparams, epoch, dataset, idx, split_label,
                                         base("reconstructions", "per_class") / f"recon_{tag}_{em}_{ckpt_path.stem}_idx{idx}.png",
                                         device, metric_fn, metric_lbl)
        elif m == "top_atoms":
            atoms_dir = base("atoms", "top_atoms")
            vis_top_atoms(encoder, hparams, epoch, atoms_dir / f"top_atoms_{ckpt_path.stem}.png", dataset, device)
            vis_atom_class_analysis(encoder, atoms_dir / f"atom_class_analysis_{ckpt_path.stem}.png",
                                    dataset, device, split_indices=test_indices)
            vis_dictionary_coherence(encoder, hparams, epoch,
                                     atoms_dir / f"dictionary_coherence_{ckpt_path.stem}.png")
            pool = random.sample(test_indices, min(20, len(test_indices)))
            vis_sparse_decomposition(encoder, decoder, hparams, epoch, dataset, pool[0],
                                     base("atoms", "decomposition") / f"sparse_decomp_{ckpt_path.stem}.png",
                                     device, candidate_pool=pool)
        elif m == "error_map":
            for em in metrics:
                metric_fn, metric_lbl = get_rec_metric_fn(em, data_range)
                idx = random.choice(test_indices)
                vis_error_map(encoder, decoder, hparams, epoch,
                              base("reconstructions", "error_maps") / f"error_map_{em}_{ckpt_path.stem}.png",
                              dataset, device, idx, "TEST", metric_fn, metric_lbl)

    print(f"\nVisualizaciones guardadas en: {cfg.exp_vis_dir(rec_mode, code_dim)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Genera visualizaciones de checkpoints LISTA.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--checkpoint",    type=str, default=None)
    g.add_argument("--rec_mode",      type=str, default=None, choices=["mse", "mae", "ssim"])
    g.add_argument("--log",           type=str, default=None)
    p.add_argument("--watch",         type=int, default=0)
    p.add_argument("--code_dim",      type=int, default=None)
    p.add_argument("--ckpt_type",     type=str, default="both", choices=["best", "last", "both"])
    p.add_argument("--mode",          type=str, default="all",
                   choices=["all", "loss_curves", "recon", "top_atoms", "error_map", "evo_all"])
    p.add_argument("--eval_compare",  action="store_true",
                   help="Comparativa de evaluación entre modos (requiere --code_dim).")
    p.add_argument("--cells_dir",     type=str, default=None)
    p.add_argument("--n_samples",     type=int, default=4)
    p.add_argument("--vis_metric",    type=str, default="all", choices=cfg.EVAL_METRICS + ["all"])
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  vis_metric: {args.vis_metric}")

    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ no encontrado: {cells_dir}")

    if args.eval_compare:
        matplotlib.use("Agg")
        if not args.code_dim:
            raise ValueError("--code_dim es requerido con --eval_compare.")
        print(f"Generando comparativa — dim={args.code_dim}...")
        vis_eval_comparison(args.code_dim, cfg.exp_comparativa_dir("7_eval_plots"))
        print("Listo.")
        return

    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            raise FileNotFoundError(f"Log no encontrado: {log_path}")
        run_log_mode(log_path, None, args.watch)
    elif args.checkpoint:
        matplotlib.use("Agg")
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
        run_checkpoint_mode(ckpt_path, cells_dir, device, args.mode, args.n_samples, args.vis_metric)
    elif args.rec_mode:
        matplotlib.use("Agg")
        if not args.code_dim:
            raise ValueError("--code_dim es requerido con --rec_mode.")
        run_exp_mode(args.rec_mode, args.code_dim, cells_dir, device,
                     args.vis_metric, args.ckpt_type, args.mode)
    else:
        raise ValueError("Especifica --log, --rec_mode --code_dim, --checkpoint o --eval_compare.")


if __name__ == "__main__":
    main()
