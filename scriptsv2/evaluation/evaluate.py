"""
Evaluates LISTA checkpoints saved under experiments/1_checkpoints/.

All metrics use a centered Gaussian mask (Option B). Results are saved to
experiments/2_metrics/<rec_mode>/dim_<code_dim>/, and dictionary summaries
to experiments/5_dictionaries/.

Usage:
    python evaluation/evaluate.py --rec_mode mse --code_dim 128
    python evaluation/evaluate.py --rec_mode mae --code_dim 1024 --eval_metric ssim
    python evaluation/evaluate.py --checkpoint experiments/1_checkpoints/mse/dim_128/best/best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core import configuration as cfg
from core.dataset import CellsDataset
from core.loss    import gaussian_center_mask, sparsity_metrics, _ssim_kernel, _ssim_map
from core.model   import LinearLISTAEncoder, LinearLISTADecoder

H = W = 64
C = 3


def load_model(ckpt_path: Path, device):
    ckpt        = torch.load(ckpt_path, map_location=device)
    hparams     = ckpt.get("hparams", {})
    code_dim    = hparams.get("code_dim", cfg.CODE_DIM)
    enc_state   = ckpt["encoder_state"]
    encoder     = LinearLISTAEncoder(
        cfg.IN_DIM, code_dim, hparams.get("num_iters", cfg.NUM_ITERS),
    ).to(device)
    decoder = LinearLISTADecoder(encoder).to(device)
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()
    epoch = ckpt.get("epoch", "?")
    return encoder, decoder, hparams, epoch


def compute_mse_gauss(x, x_hat):
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)) ** 2).mean(dim=[1, 2, 3]).cpu().numpy().tolist()


def compute_mae_gauss(x, x_hat):
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)).abs()).mean(dim=[1, 2, 3]).cpu().numpy().tolist()


def compute_psnr(x, x_hat, data_range):
    mse = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()
    return (10 * np.log10(data_range ** 2 / (mse + 1e-10))).tolist()


def compute_ssim_gauss_batch(x, x_hat, data_range: float) -> list:
    B    = x.size(0)
    x4d  = x.view(B, C, H, W)
    xh4d = x_hat.view(B, C, H, W)
    kernel = _ssim_kernel(cfg.SSIM_KERNEL_SIZE, cfg.SSIM_SIGMA, C, x.device)
    pad    = cfg.SSIM_KERNEL_SIZE // 2
    C1     = (cfg.SSIM_K1 * data_range) ** 2
    C2     = (cfg.SSIM_K2 * data_range) ** 2
    ssim_map = _ssim_map(x4d, xh4d, kernel, pad, C1, C2)
    w        = gaussian_center_mask(device=x.device).view(1, 1, H, W)
    return (ssim_map * w).mean(dim=[1, 2, 3]).cpu().numpy().tolist()


def stats(arr):
    a = np.array(arr)
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "min":  float(a.min()),  "max": float(a.max()),
        "p25":  float(np.percentile(a, 25)),
        "p50":  float(np.percentile(a, 50)),
        "p75":  float(np.percentile(a, 75)),
    }


def _load_indices(ckpt_path: Path, rec_mode: str, code_dim: int, split: str, N: int):
    split_path = cfg.exp_split_path(rec_mode, code_dim)
    if split == "full":
        return list(range(N))
    key = "test_indices" if split == "test" else "train_indices"
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)[key]
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    indices  = ckpt_raw.get(key, None)
    if indices is None:
        raise RuntimeError(f"No se encontraron {key} para split='{split}'.")
    return indices


@torch.no_grad()
def evaluate_one(ckpt_path: Path, cells_dir: Path, split: str,
                 eval_metric: str, rec_mode: str, code_dim: int,
                 dataset=None, indices=None, data_range=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, hparams, epoch = load_model(ckpt_path, device)
    if rec_mode is None:
        rec_mode = hparams.get("rec_mode", "mse")
    if code_dim is None:
        code_dim = hparams.get("code_dim", cfg.CODE_DIM)

    if data_range is None:
        data_range = hparams.get("ssim_data_range", None)

    if dataset is None:
        dataset = CellsDataset(cells_dir, augment=False, flatten=True)
        if data_range is None:
            data_range = dataset.data_range

    N = len(dataset)

    if indices is None:
        indices = _load_indices(ckpt_path, rec_mode, code_dim, split, N)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size  = cfg.EVAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    CODE_DIM    = encoder.code_dim
    CLASS_NAMES = ["mitotic_figure", "not_mitotic_figure"]
    atom_act_sum   = {c: np.zeros(CODE_DIM) for c in CLASS_NAMES}
    atom_act_count = {c: 0                  for c in CLASS_NAMES}
    atom_active_sum = np.zeros(CODE_DIM)   # number of samples that activate each atom

    all_mse_g  = []
    all_mae_g  = []
    all_psnr   = []
    all_ssim_g = []
    all_zeros  = []
    all_l0     = []
    all_l1n    = []
    all_labels = []

    DEAD_THR  = 1.0 / cfg.ALIVE_WINDOW_EPOCHS
    DYING_THR = 0.25

    # Cumulative trajectory, sample by sample
    traj = {
        "samples":       [],  # number of samples processed up to this point
        "n_alive":       [],  # atoms with rate > DEAD_THR
        "n_dead":        [],  # atoms with rate <= DEAD_THR
        "n_dying":       [],  # atoms with DEAD_THR < rate < DYING_THR
        "n_seen_once":   [],  # atoms activated at least once
        "active_mean":   [],  # cumulative mean L0
        "frac_zeros":    [],  # cumulative fraction of zeros
    }
    _traj_count    = np.zeros(CODE_DIM, dtype=np.int64)  # cumulative count per atom
    _traj_n        = 0                                    # samples processed
    _traj_l0_sum   = 0.0                                  # sum of L0
    _traj_zeros_sum= 0.0                                  # sum of zeros

    for xb, yb in loader:
        xb    = xb.to(device, non_blocking=True)
        z     = encoder(xb)
        x_hat = decoder(z)

        all_mse_g.extend(compute_mse_gauss(xb, x_hat))
        all_mae_g.extend(compute_mae_gauss(xb, x_hat))
        all_psnr.extend(compute_psnr(xb, x_hat, data_range))
        all_ssim_g.extend(compute_ssim_gauss_batch(xb, x_hat, data_range))

        fz, _ = sparsity_metrics(z)
        all_zeros.append(fz)
        all_l0.extend((z > 0).float().sum(dim=1).cpu().numpy().tolist())
        all_l1n.extend(z.abs().sum(dim=1).cpu().numpy().tolist())
        all_labels.extend(yb.numpy().tolist())

        z_cpu = z.detach().cpu().numpy()
        atom_active_sum += (z_cpu > 0).sum(axis=0)
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            mask = (yb.numpy() == cls_id)
            if mask.sum() > 0:
                atom_act_sum[cls_name]   += np.abs(z_cpu[mask]).sum(axis=0)
                atom_act_count[cls_name] += int(mask.sum())

        # Trajectory: process sample by sample within the batch
        active_per_sample = (z_cpu > 0)  # shape [B, code_dim]
        for i in range(z_cpu.shape[0]):
            _traj_count    += active_per_sample[i].astype(np.int64)
            _traj_n        += 1
            _traj_l0_sum   += float(active_per_sample[i].sum())
            _traj_zeros_sum += float((~active_per_sample[i]).sum())

            rate       = _traj_count / _traj_n
            n_alive_t  = int((rate > DEAD_THR).sum())
            n_dead_t   = CODE_DIM - n_alive_t
            n_dying_t  = int(((rate > DEAD_THR) & (rate < DYING_THR)).sum())
            n_seen_t   = int((_traj_count > 0).sum())

            traj["samples"].append(_traj_n)
            traj["n_alive"].append(n_alive_t)
            traj["n_dead"].append(n_dead_t)
            traj["n_dying"].append(n_dying_t)
            traj["n_seen_once"].append(n_seen_t)
            traj["active_mean"].append(_traj_l0_sum / _traj_n)
            traj["frac_zeros"].append(_traj_zeros_sum / (_traj_n * CODE_DIM))

    # Atom health: per-atom activation rate over the evaluated split
    n_samples       = len(all_mse_g)
    activation_rate = atom_active_sum / max(n_samples, 1)
    alive_mask = activation_rate > DEAD_THR
    n_alive    = int(alive_mask.sum())
    n_dead     = CODE_DIM - n_alive
    n_dying    = int(((activation_rate > DEAD_THR) & (activation_rate < DYING_THR)).sum())
    atom_health = {
        "dead_threshold":    DEAD_THR,
        "dying_threshold":   DYING_THR,
        "n_alive":           n_alive,
        "n_dead":            n_dead,
        "n_dying":           n_dying,
        "activation_rate":   activation_rate.tolist(),
        "dead_atom_indices": (~alive_mask).nonzero()[0].tolist(),
    }

    atom_trajectory = {
        "dead_threshold":  DEAD_THR,
        "dying_threshold": DYING_THR,
        "samples":         traj["samples"],
        "n_alive":         traj["n_alive"],
        "n_dead":          traj["n_dead"],
        "n_dying":         traj["n_dying"],
        "n_seen_once":     traj["n_seen_once"],
        "active_mean":     traj["active_mean"],
        "frac_zeros":      traj["frac_zeros"],
    }

    results = {
        "checkpoint":      str(ckpt_path),
        "checkpoint_name": ckpt_path.name,
        "epoch":           epoch,
        "split":           split,
        "eval_metric":     eval_metric,
        "n_samples":       n_samples,
        "config":          hparams.get("config", "?"),
        "code_dim":        CODE_DIM,
        "rec_mode_train":  hparams.get("rec_mode", rec_mode),
        "ssim_data_range": data_range,
        "eval_design":     "gaussian_mask_opcion_B",
        "mse_gaussiana":   stats(all_mse_g),
        "mae_gaussiana":   stats(all_mae_g),
        "psnr_db":         stats(all_psnr),
        "ssim_gaussiano":  {
            "sigma": cfg.SSIM_SIGMA,
            "mean":  float(np.mean(all_ssim_g)),
            "std":   float(np.std(all_ssim_g)),
        },
        "sparsity": {
            "frac_zeros": stats(all_zeros),
            "l0_activos": stats(all_l0),
            "l1_norma":   stats(all_l1n),
        },
        "atom_health":      atom_health,
        "atom_trajectory":  atom_trajectory,
    }

    labels_arr = np.array(all_labels)
    mse_arr    = np.array(all_mse_g)
    mae_arr    = np.array(all_mae_g)
    l0_arr     = np.array(all_l0)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = labels_arr == cls_id
        if mask.sum() == 0:
            continue
        results[f"clase_{cls_name}"] = {
            "n":        int(mask.sum()),
            "mse_mean": float(mse_arr[mask].mean()),
            "mse_std":  float(mse_arr[mask].std()),
            "mae_mean": float(mae_arr[mask].mean()),
            "mae_std":  float(mae_arr[mask].std()),
            "l0_mean":  float(l0_arr[mask].mean()),
            "l0_std":   float(l0_arr[mask].std()),
        }

    atom_activation = {}
    for cls_name in CLASS_NAMES:
        n = atom_act_count[cls_name]
        if n > 0:
            mean_act = atom_act_sum[cls_name] / n
            atom_activation[cls_name] = {
                str(k): float(mean_act[k]) for k in range(CODE_DIM)
            }

    if all(c in atom_activation for c in CLASS_NAMES):
        mit  = np.array([atom_activation["mitotic_figure"][str(k)]     for k in range(CODE_DIM)])
        nmit = np.array([atom_activation["not_mitotic_figure"][str(k)] for k in range(CODE_DIM)])
        diff = mit - nmit
        results["atom_analysis"] = {
            "activation_per_class":          atom_activation,
            "diff_mitotica_minus_nomitotica": {str(k): float(diff[k]) for k in range(CODE_DIM)},
            "top10_atomos_mas_mitoticos":     np.argsort(diff)[::-1][:10].tolist(),
            "top10_atomos_mas_nomitoticos":   np.argsort(diff)[:10].tolist(),
        }

    W_e    = encoder.W_e.weight.detach().cpu().numpy()
    norms  = np.linalg.norm(W_e, axis=1, keepdims=True) + 1e-8
    W_norm = W_e / norms
    gram   = np.abs(W_norm @ W_norm.T)
    np.fill_diagonal(gram, 0.0)
    n_pairs = CODE_DIM * (CODE_DIM - 1)
    upper   = gram[np.triu_indices(CODE_DIM, k=1)]
    results["dictionary_coherence"] = {
        "coherence_mean": float(gram.sum() / n_pairs),
        "coherence_max":  float(gram.max()),
        "coherence_p25":  float(np.percentile(upper, 25)),
        "coherence_p50":  float(np.percentile(upper, 50)),
        "coherence_p75":  float(np.percentile(upper, 75)),
    }

    return results, dataset, indices, data_range


def print_results(results, label=""):
    tag = f"  [{label}]" if label else " "
    em  = results.get("eval_metric", "?")
    print(f"{tag} epoch={results['epoch']}  n={results['n_samples']}  eval_metric={em}")
    print(f"    MSE gaussiana : {results['mse_gaussiana']['mean']:.6f} "
          f"+/- {results['mse_gaussiana']['std']:.6f}")
    print(f"    MAE gaussiana : {results['mae_gaussiana']['mean']:.6f} "
          f"+/- {results['mae_gaussiana']['std']:.6f}")
    print(f"    PSNR          : {results['psnr_db']['mean']:.2f} "
          f"+/- {results['psnr_db']['std']:.2f} dB")
    print(f"    SSIM gaussiano: {results['ssim_gaussiano']['mean']:.6f} "
          f"+/- {results['ssim_gaussiano']['std']:.6f}")
    print(f"    frac_zeros    : {results['sparsity']['frac_zeros']['mean']:.4f}")
    print(f"    L0 activos    : {results['sparsity']['l0_activos']['mean']:.1f} "
          f"/ {results.get('code_dim', cfg.CODE_DIM)}")
    if "atom_health" in results:
        ah = results["atom_health"]
        code_dim = results.get("code_dim", cfg.CODE_DIM)
        print(f"    Átomos vivos  : {ah['n_alive']} / {code_dim}  "
              f"(muertos={ah['n_dead']}  agonizantes={ah['n_dying']}  "
              f"thr={ah['dead_threshold']})")
    if "dictionary_coherence" in results:
        dc = results["dictionary_coherence"]
        print(f"    Coherencia dic: mean={dc['coherence_mean']:.4f}  "
              f"max={dc['coherence_max']:.4f}  p50={dc['coherence_p50']:.4f}")
    if "atom_analysis" in results:
        aa = results["atom_analysis"]
        print(f"    Top átomos mitóticos   : {aa['top10_atomos_mas_mitoticos'][:5]}")
        print(f"    Top átomos no-mitóticos: {aa['top10_atomos_mas_nomitoticos'][:5]}")


def print_compare(r_best, r_last):
    print(f"\n{'='*65}")
    print(f"  COMPARATIVA  best (epoch {r_best['epoch']}) vs last (epoch {r_last['epoch']})"
          f"  [{r_best.get('eval_metric','?')}]")
    print(f"{'='*65}")

    def delta(key_path, better="lower"):
        def get(r, kp):
            v = r
            for k in kp:
                v = v[k]
            return v
        vb   = get(r_best, key_path)
        vl   = get(r_last, key_path)
        diff = vl - vb
        if better == "lower":
            winner = "last ✓" if vl < vb else "best ✓" if vb < vl else "igual"
        else:
            winner = "last ✓" if vl > vb else "best ✓" if vb > vl else "igual"
        return vb, vl, diff, winner

    metrics = [
        ("MSE gaussiana (↓)", ["mse_gaussiana",  "mean"], "lower"),
        ("MAE gaussiana (↓)", ["mae_gaussiana",  "mean"], "lower"),
        ("PSNR dB       (↑)", ["psnr_db",        "mean"], "higher"),
        ("SSIM gauss    (↑)", ["ssim_gaussiano", "mean"], "higher"),
        ("frac_zeros    (↓)", ["sparsity", "frac_zeros", "mean"], "lower"),
        ("L0 activos    (↑)", ["sparsity", "l0_activos", "mean"], "higher"),
        ("Coherencia dic(↓)", ["dictionary_coherence", "coherence_mean"], "lower"),
    ]

    print(f"  {'Métrica':<25}  {'best':>10}  {'last':>10}  {'Δ(last-best)':>14}  {'Ganador'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*10}")
    for name, kp, better in metrics:
        try:
            vb, vl, diff, winner = delta(kp, better)
            print(f"  {name:<25}  {vb:>10.5f}  {vl:>10.5f}  {diff:>+14.5f}  {winner}")
        except Exception:
            pass
    print(f"{'='*65}")


def _save_results(results, out_dir: Path, filename: str, rec_mode: str, code_dim: int):
    """Save the results JSON and copy a summary to the dictionaries directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Guardado: {out}")

    # The summary in 5_dictionaries carries rec_mode and code_dim in the filename:
    # that folder holds all six configurations, and "eval_test_best.json" alone
    # wouldn't tell them apart.
    dicts_dir = cfg.exp_dicts_dir(rec_mode, code_dim)
    dict_out  = dicts_dir / f"{rec_mode}_{code_dim}_{filename}"
    with open(dict_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Dict summary: {dict_out}")


def _run_eval_metric(rec_mode, code_dim, ckpt_best, ckpt_last,
                     cells_dir, split, eval_metric):
    rec_dir  = cfg.exp_metrics_dir(rec_mode, code_dim, "reconstruction")
    spar_dir = cfg.exp_metrics_dir(rec_mode, code_dim, "sparsity")

    r_best = r_last = None
    dataset = indices = data_range = None

    if ckpt_best.exists():
        print(f"\n{'='*50}\n  Evaluando best.pt  [{eval_metric}] ...")
        r_best, dataset, indices, data_range = evaluate_one(
            ckpt_best, cells_dir, split, eval_metric, rec_mode, code_dim
        )
        print_results(r_best, "best.pt")
        _save_results(r_best, rec_dir / "per_class", f"eval_{split}_best.json",
                      rec_mode, code_dim)
    else:
        print("[Aviso] best.pt no encontrado.")

    if ckpt_last.exists():
        print(f"\n{'='*50}\n  Evaluando last.pt  [{eval_metric}] ...")
        r_last, dataset, indices, data_range = evaluate_one(
            ckpt_last, cells_dir, split, eval_metric, rec_mode, code_dim,
            dataset=dataset, indices=indices, data_range=data_range
        )
        print_results(r_last, "last.pt")
        _save_results(r_last, rec_dir / "per_class", f"eval_{split}_last.json",
                      rec_mode, code_dim)
    else:
        print("[Aviso] last.pt no encontrado.")

    if r_best and r_last:
        print_compare(r_best, r_last)
        compare = {
            "split":       split,
            "eval_metric": eval_metric,
            "best_epoch":  r_best["epoch"],
            "last_epoch":  r_last["epoch"],
            "best":        r_best,
            "last":        r_last,
        }
        out = rec_dir / "checkpoint_evolution" / f"compare_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(compare, f, indent=2)
        print(f"\n  Comparativa guardada en: {out}")

        # Sparsity summary — split into active_atoms/ and frac_zeros/
        active_dir = spar_dir / "active_atoms"
        frac_dir   = spar_dir / "frac_zeros"
        active_dir.mkdir(parents=True, exist_ok=True)
        frac_dir.mkdir(parents=True, exist_ok=True)
        for tag, res in [("best", r_best), ("last", r_last)]:
            spar = res["sparsity"]
            with open(active_dir / f"active_atoms_{tag}.json", "w") as f:
                json.dump({"l0_activos": spar["l0_activos"], "epoch": res["epoch"],
                           "code_dim": res["code_dim"]}, f, indent=2)
            with open(frac_dir / f"frac_zeros_{tag}.json", "w") as f:
                json.dump({"frac_zeros": spar["frac_zeros"], "epoch": res["epoch"],
                           "code_dim": res["code_dim"]}, f, indent=2)

    if r_best or r_last:
        bias_dir = rec_dir / "class_bias"
        bias_dir.mkdir(parents=True, exist_ok=True)
        for tag, res in [("best", r_best), ("last", r_last)]:
            if res is None:
                continue
            bias = {}
            for cls_name in ["mitotic_figure", "not_mitotic_figure"]:
                key = f"clase_{cls_name}"
                if key in res:
                    bias[cls_name] = res[key]
            if bias:
                with open(bias_dir / f"class_bias_{tag}.json", "w") as f:
                    json.dump(bias, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evalúa checkpoints LISTA desde experiments/1_checkpoints/."
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--checkpoint", type=str,
                     help="Ruta a un único .pt en experiments/1_checkpoints/.")
    grp.add_argument("--rec_mode",   type=str, choices=["mse", "mae", "ssim"],
                     help="Evalúa best.pt y last.pt de esta configuración.")
    p.add_argument("--code_dim",    type=int,  default=cfg.CODE_DIM)
    p.add_argument("--cells_dir",   type=str,  default=None)
    p.add_argument("--split",       type=str,  default="test",
                   choices=["full", "train", "test"])
    p.add_argument("--eval_metric", type=str,  default="all",
                   choices=cfg.EVAL_METRICS + ["all"])
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"No se encontró cells/: {cells_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}  |  split: {args.split}  |  eval_metric: {args.eval_metric}")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
        rec_mode, code_dim = cfg.rec_mode_and_dim_from_ckpt(ckpt_path)
        if rec_mode is None:
            ckpt_raw  = torch.load(ckpt_path, map_location="cpu")
            hparams   = ckpt_raw.get("hparams", {})
            rec_mode  = hparams.get("rec_mode", "mse")
            code_dim  = hparams.get("code_dim", cfg.CODE_DIM)
        metrics = cfg.EVAL_METRICS if args.eval_metric == "all" else [args.eval_metric]
        for em in metrics:
            print(f"\n{'='*50}\n  Evaluando {ckpt_path.name}  [{em}] ...")
            results, _, _, _ = evaluate_one(ckpt_path, cells_dir, args.split, em,
                                            rec_mode, code_dim)
            print_results(results, label=ckpt_path.name)
            out_dir = cfg.exp_metrics_dir(rec_mode, code_dim, "reconstruction", "per_class")
            with open(out_dir / f"eval_{args.split}_{ckpt_path.stem}.json", "w") as f:
                json.dump(results, f, indent=2)
    else:
        if args.rec_mode is None:
            raise ValueError("Se requiere --rec_mode o --checkpoint.")
        rec_mode  = args.rec_mode
        code_dim  = args.code_dim
        best_path = cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt"
        last_path = cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt"
        metrics   = cfg.EVAL_METRICS if args.eval_metric == "all" else [args.eval_metric]
        for em in metrics:
            _run_eval_metric(rec_mode, code_dim, best_path, last_path,
                             cells_dir, args.split, em)


if __name__ == "__main__":
    main()
