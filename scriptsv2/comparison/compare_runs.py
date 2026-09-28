"""
Generates a cross-model comparison table between different LISTA runs.

ALWAYS evaluates both best.pt and last.pt for each configuration and
produces TWO separate tables — one for best and one for last.

Usage:
    python comparison/compare_runs.py --configs mse:128 mae:128 ssim:128
    python comparison/compare_runs.py --modes mse mae ssim --code_dim 128
    python comparison/compare_runs.py --configs mse:64 mse:128 mse:256
"""

import argparse
import csv
import json
import sys
from datetime import datetime
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
from core.loss    import gaussian_center_mask, _ssim_kernel, _ssim_map
from core.model   import LinearLISTAEncoder, LinearLISTADecoder

H = W = cfg.PATCH_SIZE
C = 3


def load_model(ckpt_path: Path, device):
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
    decoder.eval()
    epoch = ckpt.get("epoch", "?")
    return encoder, decoder, hparams, epoch


def load_test_indices(rec_mode: str, code_dim: int, ckpt_path: Path) -> list:
    split_path = cfg.exp_split_path(rec_mode, code_dim)
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)["test_indices"]
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    indices  = ckpt_raw.get("test_indices", None)
    if indices is None:
        raise RuntimeError(f"No se encontraron test_indices para {rec_mode}/dim_{code_dim}")
    return indices


def compute_mse_gauss(x, x_hat):
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)) ** 2).mean().item()


def compute_mae_gauss(x, x_hat):
    B = x.size(0)
    w = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    return (w * (x_hat.view(B, C, H, W) - x.view(B, C, H, W)).abs()).mean().item()


def compute_psnr(x, x_hat, data_range):
    mse = ((x - x_hat) ** 2).mean().item()
    return 10 * np.log10(data_range ** 2 / (mse + 1e-10))


def compute_ssim_gauss_batch(x, x_hat, data_range: float) -> float:
    B    = x.size(0)
    x4d  = x.view(B, C, H, W)
    xh4d = x_hat.view(B, C, H, W)
    kernel = _ssim_kernel(cfg.SSIM_KERNEL_SIZE, cfg.SSIM_SIGMA, C, x.device)
    pad    = cfg.SSIM_KERNEL_SIZE // 2
    C1     = (cfg.SSIM_K1 * data_range) ** 2
    C2     = (cfg.SSIM_K2 * data_range) ** 2
    ssim_map = _ssim_map(x4d, xh4d, kernel, pad, C1, C2)
    w        = gaussian_center_mask(device=x.device).view(1, 1, H, W)
    return (ssim_map * w).mean().item()


def compute_shared_indices(configs: list, ckpt_paths: list) -> list:
    """Compute the intersection of test indices shared by all configs.

    Args:
        configs: List of (rec_mode, code_dim) tuples.
        ckpt_paths: Checkpoint path for each config, in the same order.

    Returns:
        Sorted list of indices present in every config's test set.
    """
    sets = []
    for (rec_mode, code_dim), ckpt_path in zip(configs, ckpt_paths):
        indices = load_test_indices(rec_mode, code_dim, ckpt_path)
        sets.append(set(indices))
    shared = sets[0]
    for s in sets[1:]:
        shared = shared & s
    return sorted(list(shared))


@torch.no_grad()
def evaluate_on_indices(ckpt_path: Path, dataset: CellsDataset,
                        shared_indices: list, data_range: float,
                        device) -> dict:
    encoder, decoder, hparams, epoch = load_model(ckpt_path, device)

    loader = DataLoader(
        Subset(dataset, shared_indices),
        batch_size  = cfg.EVAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    all_mse_g  = []
    all_mae_g  = []
    all_psnr   = []
    all_ssim_g = []

    for xb, _ in loader:
        xb    = xb.to(device, non_blocking=True)
        z     = encoder(xb)
        x_hat = decoder(z)

        all_mse_g.append(compute_mse_gauss(xb, x_hat))
        all_mae_g.append(compute_mae_gauss(xb, x_hat))
        all_psnr.append(compute_psnr(xb, x_hat, data_range))
        all_ssim_g.append(compute_ssim_gauss_batch(xb, x_hat, data_range))

    return {
        "epoch":      epoch,
        "rec_mode":   hparams.get("rec_mode", "?"),
        "config":     hparams.get("config", "?"),
        "lmbd_l1":    hparams.get("lmbd_l1", "?"),
        "lmbd_rec":   hparams.get("lmbd_rec", 1.0),
        "n_samples":  len(shared_indices),
        "mae_gauss":  float(np.mean(all_mae_g)),
        "mse_gauss":  float(np.mean(all_mse_g)),
        "psnr_db":    float(np.mean(all_psnr)),
        "ssim_gauss": float(np.mean(all_ssim_g)),
    }


def print_table(rows: list, shared_n: int, label: str):
    col_w  = 13
    header = (f"{'Modelo':<32}  {'Modo':<6}  {'Epoch':>6}  "
              f"{'MAE_gauss':>{col_w}}  {'MSE_gauss':>{col_w}}  "
              f"{'PSNR_dB':>{col_w}}  {'SSIM_gauss':>{col_w}}")
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(f"  TABLA {label}  —  n_shared={shared_n}  (Opción B: máscara gaussiana)")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['run_name']:<32}  "
            f"{r['rec_mode']:<6}  "
            f"{str(r['epoch']):>6}  "
            f"{r['mae_gauss']:>{col_w}.6f}  "
            f"{r['mse_gauss']:>{col_w}.6f}  "
            f"{r['psnr_db']:>{col_w}.3f}  "
            f"{r['ssim_gauss']:>{col_w}.6f}"
        )
    print(sep)


def save_csv(rows: list, csv_path: Path, shared_n: int,
             shared_json_path: Path, label: str):
    fieldnames = [
        "run_name", "run_dir", "config", "rec_mode", "epoch",
        "lmbd_l1", "lmbd_rec", "n_samples",
        "mae_gauss", "mse_gauss", "psnr_db", "ssim_gauss",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.write("\n")
        f.write(f"# Tabla: {label}\n")
        f.write(f"# Diseño evaluacion: Opcion B — mascara gaussiana en todas las metricas\n")
        f.write(f"# SSIM gaussiano: implementacion manual con mascara centrada (no ignite)\n")
        f.write(f"# Evaluacion sobre interseccion de test sets: {shared_n} muestras\n")
        f.write(f"# shared_test_json: {shared_json_path}\n")
        f.write(f"# Runs: {', '.join(r['run_name'] for r in rows)}\n")
        f.write(f"# NOTA: cada modelo fue entrenado con su propio split 80/20 aleatorio.\n")
        f.write(f"#       Se usa la interseccion de los test sets para comparacion justa.\n")
    print(f"  CSV guardado en: {csv_path}")


METRIC_COLS = [
    ("MAE gauss (↓)", "mae_gauss",  "lower",  ".6f"),
    ("MSE gauss (↓)", "mse_gauss",  "lower",  ".6f"),
    ("PSNR dB   (↑)", "psnr_db",   "higher", ".3f"),
    ("SSIM gauss(↑)", "ssim_gauss", "higher", ".6f"),
]


def _best_row_idx(rows: list, key: str, better: str) -> int:
    vals = [r[key] for r in rows]
    return vals.index(min(vals) if better == "lower" else max(vals))


def save_comparison_table_png(rows_best: list, rows_last: list,
                               shared_n: int, out_path: Path):
    col_labels = ["Modelo (run)", "Modo", "Epoch"] + [m[0] for m in METRIC_COLS]

    def build_cell_data(rows):
        data, best_col = [], {}
        for _, key, better, _ in METRIC_COLS:
            best_col[key] = _best_row_idx(rows, key, better)
        for r in rows:
            row_cells = [
                r["run_name"],
                r.get("rec_mode", "?"),
                str(r["epoch"]),
            ]
            for _, key, _, fmt in METRIC_COLS:
                row_cells.append(f"{r[key]:{fmt}}")
            data.append(row_cells)
        return data, best_col

    data_best, best_col_best = build_cell_data(rows_best)
    data_last, best_col_last = build_cell_data(rows_last)

    n_rows = len(rows_best)
    n_cols = len(col_labels)
    HDR, GREEN, ROW_A, ROW_B = "#2c3e50", "#d5f5e3", "#ffffff", "#f0f0f0"

    fig, axes = plt.subplots(2, 1, figsize=(16, max(4, n_rows * 0.65 + 2.5) * 2))

    for ax, data, best_col, label in (
        (axes[0], data_best, best_col_best, "BEST.PT"),
        (axes[1], data_last, best_col_last, "LAST.PT"),
    ):
        ax.axis("off")
        ax.set_title(label, fontsize=12, fontweight="bold",
                     color="white", backgroundcolor=HDR, pad=6)

        table = ax.table(cellText=data, colLabels=col_labels,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(list(range(n_cols)))

        for j in range(n_cols):
            table[0, j].set_facecolor(HDR)
            table[0, j].set_text_props(color="white", fontweight="bold")

        for i in range(n_rows):
            bg = ROW_A if i % 2 else ROW_B
            for j in range(n_cols):
                table[i + 1, j].set_facecolor(bg)

        for col_idx, (_, key, _, _) in enumerate(METRIC_COLS, start=3):
            best_i = best_col[key]
            table[best_i + 1, col_idx].set_facecolor(GREEN)

    fig.suptitle(
        f"Comparativa cross-modelo  |  n_shared = {shared_n}  "
        f"|  Opción B: máscara gaussiana",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tabla PNG guardada en: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Tabla comparativa cross-modelo LISTA. "
            "Genera DOS tablas: una para best.pt y otra para last.pt."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--configs", nargs="+", metavar="REC_MODE:CODE_DIM",
                   help="Lista de configuraciones a comparar, p.ej. mse:128 mae:128 ssim:128")
    g.add_argument("--modes",   nargs="+", metavar="REC_MODE",
                   choices=["mse", "mae", "ssim"],
                   help="Lista de modos (usar con --code_dim).")
    p.add_argument("--code_dim",  type=int, default=None,
                   help="Dimensión compartida (usar con --modes).")
    p.add_argument("--split",     type=str, default="test",
                   choices=["test", "full"])
    p.add_argument("--cells_dir", type=str, default=None)
    return p.parse_args()


def _parse_configs(args) -> list:
    """Return a list of (rec_mode, code_dim) tuples."""
    if args.configs:
        result = []
        for spec in args.configs:
            parts = spec.split(":")
            if len(parts) != 2:
                raise ValueError(f"Formato inválido '{spec}'. Usa rec_mode:code_dim (p.ej. mse:128)")
            result.append((parts[0], int(parts[1])))
        return result
    if not args.code_dim:
        raise ValueError("--code_dim es requerido con --modes.")
    return [(m, args.code_dim) for m in args.modes]


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"No se encontró cells/: {cells_dir}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs   = _parse_configs(args)

    print(f"\nDispositivo      : {device}")
    print(f"Split            : {args.split}")
    print(f"Configs a comparar: {configs}")
    print(f"Diseño eval      : Opción B — máscara gaussiana en todas las métricas")
    print(f"Checkpoints      : best.pt Y last.pt para cada config")

    best_ckpts = []
    last_ckpts = []
    for rec_mode, code_dim in configs:
        best = cfg.exp_checkpoints_dir(rec_mode, code_dim, "best") / "best.pt"
        last = cfg.exp_checkpoints_dir(rec_mode, code_dim, "last") / "last.pt"
        if not best.exists():
            raise FileNotFoundError(f"best.pt no encontrado para {rec_mode}/dim_{code_dim}")
        if not last.exists():
            raise FileNotFoundError(f"last.pt no encontrado para {rec_mode}/dim_{code_dim}")
        best_ckpts.append(best)
        last_ckpts.append(last)

    print("\nCargando dataset...")
    dataset    = CellsDataset(cells_dir, augment=False, flatten=True)
    data_range = dataset.data_range
    N          = len(dataset)
    print(f"  {N} imágenes  |  data_range={data_range:.4f}")

    if args.split == "test":
        print("\nCalculando intersección de test sets...")
        shared_indices = compute_shared_indices(configs, best_ckpts)
        shared_n       = len(shared_indices)
        print(f"  Intersección: {shared_n} índices")
        for (rec_mode, code_dim), ckpt in zip(configs, best_ckpts):
            own = load_test_indices(rec_mode, code_dim, ckpt)
            pct = 100 * shared_n / max(len(own), 1)
            config_tag = f"{rec_mode}/dim_{code_dim}"
            print(f"    {config_tag:<40} test={len(own):>5}  shared={shared_n}  ({pct:.1f}%)")
        if shared_n < 100:
            print(f"\n  [WARNING] Intersección de solo {shared_n} muestras (<100). "
                  "Resultados pueden ser poco representativos.")
    else:
        shared_indices = list(range(N))
        shared_n       = N
        print(f"\nUsando split='full': {shared_n} imágenes")

    out_dir = cfg.exp_comparativa_dir("2_metrics")
    shared_json_path = out_dir / f"shared_test_{timestamp}.json"
    with open(shared_json_path, "w") as f:
        json.dump({
            "timestamp":      timestamp,
            "split":          args.split,
            "n_shared":       shared_n,
            "shared_indices": shared_indices,
            "configs":        [{"rec_mode": rm, "code_dim": cd} for rm, cd in configs],
            "eval_design":    "gaussian_mask_opcion_B",
        }, f, indent=2)
    print(f"\nIntersección guardada en: {shared_json_path}")

    rows_best = []
    rows_last = []

    for (rec_mode, code_dim), best_ckpt, last_ckpt in zip(configs, best_ckpts, last_ckpts):
        config_tag = f"{rec_mode}/dim_{code_dim}"
        print(f"\n{'='*55}\n  Config: {config_tag}")

        print(f"  Evaluando best.pt ...")
        m_best = evaluate_on_indices(best_ckpt, dataset, shared_indices, data_range, device)
        rows_best.append({"run_name": config_tag, "run_dir": str(best_ckpt.parent), **m_best})
        print(f"    epoch={m_best['epoch']}  MAE={m_best['mae_gauss']:.6f}  "
              f"MSE={m_best['mse_gauss']:.6f}  PSNR={m_best['psnr_db']:.3f}  "
              f"SSIM={m_best['ssim_gauss']:.6f}")

        print(f"  Evaluando last.pt ...")
        m_last = evaluate_on_indices(last_ckpt, dataset, shared_indices, data_range, device)
        rows_last.append({"run_name": config_tag, "run_dir": str(last_ckpt.parent), **m_last})
        print(f"    epoch={m_last['epoch']}  MAE={m_last['mae_gauss']:.6f}  "
              f"MSE={m_last['mse_gauss']:.6f}  PSNR={m_last['psnr_db']:.3f}  "
              f"SSIM={m_last['ssim_gauss']:.6f}")

    print_table(rows_best, shared_n, "BEST.PT")
    print_table(rows_last, shared_n, "LAST.PT")

    csv_best = out_dir / f"compare_{timestamp}_best.csv"
    csv_last = out_dir / f"compare_{timestamp}_last.csv"

    print()
    save_csv(rows_best, csv_best, shared_n, shared_json_path, "BEST.PT")
    save_csv(rows_last, csv_last, shared_n, shared_json_path, "LAST.PT")

    png_path = out_dir / f"compare_{timestamp}.png"
    save_comparison_table_png(rows_best, rows_last, shared_n, png_path)

    print(f"\nFicheros generados en: {out_dir}")


if __name__ == "__main__":
    main()
