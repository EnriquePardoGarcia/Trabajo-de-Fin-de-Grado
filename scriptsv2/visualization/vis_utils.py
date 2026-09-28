"""Shared utilities: model loading, metrics, and image helpers."""

import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import configuration as cfg
from core.loss  import _ssim_kernel, _ssim_map, gaussian_center_mask
from core.model import LinearLISTAEncoder, LinearLISTADecoder

H = W = cfg.PATCH_SIZE
C = 3


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_img(x_flat: torch.Tensor, mean: float, std: float) -> np.ndarray:
    img = x_flat.view(C, H, W).detach().cpu()
    img = (img * std + mean).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def atom_img(a: torch.Tensor) -> np.ndarray:
    img = a.view(C, H, W).detach().cpu()
    mx  = img.abs().max().clamp(min=1e-8)
    img = (img / mx + 1.0) / 2.0
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Individual reconstruction metrics
# ---------------------------------------------------------------------------

def compute_mse_gauss_single(x_flat, x_hat_flat) -> float:
    w    = gaussian_center_mask().view(1, H, W).expand(C, H, W)
    diff = (x_hat_flat.view(C, H, W) - x_flat.view(C, H, W)) ** 2
    return (w * diff).mean().item()


def compute_mae_gauss_single(x_flat, x_hat_flat) -> float:
    w    = gaussian_center_mask().view(1, H, W).expand(C, H, W)
    diff = (x_hat_flat.view(C, H, W) - x_flat.view(C, H, W)).abs()
    return (w * diff).mean().item()


def compute_ssim_single(x_flat, x_hat_flat, data_range: float) -> float:
    x4d  = x_flat.view(1, C, H, W)
    xh4d = x_hat_flat.view(1, C, H, W)
    kernel   = _ssim_kernel(cfg.SSIM_KERNEL_SIZE, cfg.SSIM_SIGMA, C, x4d.device)
    pad      = cfg.SSIM_KERNEL_SIZE // 2
    C1       = (cfg.SSIM_K1 * data_range) ** 2
    C2       = (cfg.SSIM_K2 * data_range) ** 2
    ssim_map = _ssim_map(x4d, xh4d, kernel, pad, C1, C2)
    w        = gaussian_center_mask().view(1, 1, H, W)
    return (ssim_map * w).mean().item()


def get_rec_metric_fn(vis_metric: str, data_range: float):
    if vis_metric == "mse":
        return (lambda x, xh: compute_mse_gauss_single(x.cpu(), xh.cpu()), "MSE_gauss")
    elif vis_metric == "mae":
        return (lambda x, xh: compute_mae_gauss_single(x.cpu(), xh.cpu()), "MAE_gauss")
    else:
        return (lambda x, xh: compute_ssim_single(x.cpu(), xh.cpu(), data_range), "SSIM_gauss")


# ---------------------------------------------------------------------------
# Model, history, and split loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device):
    ckpt      = torch.load(ckpt_path, map_location=device)
    hparams   = ckpt.get("hparams", {})
    epoch     = ckpt.get("epoch", "?")
    code_dim  = hparams.get("code_dim", cfg.CODE_DIM)
    enc_state = ckpt["encoder_state"]
    encoder   = LinearLISTAEncoder(
        cfg.IN_DIM, code_dim, hparams.get("num_iters", cfg.NUM_ITERS)
    ).to(device)
    decoder = LinearLISTADecoder(encoder).to(device)
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval(); decoder.eval()
    print(f"  Cargado: {ckpt_path.name}  (epoch {epoch}  config={hparams.get('config','?')})")
    return encoder, decoder, hparams, epoch


def load_history(rec_mode: str, code_dim: int, ckpt_path: Path = None):
    history_path = cfg.exp_logs_dir(rec_mode, code_dim) / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        h = ckpt.get("history", {})
        if h:
            return h
    return {}


def get_split_indices(rec_mode: str, code_dim: int, ckpt_path: Path, dataset_len: int):
    split_path = cfg.exp_split_path(rec_mode, code_dim)
    if split_path.exists():
        with open(split_path) as f:
            sp = json.load(f)
        return sp["train_indices"], sp["test_indices"]
    ckpt_raw  = torch.load(ckpt_path, map_location="cpu")
    test_idx  = ckpt_raw.get("test_indices", [])
    test_set  = set(test_idx)
    train_idx = [i for i in range(dataset_len) if i not in test_set]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Evaluation JSON loading
# ---------------------------------------------------------------------------

def load_eval_json(rec_mode: str, code_dim: int, ckpt: str) -> dict | None:
    path = (cfg.exp_metrics_dir(rec_mode, code_dim, "reconstruction", "per_class")
            / f"eval_test_{ckpt}.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
