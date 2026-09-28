from datetime import datetime
from pathlib import Path

# Root directories of the project
CELLS_DIR       = Path(__file__).resolve().parent.parent.parent / "data" / "cells"
RUNS_DIR        = Path(__file__).resolve().parent.parent.parent / "runs"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# LISTA model dimensions: PATCH_SIZE is the input patch side (px), IN_DIM is
# the flattened RGB patch size, and CODE_DIM is the number of dictionary atoms.
PATCH_SIZE = 64
CODE_DIM   = 128
IN_DIM     = PATCH_SIZE * PATCH_SIZE * 3
NUM_ITERS = 3      # ISTA unrolling steps

# Training hyperparameters
NUM_EPOCHS   = 30000
BATCH_SIZE   = 4096
LMBD_REC     = 1.0    # weight of the reconstruction loss

# Sparsity comes only from ReLU + L1 on z (no TopK); the shrinkage threshold
# is learned as S's bias. The active-atom-target controller in train.py runs
# in two phases after warmup:
#   Phase A - L1 stays at 0 until reconstruction plateaus, i.e. until the
#             best MSE over a window of L1_CONV_PATIENCE epochs improves by
#             less than L1_CONV_REL_TOL relative to the previous window.
#   Phase B - lmbd starts at L1_SEED and is scaled by (1 +/- L1_GROWTH) per
#             epoch to drive the EMA of active atoms/sample toward
#             L1_TARGET_ACTIVE_FRAC * code_dim, within a +/-L1_ACTIVE_TOL
#             dead band. Reconstruction quality is not gated in this phase.
LMBD_L1               = 1.0    # safety ceiling for lmbd; the feedback loop is self-limiting,
                                # so this cap is a margin rather than an operating point
L1_SEED               = 1e-7   # initial lmbd when Phase B starts
L1_GROWTH             = 0.05   # per-epoch multiplicative step (x1.05 up/down)
L1_TARGET_ACTIVE_FRAC = 0.50   # target active atoms/sample, as a fraction of code_dim
L1_ACTIVE_TOL         = 0.05   # dead band: no adjustment while |active - target| < tol * target
L1_CONV_PATIENCE      = 400    # half-window (epochs) for the Phase A plateau criterion
L1_CONV_REL_TOL       = 0.003  # minimum relative MSE improvement between windows to keep waiting

WARMUP_EPOCHS        = 100    # warmup epochs (S frozen, lambda_l1=0)
FREEZE_S_WARMUP      = True   # keep S=0 during warmup

# Dead-atom revival (neuron resampling, Anthropic "Towards Monosemanticity"):
# every N epochs, W_e of inactive atoms is reoriented toward poorly reconstructed patches.
ALIVE_WINDOW_EPOCHS  = 10    # window in epochs used to compute alive atoms
WARMUP_REINIT_EVERY  = 10    # resample dead atoms every N epochs during warmup (0 = off)
REVIVAL_EVERY        = 0     # resample dead atoms every N epochs post-warmup (0 = off)

LR           = 1e-3
WEIGHT_DECAY = 1e-4
S_WEIGHT_DECAY = 1e-2        # weight decay for S.weight (higher than W_e to avoid a kill matrix)
PRINT_EVERY  = 10

TRAIN_FRACTION = 0.80
TEST_FRACTION  = 0.20

# Sigma of the centered Gaussian mask (in [-1, 1] coordinates)
GAUSSIAN_SIGMA = 0.35

# SSIM_DATA_RANGE is not defined here as a constant; it is computed from the data
# in CellsDataset._compute_statistics() and propagated as hparams["ssim_data_range"].
SSIM_KERNEL_SIZE = 11
SSIM_SIGMA       = 1.5
SSIM_K1          = 0.01
SSIM_K2          = 0.03

# Data augmentation parameters
AUG_HFLIP_P          = 0.5    # random horizontal flip (cells have no preferred orientation)
AUG_VFLIP_P          = 0.5    # random vertical flip
AUG_ROTATION_DEGREES = 180    # random rotation +-180 deg; corners filled with the mean
AUG_BRIGHTNESS       = 0.40   # brightness jitter x[0.6, 1.4]: simulates illumination variation
AUG_CONTRAST         = 0.40   # contrast jitter x[0.6, 1.4]: compensates for staining differences
AUG_BLUR_KERNEL      = 3      # Gaussian blur: simulates scanner focus variation
AUG_BLUR_SIGMA_MIN   = 0.1    # minimum blur sigma (near-negligible effect)
AUG_BLUR_SIGMA_MAX   = 2.0    # maximum blur sigma (noticeable blur)
AUG_BLUR_P           = 0.5    # probability of applying the blur
# Each image is seen AUG_REPEAT times per epoch with different augmentations.
AUG_REPEAT           = 1

# Evaluation parameters
EVAL_BATCH_SIZE  = 512
# Single sigma used in evaluate.py with ignite.metrics.SSIM
EVAL_SSIM_SIGMA  = 1.5

VIS_N_TOP_ATOMS = 6  # most active atoms to show in visualizations

CONFIG_MAP = {
    "mse_no_aug":  {"rec_mode": "mse",  "augment": False},
    "mse_aug":     {"rec_mode": "mse",  "augment": True},
    "mae_no_aug":  {"rec_mode": "mae",  "augment": False},
    "mae_aug":     {"rec_mode": "mae",  "augment": True},
    "ssim_no_aug": {"rec_mode": "ssim", "augment": False},
    "ssim_aug":    {"rec_mode": "ssim", "augment": True},
}

EVAL_METRICS = ["mse", "mae", "ssim"]


# ---------------------------------------------------------------------------
# Helpers for the experiments/ tree
# ---------------------------------------------------------------------------

def _dim_str(code_dim: int) -> str:
    return f"dim_{code_dim}"


def exp_logs_dir(rec_mode: str, code_dim: int) -> Path:
    """Return (and create) experiments/6_logs/<rec_mode>/dim_<code_dim>/."""
    d = EXPERIMENTS_DIR / "6_logs" / rec_mode / _dim_str(code_dim)
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_checkpoints_dir(rec_mode: str, code_dim: int, ckpt_type: str = "") -> Path:
    """Return (and create) experiments/1_checkpoints/<rec_mode>/dim_<code_dim>/[best|last|jumps]/."""
    base = EXPERIMENTS_DIR / "1_checkpoints" / rec_mode / _dim_str(code_dim)
    d = (base / ckpt_type) if ckpt_type else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_metrics_dir(rec_mode: str, code_dim: int, *subdirs: str) -> Path:
    """Return (and create) experiments/2_metrics/<rec_mode>/dim_<code_dim>/[subdirs...]/."""
    d = EXPERIMENTS_DIR / "2_metrics" / rec_mode / _dim_str(code_dim)
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_vis_dir(rec_mode: str, code_dim: int, *subdirs: str) -> Path:
    """Return (and create) experiments/3_visualization/<rec_mode>/dim_<code_dim>/[subdirs...]/."""
    d = EXPERIMENTS_DIR / "3_visualization" / rec_mode / _dim_str(code_dim)
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_analysis_dir(rec_mode: str, code_dim: int, *subdirs: str) -> Path:
    """Return (and create) experiments/4_analysis/<rec_mode>/dim_<code_dim>/[subdirs...]/."""
    d = EXPERIMENTS_DIR / "4_analysis" / rec_mode / _dim_str(code_dim)
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_dicts_dir(rec_mode: str, code_dim: int) -> Path:
    """Return (and create) experiments/5_dictionaries/<rec_mode>/summary_dim_<code_dim>/."""
    d = EXPERIMENTS_DIR / "5_dictionaries" / rec_mode / f"summary_{_dim_str(code_dim)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_comparativa_dir(section_num: str, *subdirs: str) -> Path:
    """Return (and create) experiments/<section_num>/global_comparison/[subdirs...]/."""
    d = EXPERIMENTS_DIR / section_num / "global_comparison"
    for s in subdirs:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers for the runs/ tree (internal per-run structure)
# ---------------------------------------------------------------------------

def make_run_dir(config_name: str, rec_mode: str, timestamp: str = None) -> Path:
    """Create the run directory with standard subdirectories and return its path.

    Structure: runs/<rec_mode>/<config_name>_<timestamp>/
    Only creates checkpoints/ and logs/. results/ and visualizations/ are created
    on demand when running evaluate.py or visualize.py.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / rec_mode / f"{config_name}_{timestamp}"
    for subdir in ("checkpoints", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def make_results_dir(run_dir: Path, eval_metric: str) -> Path:
    """Create and return the results subfolder for a given evaluation metric.

    Structure: <run_dir>/results/eval_<eval_metric>/
    """
    d = run_dir / "results" / f"eval_{eval_metric}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_vis_dir(run_dir: Path, vis_metric: str) -> Path:
    """Create and return the visualizations subfolder for a given metric.

    Structure: <run_dir>/visualizations/vis_<vis_metric>/
    Subfolders history/, best/, last/, compare/ are created inside it.
    """
    d = run_dir / "visualizations" / f"vis_{vis_metric}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir_from_checkpoint(ckpt_path: Path) -> Path:
    """Return the run_dir from a checkpoint path (legacy runs/ structure)."""
    return Path(ckpt_path).resolve().parent.parent


def exp_split_path(rec_mode: str, code_dim: int) -> Path:
    """Canonical split path: experiments/1_checkpoints/<rec_mode>/dim_<code_dim>/split.json"""
    return exp_checkpoints_dir(rec_mode, code_dim) / "split.json"


def rec_mode_and_dim_from_ckpt(ckpt_path: Path):
    """Derive (rec_mode, code_dim) from a path under experiments/1_checkpoints/.

    Expected structure: .../1_checkpoints/<rec_mode>/dim_<code_dim>/<ckpt_type>/<name>.pt

    Returns:
        (rec_mode, code_dim), or (None, None) if it cannot be derived.
    """
    p = Path(ckpt_path).resolve()
    try:
        dim_dir  = p.parent.parent       # dim_128/
        mode_dir = dim_dir.parent        # mse/
        code_dim = int(dim_dir.name.split("_", 1)[1])
        rec_mode = mode_dir.name
        return rec_mode, code_dim
    except Exception:
        return None, None
