"""
Supervised CNN — baseline for mitotic vs. non-mitotic classification.

Trains a simple CNN on 64x64x3 patches with direct labels.
Training only; evaluation is done with a separate script.

If --split_json is passed, it reuses the train/test split from a LISTA run
to ensure direct comparability.

Usage:
    python baseline/baseline_cnn.py
    python baseline/baseline_cnn.py --split_json experiments/1_checkpoints/mse/dim_128/split.json
    python baseline/baseline_cnn.py --num_epochs 100 --lr 1e-3
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from core import configuration as cfg
from core.dataset import CellsDataset
from baseline import cnn_cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Supervised CNN for 64x64x3 patches -> binary (mitotic / non-mitotic)."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.MaxPool2d(2),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.MaxPool2d(2),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += xb.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CNN baseline — curvas de entrenamiento", fontsize=12)

    axes[0].plot(epochs, history["train_loss"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _make_split(N, train_frac, test_frac):
    idx    = list(range(N))
    random.shuffle(idx)
    n_test = max(1, round(N * test_frac))
    return idx[:-n_test], idx[-n_test:]


def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena CNN supervisada (baseline) para clasificación mitótica."
    )
    p.add_argument("--cells_dir",      type=str,   default=None)
    p.add_argument("--split_json",     type=str,   default=None,
                   help="split.json de un run LISTA para reutilizar el mismo train/test.")
    p.add_argument("--num_epochs",     type=int,   default=cnn_cfg.NUM_EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=cnn_cfg.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=cnn_cfg.LR)
    p.add_argument("--weight_decay",   type=float, default=cnn_cfg.WEIGHT_DECAY)
    p.add_argument("--dropout",        type=float, default=cnn_cfg.DROPOUT)
    p.add_argument("--augment",        action="store_true",  default=cnn_cfg.AUGMENT)
    p.add_argument("--no_augment",     dest="augment", action="store_false")
    p.add_argument("--train_fraction", type=float, default=cnn_cfg.TRAIN_FRACTION)
    p.add_argument("--test_fraction",  type=float, default=cnn_cfg.TEST_FRACTION)
    return p.parse_args()


def main():
    args = parse_args()

    if "CNN_RUN_DIR" not in os.environ:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = cfg.RUNS_DIR / "cnn" / f"cnn_{timestamp}"
        logs_dir  = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path  = logs_dir / f"train_1_{args.num_epochs}.log"

        env = os.environ.copy()
        env["CNN_RUN_DIR"] = str(run_dir)

        with open(log_path, "w") as log_file:
            subprocess.Popen(
                ["nohup", sys.executable] + sys.argv,
                stdout     = log_file,
                stderr     = log_file,
                stdin      = subprocess.DEVNULL,
                env        = env,
                preexec_fn = os.setpgrp,
            )

        print(f"Run dir : {run_dir}")
        print(f"Log     : {log_path}")
        print(f"\nSeguir progreso:")
        print(f"  tail -f {log_path}")
        return

    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ no encontrado: {cells_dir}")

    run_dir = Path(os.environ["CNN_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"CNN Baseline supervisada")
    print(f"Dispositivo  : {device}")
    print(f"Run dir      : {run_dir}")
    print(f"Epochs       : {args.num_epochs}  |  lr: {args.lr}  |  batch: {args.batch_size}")
    print(f"Augmentation : {args.augment}")
    print(f"Split        : train={args.train_fraction}  test={args.test_fraction}")
    print(f"{'='*60}\n")

    dataset = CellsDataset(cells_dir, augment=args.augment, flatten=False)
    N       = len(dataset)

    if args.split_json:
        split_path = Path(args.split_json)
        with open(split_path) as f:
            split_data = json.load(f)
        train_idx = split_data["train_indices"]
        test_idx  = split_data["test_indices"]
        print(f"Split cargado desde: {split_path}")
    else:
        train_idx, test_idx = _make_split(N, args.train_fraction, args.test_fraction)
        with open(run_dir / "split.json", "w") as f:
            json.dump({
                "train_indices": train_idx, "test_indices": test_idx,
                "n_train": len(train_idx),  "n_test":  len(test_idx),
            }, f, indent=2)

    print(f"Train: {len(train_idx):>6}  |  Test: {len(test_idx):>6}  (test reservado para evaluación)")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    model     = SimpleCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print(f"Parámetros CNN: {sum(p.numel() for p in model.parameters()):,}\n")

    history = {"train_loss": [], "train_acc": []}

    for epoch in range(1, args.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if epoch <= 5 or epoch % 5 == 0:
            print(f"Epoch {epoch:>4}/{args.num_epochs}  loss={tr_loss:.4f}  acc={tr_acc:.4f}")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "args": vars(args),
                "test_indices": test_idx}, ckpt_dir / "last.pt")

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, logs_dir / "training_curves.png")

    print(f"\nEntrenamiento finalizado.")
    print(f"Checkpoint : {ckpt_dir / 'last.pt'}")
    print(f"Split      : {run_dir / 'split.json'}")


if __name__ == "__main__":
    main()
