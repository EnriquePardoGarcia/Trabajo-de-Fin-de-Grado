"""
Classifier over sparse (LISTA) features — KNN and MLP.

Loads a trained LISTA encoder, extracts the sparse codes Z for the canonical
train/test split, and trains a KNN and/or MLP on those features.

Usage:
    python evaluation/sparse_classifier.py --rec_mode mse --code_dim 128
    python evaluation/sparse_classifier.py --checkpoint experiments/1_checkpoints/mse/dim_128/best/best.pt
    python evaluation/sparse_classifier.py --rec_mode mse --code_dim 128 --classifier knn
    python evaluation/sparse_classifier.py --rec_mode mse --code_dim 128 --classifier both --knn_k 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from core import configuration as cfg
from core.dataset import CellsDataset
from core.model import LinearLISTADecoder, LinearLISTAEncoder

CLASS_NAMES = ["mitotic_figure", "not_mitotic_figure"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(ckpt_path: Path, cells_dir: Path, device,
                     rec_mode: str = None, code_dim: int = None):
    """Extract Z (N, code_dim) using the LISTA encoder and return train/test splits."""
    ckpt    = torch.load(ckpt_path, map_location=device)
    hparams = ckpt.get("hparams", {})

    encoder = LinearLISTAEncoder(
        cfg.IN_DIM,
        hparams.get("code_dim", cfg.CODE_DIM),
        hparams.get("num_iters", cfg.NUM_ITERS),
    ).to(device)
    LinearLISTADecoder(encoder)  # needed for tied weights
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    stats   = (hparams["dataset_mean"], hparams["dataset_std"])
    dataset = CellsDataset(cells_dir, augment=False, flatten=True, stats=stats)
    loader  = DataLoader(dataset, batch_size=cfg.EVAL_BATCH_SIZE,
                         shuffle=False, num_workers=4,
                         pin_memory=(device.type == "cuda"))

    all_z, all_labels = [], []
    for xb, yb in loader:
        all_z.append(encoder(xb.to(device, non_blocking=True)).cpu().numpy())
        all_labels.append(yb.numpy())

    Z      = np.vstack(all_z)
    labels = np.concatenate(all_labels)

    # Look for split.json: first in the experiment's canonical location,
    # then embedded in the checkpoint itself
    split_path = None
    if rec_mode is not None and code_dim is not None:
        split_path = cfg.exp_split_path(rec_mode, code_dim)
    if split_path is None or not split_path.exists():
        # derive from the checkpoint path
        rm, cd = cfg.rec_mode_and_dim_from_ckpt(ckpt_path)
        if rm is not None:
            split_path = cfg.exp_split_path(rm, cd)

    if split_path is not None and split_path.exists():
        with open(split_path) as f:
            split_data = json.load(f)
        train_idx = split_data["train_indices"]
        test_idx  = split_data["test_indices"]
    else:
        test_idx = ckpt.get("test_indices", None)
        if test_idx is None:
            raise RuntimeError("No se encontró split.json ni test_indices en el checkpoint.")
        test_set  = set(test_idx)
        train_idx = [i for i in range(len(dataset)) if i not in test_set]

    Z_train = Z[train_idx];  y_train = labels[train_idx]
    Z_test  = Z[test_idx];   y_test  = labels[test_idx]

    n_mit  = int((y_test == 0).sum())
    n_nmit = int((y_test == 1).sum())
    print(f"  Features extraídas: train={len(Z_train)}  test={len(Z_test)}")
    print(f"  Test — mitotic={n_mit}  non-mitotic={n_nmit}")

    return Z_train, y_train, Z_test, y_test, hparams, ckpt.get("epoch", "?")


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def run_knn(Z_train, y_train, Z_test, y_test, n_neighbors=5):
    print(f"\n  [KNN]  k={n_neighbors}")
    scaler = StandardScaler()
    Ztr_s  = scaler.fit_transform(Z_train)
    Zte_s  = scaler.transform(Z_test)

    knn   = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    knn.fit(Ztr_s, y_train)

    preds = knn.predict(Zte_s)
    probs = knn.predict_proba(Zte_s)[:, 0]  # P(mitotic)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score((y_test == 0).astype(int), probs)
    cm    = confusion_matrix(y_test, preds)

    print(f"    Accuracy : {acc:.4f}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

    return {
        "classifier":  "KNN",
        "n_neighbors": n_neighbors,
        "accuracy":    acc,
        "auc":         auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, preds, target_names=CLASS_NAMES, output_dict=True
        ),
    }, cm


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      nn.BatchNorm1d(hidden),      nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def run_mlp(Z_train, y_train, Z_test, y_test,
            hidden=64, num_epochs=100, lr=1e-3, batch_size=256, device=None):
    print(f"\n  [MLP]  hidden={hidden}  epochs={num_epochs}  lr={lr}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    Ztr_s  = torch.from_numpy(scaler.fit_transform(Z_train).astype(np.float32))
    Zte_s  = torch.from_numpy(scaler.transform(Z_test).astype(np.float32))
    ytr    = torch.from_numpy(y_train.astype(np.int64))
    yte    = torch.from_numpy(y_test.astype(np.int64))

    train_loader = DataLoader(TensorDataset(Ztr_s, ytr),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Zte_s, yte),
                              batch_size=batch_size, shuffle=False)

    model     = MLP(Z_train.shape[1], hidden=hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)
        scheduler.step()

        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                vl_loss    += loss.item() * xb.size(0)
                vl_correct += (logits.argmax(1) == yb).sum().item()
                vl_total   += xb.size(0)

        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(vl_loss / vl_total)
        history["train_acc"].append(tr_correct / tr_total)
        history["val_acc"].append(vl_correct / vl_total)

        if epoch <= 5 or epoch % 20 == 0:
            print(f"    Epoch {epoch:>3}/{num_epochs}  "
                  f"loss={tr_loss/tr_total:.4f}/{vl_loss/vl_total:.4f}  "
                  f"acc={tr_correct/tr_total:.4f}/{vl_correct/vl_total:.4f}")

    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    acc = accuracy_score(all_true, all_preds)
    auc = roc_auc_score((all_true == 0).astype(int), all_probs)
    cm  = confusion_matrix(all_true, all_preds)

    print(f"\n    Resultado final MLP:")
    print(f"    Accuracy : {acc:.4f}  |  AUC: {auc:.4f}")
    print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=4))

    return {
        "classifier": "MLP",
        "hidden":     hidden,
        "num_epochs": num_epochs,
        "accuracy":   acc,
        "auc":        auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            all_true, all_preds, target_names=CLASS_NAMES, output_dict=True
        ),
        "history": history,
    }, cm, history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Mitotic", "Non-mitotic"])
    ax.set_yticklabels(["Mitotic", "Non-mitotic"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    _save(fig, out_path)


def plot_mlp_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("MLP sobre features LISTA — curvas de entrenamiento", fontsize=11)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_path)


def plot_comparison(results, out_path):
    names = [r["classifier"] for r in results]
    accs  = [r["accuracy"]   for r in results]
    aucs  = [r["auc"]        for r in results]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2.5), 4))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, aucs, w, label="AUC",      color="darkorange", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Clasificadores sobre features LISTA — comparativa")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    for i, (acc, auc) in enumerate(zip(accs, aucs)):
        ax.text(i - w/2, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, auc + 0.01, f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Helpers to resolve checkpoint from rec_mode/code_dim
# ---------------------------------------------------------------------------

def _resolve_checkpoint(args) -> tuple[Path, str, int]:
    """Return (ckpt_path, rec_mode, code_dim)."""
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        rec_mode, code_dim = cfg.rec_mode_and_dim_from_ckpt(ckpt_path)
        if rec_mode is None:
            rec_mode = args.rec_mode or "unknown"
            code_dim = args.code_dim or cfg.CODE_DIM
        return ckpt_path, rec_mode, code_dim

    if not args.rec_mode or not args.code_dim:
        raise ValueError("Especifica --rec_mode y --code_dim, o --checkpoint.")

    rec_mode  = args.rec_mode
    code_dim  = args.code_dim
    ckpt_type = args.ckpt_type or "best"
    ckpt_dir  = cfg.exp_checkpoints_dir(rec_mode, code_dim, ckpt_type)
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {ckpt_dir}")
    ckpt_path = sorted(candidates)[-1]
    return ckpt_path, rec_mode, code_dim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="KNN/MLP sobre features sparse extraídas del encoder LISTA."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, default=None,
                   help="Ruta directa al checkpoint LISTA (.pt).")
    g.add_argument("--rec_mode",   type=str, default=None,
                   choices=["mse", "mae", "ssim"],
                   help="Modo de reconstrucción (usa con --code_dim).")
    p.add_argument("--code_dim",   type=int, default=None,
                   help="Dimensión del código (usa con --rec_mode).")
    p.add_argument("--ckpt_type",  type=str, default="best",
                   choices=["best", "last"],
                   help="Tipo de checkpoint a evaluar (best o last). Default: best.")
    p.add_argument("--cells_dir",  type=str, default=None)
    p.add_argument("--classifier", type=str, default="both",
                   choices=["knn", "mlp", "both"])
    p.add_argument("--knn_k",      type=int,   default=5)
    p.add_argument("--mlp_hidden", type=int,   default=64)
    p.add_argument("--mlp_epochs", type=int,   default=100)
    p.add_argument("--mlp_lr",     type=float, default=1e-3)
    p.add_argument("--mlp_batch",  type=int,   default=256)
    return p.parse_args()


def main():
    args      = parse_args()
    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR

    if not cells_dir.exists():
        raise FileNotFoundError(f"cells/ no encontrado: {cells_dir}")

    ckpt_path, rec_mode, code_dim = _resolve_checkpoint(args)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory split by classifier type
    knn_dir = cfg.exp_analysis_dir(rec_mode, code_dim, "weak_classifiers", "knn")
    mlp_dir = cfg.exp_analysis_dir(rec_mode, code_dim, "weak_classifiers", "mlp")

    print(f"\n{'='*60}")
    print(f"Sparse classifier — features LISTA")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"rec_mode     : {rec_mode}  |  code_dim: {code_dim}")
    print(f"Classifier   : {args.classifier}")
    print(f"Dispositivo  : {device}")
    print(f"Output KNN   : {knn_dir}")
    print(f"Output MLP   : {mlp_dir}")
    print(f"{'='*60}\n")

    Z_train, y_train, Z_test, y_test, hparams, epoch = extract_features(
        ckpt_path, cells_dir, device, rec_mode=rec_mode, code_dim=code_dim
    )
    config = hparams.get("config", "?")
    print(f"\n  Config: {config}  |  epoch: {epoch}  |  code_dim: {Z_train.shape[1]}")

    # Includes rec_mode and code_dim in the filename: these files get compared
    # across all six configurations, and a name like "summary_best_best.json"
    # wouldn't say which one it came from.
    ckpt_tag = f"{rec_mode}_{code_dim}_{ckpt_path.parent.name}_{ckpt_path.stem}"
    all_results = []

    if args.classifier in ("knn", "both"):
        res, cm = run_knn(Z_train, y_train, Z_test, y_test, n_neighbors=args.knn_k)
        res.update({"config": config, "epoch": epoch, "checkpoint": str(ckpt_path)})
        all_results.append(res)
        with open(knn_dir / f"knn_results_{ckpt_tag}.json", "w") as f:
            json.dump(res, f, indent=2)
        plot_confusion_matrix(cm, f"KNN (k={args.knn_k}) — features LISTA",
                              knn_dir / f"knn_confusion_matrix_{ckpt_tag}.png")

    if args.classifier in ("mlp", "both"):
        res, cm, hist = run_mlp(
            Z_train, y_train, Z_test, y_test,
            hidden=args.mlp_hidden, num_epochs=args.mlp_epochs,
            lr=args.mlp_lr, batch_size=args.mlp_batch, device=device,
        )
        res.update({"config": config, "epoch": epoch, "checkpoint": str(ckpt_path)})
        all_results.append(res)
        with open(mlp_dir / f"mlp_results_{ckpt_tag}.json", "w") as f:
            json.dump(res, f, indent=2)
        plot_confusion_matrix(cm, "MLP — features LISTA",
                              mlp_dir / f"mlp_confusion_matrix_{ckpt_tag}.png")
        plot_mlp_curves(hist, mlp_dir / f"mlp_training_curves_{ckpt_tag}.png")

    if len(all_results) > 1:
        cmp_dir = cfg.exp_analysis_dir(rec_mode, code_dim, "weak_classifiers")
        plot_comparison(all_results, cmp_dir / f"comparison_{ckpt_tag}.png")

    summary = {
        "checkpoint": str(ckpt_path),
        "rec_mode":   rec_mode,
        "code_dim":   int(Z_train.shape[1]),
        "config":     config,
        "epoch":      epoch,
        "n_train":    int(len(Z_train)),
        "n_test":     int(len(Z_test)),
        "results":    all_results,
    }
    cmp_dir = cfg.exp_analysis_dir(rec_mode, code_dim, "weak_classifiers")
    with open(cmp_dir / f"summary_{ckpt_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResultados guardados en: {cmp_dir}")


if __name__ == "__main__":
    main()
