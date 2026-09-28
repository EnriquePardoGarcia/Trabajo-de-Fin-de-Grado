#!/usr/bin/env python3
"""LISTA training monitor.

One-shot mode (default):
    python3 monitor.py --log <log>              # PNG with all available epochs
    python3 monitor.py --log <log> --epoch 8000 # PNG up to epoch 8000

Live mode (automatic refresh):
    python3 monitor.py --log <log> --live
    Controls:  p = pause/resume   q = close
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|"
    r"\s+loss=(\S+)\s+(\w+)=(\S+)\s+l1=(\S+)(?:\s+aux=\S+)?\s+\|"
    r"\s+zeros=(\S+)\s+active=(\S+)\s+alive=(\d+)\s+dead=(\d+)\s+dying=(\d+)"
    r".*?lmbd_l1=(\S+)"
)


def parse_log(path: Path, max_epoch: int = None):
    data = {k: [] for k in ("epoch", "total", "loss", "rec", "l1",
                             "zeros", "active", "alive", "dead", "dying", "lmbd")}
    rec_label = "rec_loss"
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if not m:
                    continue
                ep = int(m.group(1))
                if max_epoch is not None and ep > max_epoch:
                    continue
                data["epoch"].append(ep)
                data["total"].append(int(m.group(2)))
                data["loss"].append(float(m.group(3)))
                rec_label = m.group(4)
                data["rec"].append(float(m.group(5)))
                data["l1"].append(float(m.group(6)))
                data["zeros"].append(float(m.group(7)))
                data["active"].append(float(m.group(8)))
                data["alive"].append(int(m.group(9)))
                data["dead"].append(int(m.group(10)))
                data["dying"].append(int(m.group(11)))
                data["lmbd"].append(float(m.group(12)))
    except FileNotFoundError:
        print(f"[Error] No se encontró el log: {path}", file=sys.stderr)
    return {k: np.array(v) for k, v in data.items()}, rec_label


# File names of the form  train_<start>_<end>.log
RUN_LOG_RE = re.compile(r"train_(\d+)_\d+\.log$")


def find_run_logs(log_path: Path):
    """Find all .log files in the same folder named train_<start>_<end>.log,
    sorted by start epoch. Training is split across multiple files, so all of
    them must be read to reach the last epoch."""
    logs = []
    for p in log_path.parent.glob("train_*_*.log"):
        m = RUN_LOG_RE.match(p.name)
        if m:
            logs.append((int(m.group(1)), p))
    logs.sort()
    return [p for _, p in logs]


def parse_run(log_path: Path, max_epoch: int = None):
    """Parse and merge all logs of the run, deduplicating by epoch
    (if an epoch appears in multiple files, the most recent file wins)."""
    keys = ("epoch", "total", "loss", "rec", "l1",
            "zeros", "active", "alive", "dead", "dying", "lmbd")

    logs = find_run_logs(log_path)
    if not logs:
        logs = [log_path]          # fallback: use the file as-is

    merged = {}                    # epoch -> row of values
    rec_label = "rec_loss"
    for p in logs:
        d, rl = parse_log(p, max_epoch=max_epoch)
        if d["epoch"].size:
            rec_label = rl
        for i in range(d["epoch"].size):
            ep = int(d["epoch"][i])
            merged[ep] = tuple(d[k][i] for k in keys)

    data = {k: [] for k in keys}
    for ep in sorted(merged):
        for k, v in zip(keys, merged[ep]):
            data[k].append(v)
    return {k: np.array(v) for k, v in data.items()}, rec_label


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLORS = {
    "loss":   "#1f77b4",
    "rec":    "#ff7f0e",
    "l1":     "#2ca02c",
    "zeros":  "#9467bd",
    "active": "#d62728",
    "alive":  "#17becf",
    "dead":   "#8c1a1a",
    "dying":  "#e08000",
    "lmbd":   "#1f77b4",
}

TITLES = [
    "Loss total",
    "MSE loss",
    "L1 (sparsity)",
    "MSE loss + L1 (juntos)",
    "Zeros (frac.)",
    "Active atoms",
    "Alive atoms",
    "Dead atoms",
    "Dying atoms / lmbd_l1",
]


def make_figure():
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    return fig, axes


def run_title(log_path: Path, rec_label: str) -> str:
    """Derive the run title from the log path.

    Expected structure:  .../6_logs/<loss>/dim_<N>/train_*.log
    Returns e.g. "SSIM dim_512". If the path does not match, falls back to
    the rec_label parsed from the log."""
    loss = None
    dim = None
    for part in log_path.resolve().parts:
        if part.startswith("dim_"):
            dim = part[len("dim_"):]
        elif part in ("mse", "mae", "ssim"):
            loss = part.upper()
    if loss is None:
        # rec_label like "ssim_loss" / "mse_loss" / "rec_loss"
        loss = rec_label.replace("_loss", "").upper()
    dim_str = f" dim_{dim}" if dim else ""
    return f"{loss}{dim_str}"


def draw(fig, axes, d, rec_label, title_suffix="", log_path: Path = None):
    ep = d["epoch"]
    if ep.size == 0:
        print("[Aviso] No se encontraron épocas en el log.", file=sys.stderr)
        return

    configs = [
        (0, [("loss", "Loss total",  COLORS["loss"])],               True),
        (1, [("rec",  rec_label,     COLORS["rec"])],                 True),
        (2, [("l1",   "L1",          COLORS["l1"])],                  True),
        (3, [("rec",  rec_label,     COLORS["rec"]),
             ("l1",   "L1",          COLORS["l1"])],                  True),
        (4, [("zeros",  "zeros",     COLORS["zeros"])],               False),
        (5, [("active", "active",    COLORS["active"])],              False),
        (6, [("alive",  "alive",     COLORS["alive"])],               False),
        (7, [("dead",   "dead",      COLORS["dead"])],                False),
        (8, [("dying",  "dying",     COLORS["dying"]),
             ("lmbd",   "lmbd_l1",   COLORS["lmbd"])],               False),
    ]

    for ax_idx, series, log_scale in configs:
        ax = axes[ax_idx]
        ax.cla()
        ax.set_title(TITLES[ax_idx], fontsize=9, pad=4)
        ax.set_xlabel("Época", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        for key, label, color in series:
            vals = d[key].astype(float)
            if log_scale:
                vals = np.where(vals > 0, vals, np.nan)
            ax.plot(ep, vals, color=color, linewidth=0.9, label=label)

        if log_scale:
            ax.set_yscale("log")
        if len(series) > 1:
            ax.legend(fontsize=7, loc="upper right")

    total = int(d["total"][-1]) if d["total"].size else "?"
    last  = int(ep[-1])
    title = run_title(log_path, rec_label) if log_path is not None else rec_label.replace("_loss", "").upper()
    fig.suptitle(
        f"{title}  —  épocas 1-{last}{title_suffix}",
        fontsize=11, y=0.99,
    )


def save_png(fig, log_path: Path, last_epoch: int) -> Path:
    out = log_path.parent / f"{log_path.stem}_epoch{last_epoch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Monitor de entrenamiento LISTA")
    ap.add_argument("--log",      type=Path, required=True,
                    help="Ruta al fichero de log (.log)")
    ap.add_argument("--epoch",    type=int,  default=None,
                    help="Mostrar hasta esta época (default: todas)")
    ap.add_argument("--live",     action="store_true",
                    help="Modo live: actualizar automáticamente mientras entrena")
    ap.add_argument("--every",    type=int,  default=50,
                    help="[--live] Actualizar cada N épocas nuevas (default: 50)")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="[--live] Segundos entre comprobaciones (default: 3)")
    args = ap.parse_args()

    if args.live:
        _run_live(args)
    else:
        _run_oneshot(args)


def _run_oneshot(args):
    d, rec_label = parse_run(args.log, max_epoch=args.epoch)
    if d["epoch"].size == 0:
        print("No hay datos. Comprueba la ruta del log.", file=sys.stderr)
        sys.exit(1)

    last_epoch = int(d["epoch"][-1])
    fig, axes = make_figure()
    draw(fig, axes, d, rec_label, log_path=args.log)
    out = save_png(fig, args.log, last_epoch)
    print(f"PNG guardado: {out}")
    plt.show()


def _run_live(args):
    paused     = [False]
    last_drawn = [-1]

    fig, axes = make_figure()

    def on_key(event):
        if event.key == "p":
            paused[0] = not paused[0]
            print("  Monitor", "PAUSADO" if paused[0] else "activo")
        elif event.key in ("q", "escape"):
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.ion()
    plt.show()

    print(f"Monitor live  —  log: {args.log}")
    print(f"  p = pausar/reanudar   q = cerrar")
    print(f"  Actualización cada {args.every} épocas\n")

    while plt.get_fignums():
        if not paused[0]:
            d, rec_label = parse_run(args.log)
            if d["epoch"].size > 0:
                last_epoch = int(d["epoch"][-1])
                if last_epoch - last_drawn[0] >= args.every:
                    draw(fig, axes, d, rec_label, log_path=args.log)
                    fig.canvas.draw_idle()
                    last_drawn[0] = last_epoch
                    out = save_png(fig, args.log, last_epoch)
                    print(f"  [refresh] época {last_epoch}  →  {out}")
        plt.pause(args.interval)

    print("Monitor cerrado.")


if __name__ == "__main__":
    main()
