"""Parser for training log files and plotter of metrics from a .log file."""

import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|"
    r"\s+loss=([\d.]+)"
    r"\s+\w+loss=([\d.]+)"
    r"\s+l1=([\d.]+)"
    r"\s+\|"
    r"\s+zeros=([\d.]+)"
    r"\s+active=([\d.]+)"
    r"(?:\s+alive=(\d+))?"
    r"(?:\s+dead=(\d+))?"
    r"(?:\s+dying=(\d+))?"
    r"(?:(?:\s+lmbd_cur=|\s+lmbd_l1=|\s+lmbd=)([\d.e+\-]+))?"
    r"(?:\s+l0_err=([+\-\d.]+))?"
)
_LAYERS_RE = re.compile(r"\[layers\]((?:\s+i\d+=[\d.]+)+)")


def _parse_log(path: Path) -> dict:
    data = {k: [] for k in [
        "epoch", "loss", "rec", "l1", "zeros", "active",
        "alive", "dead", "dying", "lmbd", "l0_err", "best",
        "layer_i1", "layer_last",
    ]}
    pending = False
    with open(path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m:
                data["epoch"].append(int(m.group(1)))
                data["loss"].append(float(m.group(2)))
                data["rec"].append(float(m.group(3)))
                data["l1"].append(float(m.group(4)))
                data["zeros"].append(float(m.group(5)))
                data["active"].append(float(m.group(6)))
                data["alive"].append(int(m.group(7))   if m.group(7)  else 0)
                data["dead"].append(int(m.group(8))    if m.group(8)  else 0)
                data["dying"].append(int(m.group(9))   if m.group(9)  else 0)
                data["lmbd"].append(float(m.group(10)) if m.group(10) else float("nan"))
                data["l0_err"].append(float(m.group(11)) if m.group(11) else float("nan"))
                data["best"].append("* best" in line)
                data["layer_i1"].append(float("nan"))
                data["layer_last"].append(float("nan"))
                pending = True
                continue
            if pending:
                lm = _LAYERS_RE.search(line)
                if lm:
                    vals = [float(v) for v in re.findall(r"i\d+=([\d.]+)", lm.group(1))]
                    if vals:
                        data["layer_i1"][-1]   = vals[0]
                        data["layer_last"][-1] = vals[-1]
                pending = False
    return {k: np.array(v) for k, v in data.items()}


def _draw_log(fig: plt.Figure, data: dict, log_path: Path, skip_first: int = 1):
    fig.clear()
    sl = slice(skip_first, None)
    ep = data["epoch"][sl]
    if len(ep) == 0:
        return

    has_l0  = not np.all(np.isnan(data["l0_err"]))
    has_lay = not np.all(np.isnan(data["layer_i1"]))

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    def logplot(ax, ep, y, **kw):
        ax.semilogy(ep, np.where(y > 0, y, np.nan), **kw)

    ax = fig.add_subplot(gs[0, 0])
    logplot(ax, ep, data["loss"][sl], lw=0.8, color="steelblue")
    ax.set_title("Loss total (log)"); ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[0, 1])
    logplot(ax, ep, data["rec"][sl], lw=0.8, color="darkorange")
    ax.set_title("Rec loss (log)"); ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[0, 2])
    logplot(ax, ep, data["rec"][sl], lw=0.8, color="darkorange", label="rec")
    logplot(ax, ep, data["l1"][sl],  lw=0.8, color="seagreen",   label="L1")
    ax.set_title("Rec vs L1 (log)"); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ep, data["active"][sl], lw=0.8, color="crimson")
    ax.set_title("Active atoms/sample"); ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ep, data["alive"][sl], lw=0.8, color="green", label="alive")
    ax.plot(ep, data["dead"][sl],  lw=0.8, color="red",   label="dead")
    ax.set_title("Alive / Dead"); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ep, data["dying"][sl], lw=0.8, color="peru")
    ax.set_title("Dying"); ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(ep, data["lmbd"][sl], lw=0.8, color="purple")
    ax.set_title("λ"); ax.set_xlabel("Epoch")
    if np.any(data["lmbd"][sl] > 0):
        try: ax.set_yscale("log")
        except Exception: pass

    if has_l0:
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(ep, data["l0_err"][sl], lw=0.8, color="navy")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title("l0 error"); ax.set_xlabel("Epoch")

    if has_lay:
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(ep, data["layer_i1"][sl],   lw=0.8, color="teal",   label="i1 (W_e)")
        ax.plot(ep, data["layer_last"][sl], lw=0.8, color="orange", label="iN (final)")
        ax.set_title("Layer activity"); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)

    fig.suptitle(f"{log_path.name}  —  {len(ep)} epochs", fontsize=11)


def run_log_mode(log_path: Path, out_path: Path | None, watch: int):
    matplotlib.use("Agg")
    if out_path is None:
        out_path = log_path.with_suffix(".png")
    fig = plt.figure(figsize=(16, 11))
    while True:
        data = _parse_log(log_path)
        _draw_log(fig, data, log_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {out_path}  ({len(data['epoch'])} epochs)")
        if watch <= 0:
            break
        time.sleep(watch)
        fig.clear()
    plt.close(fig)
