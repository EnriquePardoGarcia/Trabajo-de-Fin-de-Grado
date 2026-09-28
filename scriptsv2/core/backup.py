import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import json
import os
import random
import shutil
import subprocess


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


from core import configuration as cfg
from core.dataset import CellsDataset
from core.loss    import mse_loss, mae_loss, ssim_loss, sparsity_metrics
from core.model   import LinearLISTAEncoder, LinearLISTADecoder



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_checkpoint(path, epoch, encoder, decoder, optimizer,
                    best_rec, hparams, history, test_indices):
    torch.save({
        "epoch":           epoch,
        "encoder_state":   encoder.state_dict(),
        "decoder_state":   decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_rec":        best_rec,
        "hparams":         hparams,
        "history":         history,
        "test_indices":    test_indices,
    }, path)



def load_checkpoint(path, encoder, decoder, optimizer, device):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {path}\n"
            f"Verifica que la ruta sea correcta y el archivo exista."
        )
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    history = ckpt.get("history", {
        "loss": [], "rec": [], "l1": [],
        "frac_zeros": [], "active": [],
        "alive": [], "dead": [], "dying": [],
    })
    return ckpt["epoch"], ckpt["best_rec"], history



def make_split(N: int, train_frac: float, test_frac: float):
    assert abs(train_frac + test_frac - 1.0) < 1e-6, \
        "train_fraction + test_fraction deben sumar 1.0"
    idx = list(range(N))
    random.shuffle(idx)
    n_test    = max(1, round(N * test_frac))
    n_train   = N - n_test
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]
    return train_idx, test_idx



def save_split(path: Path, train_idx, test_idx, hparams: dict):
    data = {
        "config":         hparams["config"],
        "rec_mode":       hparams["rec_mode"],
        "code_dim":       hparams["code_dim"],
        "n_train":        len(train_idx),
        "n_test":         len(test_idx),
        "train_fraction": hparams["train_fraction"],
        "test_fraction":  hparams["test_fraction"],
        "train_indices":  train_idx,
        "test_indices":   test_idx,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Split guardado en: {path}")



def load_split(path: Path):
    with open(path) as f:
        data = json.load(f)
    print(f"Split cargado desde: {path}  "
          f"(train={data['n_train']}  test={data['n_test']})")
    return data["train_indices"], data["test_indices"]



def build_loaders(dataset_train, hparams: dict, device, split_path: Path):
    N = len(dataset_train)
    if split_path.exists():
        train_idx, test_idx = load_split(split_path)
    else:
        train_idx, test_idx = make_split(
            N,
            hparams["train_fraction"],
            hparams["test_fraction"],
        )
        save_split(split_path, train_idx, test_idx, hparams)


    aug_repeat   = hparams.get("aug_repeat", 1)
    repeated_idx = train_idx * aug_repeat


    patch_mb = 64 * 64 * 3 * 4 / 1024**2
    print(f"Train: {len(train_idx):>6} parches  x{aug_repeat} aug_repeat = "
          f"{len(repeated_idx)} ({len(repeated_idx)*patch_mb:.1f} MB)")
    print(f"Test : {len(test_idx):>6} parches  ({len(test_idx)*patch_mb:.1f} MB)  [reservado]")


    train_loader = DataLoader(
        Subset(dataset_train, repeated_idx),
        batch_size  = hparams["batch_size"],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )


    return train_loader, train_idx, test_idx



def train(hparams: dict, cells_dir: Path, resume: str = None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    rec_mode = hparams["rec_mode"]
    code_dim = hparams["code_dim"]


    base_ckpt_dir = cfg.exp_checkpoints_dir(rec_mode, code_dim).resolve()
    best_dir      = cfg.exp_checkpoints_dir(rec_mode, code_dim, "best").resolve()
    last_dir      = cfg.exp_checkpoints_dir(rec_mode, code_dim, "last").resolve()
    jumps_dir     = cfg.exp_checkpoints_dir(rec_mode, code_dim, "jumps").resolve()
    logs_dir      = cfg.exp_logs_dir(rec_mode, code_dim).resolve()


    ckpt_best     = best_dir      / "best.pt"
    ckpt_last     = last_dir      / "last.pt"
    ckpt_current  = base_ckpt_dir / "current.pt"
    ckpt_previous = base_ckpt_dir / "previous.pt"
    split_path    = base_ckpt_dir / "split.json"


    hparams["ckpt_base"] = str(base_ckpt_dir)


    jump_thresh    = hparams.get("jump_thresh", 0.25)
    jump_cooldown  = hparams.get("jump_cooldown", 10)
    prev_buf_epoch = None
    jumped_epochs  = set()
    last_jump_epoch = -999
    jump_events    = []


    dataset_train = CellsDataset(cells_dir, augment=hparams["augment"], flatten=True)
    hparams["dataset_mean"]    = dataset_train.mean
    hparams["dataset_std"]     = dataset_train.std
    hparams["ssim_data_range"] = dataset_train.data_range
    N = len(dataset_train)


    print(f"\n{'='*60}")
    print(f"Dispositivo  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"Memoria      : {props.total_memory / 1024**3:.2f} GB")
    print(f"{'='*60}")
    print(f"Checkpoints  : {base_ckpt_dir}")
    print(f"Logs         : {logs_dir}")
    print(f"Configuración: {hparams['config']}")
    print(f"rec_mode     : {rec_mode}")
    print(f"augmentation : {hparams['augment']}")
    iter_decay_val = hparams.get("iter_decay", cfg.ITER_DECAY)
    print(f"code_dim     : {code_dim}  |  num_iters  : {hparams['num_iters']}  |  iter_decay : {iter_decay_val}")
    print(f"lmbd_rec     : {hparams['lmbd_rec']}  |  lmbd_l1: {hparams['lmbd_l1']}  |  warmup_epochs: {hparams['warmup_epochs']}")
    print(f"lr           : {hparams['lr']}  |  batch_size : {hparams['batch_size']}")
    print(f"num_epochs   : {hparams['num_epochs']}  |  print_every: {hparams['print_every']}")
    print(f"train_frac   : {hparams['train_fraction']}  |  test_frac  : {hparams['test_fraction']}")
    print(f"jump_thresh  : {hparams['jump_thresh']}")
    if rec_mode == "ssim":
        print(f"ssim         : kernel={hparams['ssim_kernel_size']}  sigma={hparams['ssim_sigma']}"
              f"  k1={hparams['ssim_k1']}  k2={hparams['ssim_k2']}"
              f"  data_range={hparams['ssim_data_range']:.4f}")
    else:
        print(f"gauss_sigma  : {hparams['gaussian_sigma']}")
    print(f"Total células: {N}  |  mean={dataset_train.mean:.6f}  "
          f"std={dataset_train.std:.6f}  data_range={dataset_train.data_range:.4f}")
    print(f"{'='*60}\n")


    encoder   = LinearLISTAEncoder(cfg.IN_DIM, code_dim, hparams["num_iters"]).to(device)
    encoder.iter_decay = hparams.get("iter_decay", cfg.ITER_DECAY)
    decoder   = LinearLISTADecoder(encoder).to(device)
    # The decoder has no parameters of its own (weights tied to the encoder); we only use encoder params.
    # S.weight is split out to apply a larger weight_decay to it and avoid the "kill matrix" pathology.
    s_wd = hparams.get("s_weight_decay", hparams["weight_decay"])
    decay_params   = [encoder.W_e.weight]
    s_decay_params = [encoder.S.weight]
    nodecay_params = [encoder.S.bias]
    optimizer = optim.AdamW(
        [{"params": decay_params,   "weight_decay": hparams["weight_decay"]},
         {"params": s_decay_params, "weight_decay": s_wd},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=hparams["lr"],
    )


    all_encoder_params = decay_params + s_decay_params + nodecay_params
    print(f"Parámetros totales : {sum(p.numel() for p in all_encoder_params):,}")
    print(f"  encoder W_e      : {encoder.W_e.weight.numel():,}  (wd={hparams['weight_decay']})")
    print(f"  encoder S.weight : {encoder.S.weight.numel():,}  (wd={s_wd})")
    print(f"  encoder S.bias   : {encoder.S.bias.numel():,}  (wd=0)\n")


    train_loader, train_indices, test_indices = build_loaders(
        dataset_train, hparams, device, split_path
    )
    hparams["split_json"] = str(split_path)


    start_epoch = 1
    best_rec    = float("inf")
    history     = {
        "loss": [], "rec": [], "l1": [],
        "frac_zeros": [], "active": [],
        "alive": [], "dead": [], "dying": [],
    }


    if resume:
        ckpt_resume = Path(resume).resolve()
        if not ckpt_resume.exists():
            raise FileNotFoundError(
                f"Checkpoint de resume no encontrado: {ckpt_resume}\n"
                f"Ruta proporcionada: {resume}\n"
                f"Verifica que el archivo exista en la ruta especificada."
            )
        start_epoch, best_rec, history = load_checkpoint(
            ckpt_resume, encoder, decoder, optimizer, device
        )
        ckpt_raw      = torch.load(ckpt_resume, map_location=device)
        test_indices  = ckpt_raw.get("test_indices", test_indices)
        start_epoch  += 1
        print(f"Reanudando desde epoch {start_epoch}  (best_rec={best_rec:.6f})")
        if ckpt_current.exists():
            prev_buf_epoch = start_epoch - 1


    data_range    = hparams["ssim_data_range"]
    lmbd_rec      = hparams.get("lmbd_rec", 1.0)
    lmbd_l1       = hparams.get("lmbd_l1", cfg.LMBD_L1)
    l1_ramp_epochs = hparams.get("l1_ramp_epochs", cfg.L1_RAMP_EPOCHS)
    warmup_epochs = hparams.get("warmup_epochs", cfg.WARMUP_EPOCHS)
    topk_warmup   = hparams.get("topk_warmup")  # None/0 -> ReLU from the start
    if topk_warmup is None:
        topk_warmup = int(code_dim * 0.10)
        print(f"[Info] topk_warmup no especificado; usando default = code_dim × 10% = {topk_warmup} átomos.")
    elif topk_warmup == 0:
        topk_warmup = None  # explicit 0 disables TopK

    # Post-warmup top-k: linear decay from topk_warmup to topk_final
    # topk_final=None -> code_dim x 5% (default); topk_final=0 -> ReLU (legacy)
    _topk_final_cfg = hparams.get("topk_final", cfg.TOPK_FINAL)
    topk_decay_epochs = int(hparams.get("topk_decay_epochs", cfg.TOPK_DECAY_EPOCHS))
    if _topk_final_cfg is None and topk_warmup is not None:
        topk_final = max(1, int(code_dim * 0.05))
    elif _topk_final_cfg == 0:
        topk_final = None  # ReLU post-warmup (legacy behavior)
    else:
        topk_final = _topk_final_cfg

    if topk_warmup is not None and topk_final is not None:
        print(f"topk schedule : {topk_warmup} → {topk_final} lineal en {topk_decay_epochs} epochs post-warmup, luego constante")
    elif topk_warmup is not None:
        print(f"topk schedule : {topk_warmup} (warmup) → ReLU post-warmup (legacy)")

    if l1_ramp_epochs > 0:
        l1_step = lmbd_l1 / l1_ramp_epochs
        print(f"l1 ramp       : 0.0 → {lmbd_l1} lineal en {l1_ramp_epochs} epochs post-warmup  (step={l1_step:.6f}/epoch)")
    else:
        print(f"l1 ramp       : desactivada — lmbd_l1={lmbd_l1} fijo desde epoch {warmup_epochs + 1}")

    if rec_mode == "mse":
        loss_fn   = lambda x, xh, z, lmbd: mse_loss(x, xh, z, lmbd, lmbd_rec, data_range)
        rec_label = "mse_loss"
    elif rec_mode == "mae":
        loss_fn   = lambda x, xh, z, lmbd: mae_loss(x, xh, z, lmbd, lmbd_rec, data_range)
        rec_label = "mae_loss"
    else:
        loss_fn   = lambda x, xh, z, lmbd: ssim_loss(x, xh, z, lmbd, lmbd_rec, data_range)
        rec_label = "ssim_loss"


    alive_window_epochs = hparams.get("alive_window", 50)
    n_batches_per_epoch = len(train_loader)
    ALIVE_WINDOW        = alive_window_epochs * n_batches_per_epoch
    atom_activity        = torch.zeros(code_dim, device=device)
    warmup_reinit_every  = hparams.get("warmup_reinit_every", cfg.WARMUP_REINIT_EVERY)
    freeze_s_warmup      = hparams.get("freeze_s_warmup", cfg.FREEZE_S_WARMUP)
    atom_activated_short = torch.zeros(code_dim, dtype=torch.bool, device=device)
    print(f"alive_window : {alive_window_epochs} epochs × {n_batches_per_epoch} batches/epoch = {ALIVE_WINDOW} batches")

    for epoch in range(start_epoch, hparams["num_epochs"] + 1):
        if epoch <= warmup_epochs:
            encoder.topk = topk_warmup
        elif topk_final is None:
            encoder.topk = None  # ReLU (legacy)
        else:
            post = epoch - warmup_epochs
            if post >= topk_decay_epochs:
                encoder.topk = topk_final
            else:
                frac = post / topk_decay_epochs
                encoder.topk = max(topk_final,
                                   round(topk_warmup + (topk_final - topk_warmup) * frac))
        encoder.train()
        decoder.train()

        if epoch <= warmup_epochs:
            _lmbd = 0.0
        elif l1_ramp_epochs > 0:
            _lmbd = lmbd_l1 * min(1.0, (epoch - warmup_epochs) / l1_ramp_epochs)
        else:
            _lmbd = lmbd_l1

        acc           = {"loss": 0., "rec": 0., "l1": 0., "fz": 0., "act": 0.}
        total_samples = 0
        n_batches     = 0

        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            B  = xb.size(0)

            z     = encoder(xb)
            x_hat = decoder(z)
            loss, rec, sparse = loss_fn(xb, x_hat, z, _lmbd)

            with torch.no_grad():
                active_mask          = (z > 0).any(dim=0).float()
                atom_activity        = atom_activity * (ALIVE_WINDOW - 1) / ALIVE_WINDOW + active_mask / ALIVE_WINDOW
                atom_activated_short |= active_mask.bool()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc["loss"] += loss.item()   * B
            acc["rec"]  += rec.item()    * B
            acc["l1"]   += sparse.item() * B
            fz, act = sparsity_metrics(z)
            acc["fz"]  += fz
            acc["act"] += act
            total_samples += B
            n_batches     += 1

        avg_loss = acc["loss"] / total_samples
        avg_rec  = acc["rec"]  / total_samples
        avg_l1   = acc["l1"]   / total_samples
        avg_fz   = acc["fz"]   / n_batches
        avg_act  = acc["act"]  / n_batches


        DEAD_THR = 1.0 / ALIVE_WINDOW  # threshold: atom active < 1 time per window -> considered dead
        n_alive = int((atom_activity > DEAD_THR).sum().item())
        n_dead  = code_dim - n_alive
        n_dying = int(((atom_activity > DEAD_THR) & (atom_activity < 0.1)).sum().item())

        # Post-warmup revival of dead atoms: every alive_window_epochs, atoms that
        # have been inactive for the whole window are reinitialized.
        # NOTE: W_e is NOT reinitialized because the decoder uses W_e^T; replacing
        # rows of W_e destroys the reconstruction (loss spike). Only S (which does
        # not affect the decoder) and the activity counter are reset.
        revival_every = hparams.get("revival_every", alive_window_epochs)
        if (epoch > warmup_epochs and revival_every > 0
                and epoch % revival_every == 0 and n_dead > 0):
            dead_idx = (atom_activity <= DEAD_THR).nonzero(as_tuple=True)[0]
            n_actual  = dead_idx.numel()
            with torch.no_grad():
                encoder.S.weight.data[dead_idx, :] = 0.0
                encoder.S.weight.data[:, dead_idx] = 0.0
                encoder.S.bias.data[dead_idx]       = 0.0
                atom_activity[dead_idx]             = DEAD_THR * 2
            print(f"  [REVIVAL] {n_actual} átomos muertos reactivados (S)")

        # Reinitialization of dead atoms during warmup
        n_inverted  = 0
        n_dead_short = int((~atom_activated_short).sum().item()) if (warmup_reinit_every > 0 or freeze_s_warmup) else 0
        if warmup_reinit_every > 0 and epoch <= warmup_epochs and epoch % warmup_reinit_every == 0:
            dead_short = (~atom_activated_short).nonzero(as_tuple=True)[0]
            n_inverted = dead_short.numel()
            if n_inverted > 0:
                with torch.no_grad():
                    # Only reset S: W_e is left untouched because the decoder uses W_e^T,
                    # and reinitializing W_e during warmup destabilizes the reconstruction.
                    encoder.S.weight.data[dead_short, :] = 0.0
                    encoder.S.weight.data[:, dead_short] = 0.0
                    encoder.S.bias.data[dead_short]      = 0.0
                    atom_activity[dead_short]            = DEAD_THR * 2
                print(f"  [WARMUP REINIT] {n_inverted} átomos reinicializados (S)")
            atom_activated_short.zero_()

        if freeze_s_warmup and epoch <= warmup_epochs and epoch % 5 == 0:
            atom_activated_short.zero_()

        if freeze_s_warmup and epoch <= warmup_epochs:
            with torch.no_grad():
                encoder.S.weight.zero_()
                encoder.S.bias.zero_()

        history["loss"].append(avg_loss)
        history["rec"].append(avg_rec)
        history["l1"].append(avg_l1)
        history["frac_zeros"].append(avg_fz)
        history["active"].append(avg_act)
        history["alive"].append(n_alive)
        history["dead"].append(n_dead)
        history["dying"].append(n_dying)


        is_best = avg_rec < best_rec
        if is_best:
            best_rec = avg_rec


        if prev_buf_epoch is not None and ckpt_previous.exists():
            rec_prev = history["rec"][-2] if len(history["rec"]) >= 2 else None
            act_prev = history["active"][-2] if len(history["active"]) >= 2 else None


            cooldown_ok = (epoch - last_jump_epoch) > jump_cooldown
            if rec_prev is not None and act_prev is not None and prev_buf_epoch not in jumped_epochs and cooldown_ok:
                rec_change = (avg_rec - rec_prev) / (rec_prev + 1e-10)
                act_change = (avg_act - act_prev) / (act_prev + 1e-10) if act_prev > 0 else 0.0


                rec_jump = abs(rec_change) > jump_thresh
                sparse_jump = abs(act_change) > jump_thresh


                if rec_jump or sparse_jump:
                    suffix = "sparse" if (rec_jump and sparse_jump) else ("up" if rec_change > 0 else "down")
                    jump_dir  = jumps_dir / f"jump_{suffix}_{prev_buf_epoch}"
                    jump_path = jump_dir / "checkpoint.pt"


                    if not jump_dir.exists():
                        jump_dir.mkdir(parents=True, exist_ok=True)
                        prev_ckpt = torch.load(ckpt_previous, map_location="cpu")


                        meta = {
                            "epoch_before": prev_buf_epoch,
                            "epoch_after": epoch,
                            "rec_before": float(rec_prev),
                            "rec_after": float(avg_rec),
                            "active_before": float(act_prev),
                            "active_after": float(avg_act),
                            "rec_change_rel": float(rec_change),
                            "act_change_rel": float(act_change),
                            "direction": "up" if rec_change > 0 else "down",
                            "affects_rec": bool(rec_jump),
                            "affects_sparse": bool(sparse_jump),
                            "suffix": suffix,
                            "dirname": jump_dir.name,
                        }


                        prev_ckpt["jump_meta"] = meta
                        torch.save(prev_ckpt, jump_path)
                        jumped_epochs.add(prev_buf_epoch)
                        last_jump_epoch = epoch
                        jump_events.append(meta)
                        print(
                            f"  [JUMP] {jump_dir.name}  "
                            f"rec: {rec_prev:.4f}→{avg_rec:.4f}  "
                            f"act: {act_prev:.1f}→{avg_act:.1f}"
                        )


        save_checkpoint(ckpt_last, epoch, encoder, decoder, optimizer,
                        best_rec, hparams, history, test_indices)


        if ckpt_current.exists():
            shutil.copy2(ckpt_current, ckpt_previous)
        save_checkpoint(ckpt_current, epoch, encoder, decoder, optimizer,
                        best_rec, hparams, history, test_indices)
        prev_buf_epoch = epoch


        marker = " * best" if is_best else ""
        print(
            f"Epoch {epoch:>6d}/{hparams['num_epochs']} | "
            f"loss={avg_loss:.6f}  {rec_label}={avg_rec:.6f}  l1={avg_l1:.6f} | "
            f"zeros={avg_fz:.3f}  active={avg_act:.1f}  "
            f"alive={n_alive}  dead={n_dead}  dying={n_dying}"
            f"{marker}  lmbd_l1={_lmbd:.6f}"
        )
        sample_batch = next(iter(train_loader))[0][:256].to(device)
        layer_acts   = encoder.layer_activity(sample_batch)
        print("  [layers] " + "  ".join(f"i{i+1}={v:.1f}" for i, v in enumerate(layer_acts)))


        if is_best:
            save_checkpoint(ckpt_best, epoch, encoder, decoder, optimizer,
                            best_rec, hparams, history, test_indices)


    print("\nEntrenamiento finalizado.")
    print(f"Checkpoints en : {base_ckpt_dir}")


    history_path = logs_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Historial en   : {history_path}")


    jump_events_path = logs_dir / "jump_events.json"
    with open(jump_events_path, "w") as f:
        json.dump(jump_events, f, indent=2)
    print(f"Jump events en : {jump_events_path}")


    print(f"\nPara evaluar   : python evaluation/evaluate.py --rec_mode {rec_mode} --code_dim {code_dim}")
    print(f"Para visualizar: python visualization/visualize.py --rec_mode {rec_mode} --code_dim {code_dim}")


    return history, encoder, decoder



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Entrena LISTA Sparse Autoencoder. "
            "Los valores por defecto vienen de core/configuration.py; "
            "cualquier argumento CLI los sobreescribe."
        )
    )
    p.add_argument("--config",         type=str, required=True,
                   choices=list(cfg.CONFIG_MAP.keys()),
                   help="mse_no_aug | mse_aug | mae_no_aug | mae_aug | ssim_no_aug | ssim_aug")
    p.add_argument("--cells_dir",      type=str,   default=None)
    p.add_argument("--num_epochs",     type=int,   default=cfg.NUM_EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--lmbd_rec",       type=float, default=cfg.LMBD_REC)
    p.add_argument("--lr",             type=float, default=cfg.LR)
    p.add_argument("--weight_decay",   type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--code_dim",       type=int,   default=cfg.CODE_DIM,
                   help="Número de átomos (ej: 128, 1024, 2048).")
    p.add_argument("--jump_window",    type=int,   default=20)
    p.add_argument("--jump_thresh",    type=float, default=0.25)
    p.add_argument("--num_iters",      type=int,   default=cfg.NUM_ITERS)
    p.add_argument("--train_fraction", type=float, default=cfg.TRAIN_FRACTION)
    p.add_argument("--test_fraction",  type=float, default=cfg.TEST_FRACTION)
    p.add_argument("--print_every",    type=int,   default=cfg.PRINT_EVERY)
    p.add_argument("--aug_repeat",     type=int,   default=cfg.AUG_REPEAT)
    p.add_argument("--resume",         type=str,   default=None,
                   help="Ruta al checkpoint .pt del que reanudar.")
    p.add_argument("--alive_window",        type=int,   default=cfg.ALIVE_WINDOW_EPOCHS,
                   help="Ventana en epochs para calcular átomos vivos.")
    p.add_argument("--warmup_epochs",       type=int,   default=cfg.WARMUP_EPOCHS,
                   help="Épocas con S congelado y λ=0 antes de activar el controlador (default: 500).")
    p.add_argument("--freeze_s_warmup",     action="store_true", default=cfg.FREEZE_S_WARMUP,
                   help="Mantener S=0 durante el calentamiento.")
    p.add_argument("--warmup_reinit_every", type=int,   default=cfg.WARMUP_REINIT_EVERY,
                   help="Reinicializar átomos inactivos cada N épocas durante warmup (0 = desactivado).")
    p.add_argument("--revival_every",       type=int,   default=cfg.REVIVAL_EVERY,
                   help="Reinicializar átomos muertos cada N épocas post-warmup (0 = desactivado).")
    p.add_argument("--s_weight_decay",      type=float, default=cfg.S_WEIGHT_DECAY,
                   help="Weight decay específico para S.weight (mayor que weight_decay evita kill matrix).")
    p.add_argument("--lmbd_l1",             type=float, default=cfg.LMBD_L1,
                   help="Peso fijo de la penalización L1 post-warmup (0 durante warmup).")
    p.add_argument("--l1_ramp_epochs",      type=int,   default=cfg.L1_RAMP_EPOCHS,
                   help="Épocas post-warmup para subir lmbd_l1 linealmente de 0 a su valor final (0 = sin rampa).")
    p.add_argument("--topk_warmup",         type=int,   default=None,
                   help="Átomos activos forzados durante el warmup con TopK (default: code_dim × 10%%). "
                        "0 = desactivar TopK y usar ReLU desde el inicio.")
    p.add_argument("--topk_final",          type=int,   default=None,
                   help="TopK objetivo post-warmup (default: code_dim × 5%%). "
                        "0 = cambiar a ReLU tras el warmup (legacy).")
    p.add_argument("--topk_decay_epochs",   type=int,   default=cfg.TOPK_DECAY_EPOCHS,
                   help="Épocas post-warmup para decaer linealmente de topk_warmup a topk_final.")
    p.add_argument("--iter_decay",          type=float, default=cfg.ITER_DECAY,
                   help="Factor de decay del topk por iteración LISTA (ej: 0.85 → i1=topk, i2=0.85·topk, i3=0.72·topk). "
                        "1.0 = sin decay (comportamiento por defecto).")
    return p.parse_args()



def main():
    args      = parse_args()

    if args.resume:
        project_root = Path(__file__).resolve().parent.parent.parent
        resume_path = Path(args.resume)

        # Relative path is resolved from the project root
        if not resume_path.is_absolute():
            abs_resume = str(project_root / resume_path)
        else:
            abs_resume = str(resume_path.resolve())
            
        for i in range(len(sys.argv) - 1):
            if sys.argv[i] == "--resume" and sys.argv[i + 1] == args.resume:
                sys.argv[i + 1] = abs_resume
                break
        args.resume = abs_resume

    cfg_flags = cfg.CONFIG_MAP[args.config]
    rec_mode  = cfg_flags["rec_mode"]


    if "LISTA_TRAINING" not in os.environ:
        logs_dir = cfg.exp_logs_dir(rec_mode, args.code_dim)


        if args.resume:
            try:
                ckpt_raw      = torch.load(Path(args.resume), map_location="cpu")
                resume_epoch  = ckpt_raw.get("epoch", None)
                new_start     = resume_epoch + 1 if resume_epoch is not None else None
                new_end       = args.num_epochs
                log_path = (logs_dir / f"train_{new_start}_{new_end}.log"
                            if new_start is not None else logs_dir / "train.log")
            except Exception as e:
                print(f"[Aviso] No se pudo leer el checkpoint de resume: {e}")
                log_path = logs_dir / "train.log"
        else:
            log_path = logs_dir / f"train_1_{args.num_epochs}.log"


        env = os.environ.copy()
        env["LISTA_TRAINING"] = "1"


        with open(log_path, "w", buffering=1) as log_file:
            subprocess.Popen(
                ["nohup", sys.executable, "-u"] + sys.argv,
                stdout     = log_file,
                stderr     = log_file,
                stdin      = subprocess.DEVNULL,
                env        = env,
                preexec_fn = os.setpgrp,
            )


        ckpt_base = cfg.exp_checkpoints_dir(rec_mode, args.code_dim)
        print(f"Checkpoints : {ckpt_base}")
        print(f"Log         : {log_path}")
        print(f"\nSeguir progreso:")
        print(f"  tail -f {log_path}")
        return


    total_frac = args.train_fraction + args.test_fraction
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train_fraction + test_fraction = {total_frac:.4f}, debe ser 1.0"
        )


    cells_dir = Path(args.cells_dir) if args.cells_dir else cfg.CELLS_DIR
    if not cells_dir.exists():
        raise FileNotFoundError(
            f"No se encontró cells/: {cells_dir}\nUsa --cells_dir para especificar la ruta."
        )


    hparams = {
        "config":           args.config,
        "rec_mode":         rec_mode,
        "augment":          cfg_flags["augment"],
        "code_dim":         args.code_dim,
        "jump_window":      args.jump_window,
        "jump_thresh":      args.jump_thresh,
        "num_iters":        args.num_iters,
        "lmbd_rec":         args.lmbd_rec,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "num_epochs":       args.num_epochs,
        "batch_size":       args.batch_size,
        "train_fraction":   args.train_fraction,
        "test_fraction":    args.test_fraction,
        "print_every":      args.print_every,
        "aug_repeat":       args.aug_repeat,
        "alive_window":        args.alive_window,
        "warmup_epochs":       args.warmup_epochs,
        "warmup_reinit_every": args.warmup_reinit_every,
        "revival_every":       args.revival_every,
        "l1_ramp_epochs":      args.l1_ramp_epochs,
        "freeze_s_warmup":     args.freeze_s_warmup,
        "s_weight_decay":      args.s_weight_decay,
        "lmbd_l1":             args.lmbd_l1,
        "topk_warmup":         args.topk_warmup,
        "topk_final":          args.topk_final,
        "topk_decay_epochs":   args.topk_decay_epochs,
        "iter_decay":          args.iter_decay,
        "gaussian_sigma":   cfg.GAUSSIAN_SIGMA,
        "ssim_kernel_size": cfg.SSIM_KERNEL_SIZE,
        "ssim_sigma":       cfg.SSIM_SIGMA,
        "ssim_k1":          cfg.SSIM_K1,
        "ssim_k2":          cfg.SSIM_K2,
        "ssim_data_range":  None,
    }


    train(hparams, cells_dir, resume=args.resume)



if __name__ == "__main__":
    main()