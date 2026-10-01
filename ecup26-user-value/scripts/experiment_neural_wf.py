#!/usr/bin/env python3
"""T3.19/E2: parameterized sequence model on walk-forward origins, fold-B validation.

Train on origins (2025-06-18..2025-12-15), validate on fold-B (ref=2026-01-14).
Config-driven: --model {gru,lstm,transformer,patched,mamba,rwkv,xlstm} --hidden
--layers --lr --window --epochs --batch. Data on CPU, batches to GPU (8GB RTX 5060).

T3.20 (2026-08-30): + mamba/rwkv/xlstm (pure-torch self-implemented blocks, d_in=16).

Usage: uv run python scripts/experiment_neural_wf.py --model mamba --hidden 128 \
       --layers 2 --lr 5e-4 --window 409 --epochs 20 --patience 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neural_models import (  # noqa: E402
    SequenceModel,
    SequenceModelMamba,
    SequenceModelPatched,
    SequenceModelRWKV,
    SequenceModelXLSTM,
)

# T3.20: models whose recurrent scan retains O(L*B*D*N) autograd state need a
# smaller batch on the 8GB RTX 5060 (Mamba: D*N state; xLSTM: H*hh*hh matrix).
SCAN_BATCH = {"mamba": 128, "xlstm": 128}

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
ORIGINS = [
    "2025-06-18",
    "2025-07-18",
    "2025-08-17",
    "2025-09-16",
    "2025-10-16",
    "2025-11-15",
    "2025-12-15",
]


def _load(path: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path(path))
    L = z["X"].shape[1]
    if window > L:
        raise ValueError(f"window={window} > cache L={L} in {path} - rebuild the cache")
    X = z["X"][:, -window:, :].astype(np.float32)
    X[:, :, LOG1P_IDX] = np.log1p(X[:, :, LOG1P_IDX])
    return X, z["y"].astype(np.float32)


def load_origins(window: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all origins (window must match the cache: seq_*_L{window}.npz)."""
    Xs, ys = [], []
    for o in ORIGINS:
        path = f"data/sequences/seq_{o}_L{window}.npz"
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {path} - build it for window={window}")
        X, y = _load(path, window)
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def early_stop_after(epoch: int, best_ep: int, patience: int) -> bool:
    """True if `epoch` (1-based) is >= `patience` epochs past the best so far.

    patience <= 0 disables early stopping (backward compat: GRU/LSTM sweeps run
    a fixed number of epochs). Used to cut the overtrain observed on the patched
    model (best @ ep4, val degraded to ep15 while train_mse kept dropping).
    """
    if patience <= 0:
        return False
    return epoch - best_ep >= patience


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=str,
        default="gru",
        help="gru|lstm|transformer|patched|mamba|rwkv|xlstm",
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument(
        "--patch", type=int, default=7, help="patch size for model=patched (days per token)"
    )
    ap.add_argument(
        "--expand",
        type=int,
        default=1,
        help="Mamba inner expansion (1=128, 2=256; 1 fits the 8GB card)",
    )
    ap.add_argument(
        "--n-heads", type=int, default=4, help="xLSTM head count (128/4 -> head dim 32)"
    )
    ap.add_argument(
        "--compile",
        type=int,
        default=1,
        help="torch.compile for mamba/rwkv/xlstm (fuses elementwise scans, ~2-4x faster)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early stop if no fold-B improvement for this many epochs (0 = disabled)",
    )
    ap.add_argument(
        "--save-thresh",
        type=float,
        default=1.6851,
        help="save the model if best fold-B RMSLE < this",
    )
    args = ap.parse_args()
    torch.manual_seed(42)
    # TF32 tensor cores for fp32 matmuls (warning-free, ~1.3x faster on RTX)
    torch.set_float32_matmul_precision("high")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val_path = f"data/sequences/seq_B_L{args.window}.npz"
    if not Path(val_path).exists():
        raise FileNotFoundError(f"missing {val_path}")
    Xv, yv = _load(val_path, args.window)
    print(
        f"val (fold-B): {len(Xv):,} seq x {Xv.shape[1]} | model={args.model} "
        f"h{args.hidden} l{args.layers} lr{args.lr} w{args.window} (origins streamed)"
    )
    X_va, y_va = torch.from_numpy(Xv), torch.from_numpy(yv)
    del Xv

    if args.model == "patched":
        model = SequenceModelPatched(
            d_in=16, hidden=args.hidden, layers=args.layers, patch=args.patch
        ).to(dev)
    elif args.model == "mamba":
        model = SequenceModelMamba(
            d_in=16, hidden=args.hidden, layers=args.layers, expand=args.expand
        ).to(dev)
    elif args.model == "rwkv":
        model = SequenceModelRWKV(d_in=16, hidden=args.hidden, layers=args.layers).to(dev)
    elif args.model == "xlstm":
        model = SequenceModelXLSTM(
            d_in=16, hidden=args.hidden, layers=args.layers, n_heads=args.n_heads
        ).to(dev)
    else:
        model = SequenceModel(d_in=16, hidden=args.hidden, layers=args.layers, kind=args.model).to(
            dev
        )
    if args.compile and args.model in ("mamba", "rwkv", "xlstm"):
        model = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    if args.batch == 512 and args.model in SCAN_BATCH:
        args.batch = SCAN_BATCH[args.model]
    bs, best, best_ep = args.batch, 1e9, 0
    for epoch in range(args.epochs):
        model.train()
        tot, n_seen = 0.0, 0
        for _oi, o in enumerate(ORIGINS):
            path = f"data/sequences/seq_{o}_L{args.window}.npz"
            if not Path(path).exists():
                raise FileNotFoundError(f"missing {path}")
            Xo, yo = _load(path, args.window)
            # zero-copy view (numpy array and torch tensor share memory) — the
            # old torch.tensor() doubled the ~6.5GB per-origin footprint and,
            # together with the resident fold-B tensor, OOM'd the 30GB machine
            # while VSCode was open (2026-08-30 incident).
            X_o, y_o = torch.from_numpy(Xo), torch.from_numpy(yo)
            del Xo, yo
            perm = torch.randperm(len(X_o))
            # compiled models: keep every batch exactly `bs` wide (a partial
            # final batch would re-trigger CUDA-graph capture, ~1-2 min each);
            # GRU/LSTM/patched keep the full set (reproducibility preserved).
            n_full = (
                len(X_o)
                if args.model not in ("mamba", "rwkv", "xlstm")
                else len(X_o) - (len(X_o) % bs)
            )
            for i in range(0, n_full, bs):
                b = perm[i : i + bs]
                xb, yb = X_o[b].to(dev), y_o[b].to(dev)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            n_seen += len(X_o)
            del X_o, y_o
        sched.step()
        model.eval()
        with torch.no_grad():
            parts = []
            for i in range(0, len(X_va), bs):
                xb = X_va[i : i + bs]
                if len(xb) < bs:
                    xb = torch.cat([xb, xb[: bs - len(xb)]], dim=0)  # pad to full bs
                # .clone() detaches the CUDA-graph output buffer (torch.compile
                # reduce-overhead reuses it across replays; accumulating the
                # aliased slices and torch.cat'ing them would read overwritten
                # data -> RuntimeError).
                parts.append(model(xb.to(dev)).clone()[: len(X_va) - i])
            ps = torch.cat(parts)
            rmsle = float(torch.sqrt(((ps.cpu() - y_va) ** 2).mean()))
        if rmsle < best:
            best, best_ep = rmsle, epoch + 1
        print(
            f"epoch {epoch + 1:2d}: train_mse={tot / n_seen:.4f} foldB_rmsle={rmsle:.4f}",
            flush=True,
        )
        if early_stop_after(epoch + 1, best_ep, args.patience):
            print(
                f"EARLY STOP @ ep {epoch + 1} (best {best:.4f} @ ep {best_ep}, "
                f"patience {args.patience})",
                flush=True,
            )
            break
    print(
        f"RESULT {args.model} h{args.hidden} l{args.layers} lr{args.lr} w{args.window}: "
        f"BEST fold-B {best:.4f} @ ep {best_ep} (tabular 1.6813)"
    )
    if best < args.save_thresh:
        save_dir = Path("models/neural")
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.model == "patched":
            name = (
                f"{args.model}_p{args.patch}_h{args.hidden}_l{args.layers}"
                f"_lr{args.lr}_w{args.window}_r{best:.4f}"
            )
        else:
            name = (
                f"{args.model}_h{args.hidden}_l{args.layers}_lr{args.lr}_w{args.window}_r{best:.4f}"
            )
        # save the raw module weights (a torch.compile'd wrapper would add an
        # "_orig_mod." prefix to every key and break downstream load_state_dict)
        raw = getattr(model, "_orig_mod", model)
        torch.save(raw.state_dict(), save_dir / f"{name}.pt")
        import json

        cfg = {
            "model": args.model,
            "hidden": args.hidden,
            "layers": args.layers,
            "lr": args.lr,
            "window": args.window,
            "epochs": args.epochs,
            "best_rmsle": best,
            "best_epoch": best_ep,
        }
        if args.model == "patched":
            cfg["patch"] = args.patch
        (save_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2))
        print(f"SAVED models/neural/{name}.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
