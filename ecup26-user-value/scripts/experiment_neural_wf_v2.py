#!/usr/bin/env python3
"""T3.19 v2: two-input neural model (sequence + 102 static features), fold-B validation.

Same walk-forward setup as experiment_neural_wf.py, plus:
- static feature cache (data/static/static_{origin}.npz) as a second input
- --norm flag: log1p heavy-tailed columns + StandardScaler (fit on train origins)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neural_models import SequenceModelStatic  # noqa: E402

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
ORIGINS = ["2025-06-18", "2025-07-18", "2025-08-17", "2025-09-16",
           "2025-10-16", "2025-11-15", "2025-12-15"]


def _load_seq(path: str, window: int):
    z = np.load(Path(path))
    X = z["X"][:, -window:, :].astype(np.float32)
    X[:, :, LOG1P_IDX] = np.log1p(X[:, :, LOG1P_IDX])
    return X, z["y"].astype(np.float32), z["user_ids"]


def _load_static(path: str, seq_users):
    z = np.load(Path(path))
    sdf = pd.DataFrame(z["X"], index=z["user_ids"])
    return sdf.reindex(pd.Index(seq_users)).fillna(0.0).to_numpy(dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--window", type=int, default=409)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--norm", type=int, default=0)
    ap.add_argument("--static-dim", type=int, default=102)
    ap.add_argument("--save-thresh", type=float, default=1.80)
    args = ap.parse_args()
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    static_dim = int(np.load("data/static/static_2025-06-18.npz")["X"].shape[1])

    train_stat = []
    for o in ORIGINS:
        sq = f"data/sequences/seq_{o}_L{args.window}.npz"
        sp = f"data/static/static_{o}.npz"
        zseq = np.load(sq)
        train_stat.append(_load_static(sp, zseq["user_ids"]))
    all_train = np.concatenate(train_stat, axis=0)
    heavy = np.zeros(all_train.shape[1], dtype=bool)
    scaler = None
    if args.norm:
        heavy = all_train.max(axis=0) > 50.0
        all_train[:, heavy] = np.log1p(all_train[:, heavy])
        scaler = StandardScaler().fit(all_train)

    def transform(s):
        s = s.copy()
        if args.norm:
            s[:, heavy] = np.log1p(s[:, heavy])
            s = scaler.transform(s).astype(np.float32)
        return s

    del train_stat, all_train

    Xv, yv, uv = _load_seq(f"data/sequences/seq_B_L{args.window}.npz", args.window)
    Sv = transform(_load_static("data/static/static_B.npz", uv))
    X_va, S_va, y_va = torch.tensor(Xv), torch.tensor(Sv), torch.tensor(yv)
    del Xv, Sv, yv, uv
    print(f"val {len(X_va):,} | {args.model} h{args.hidden} l{args.layers} "
          f"lr{args.lr} w{args.window} norm={args.norm} static={static_dim}", flush=True)

    model = SequenceModelStatic(d_in=16, hidden=args.hidden, layers=args.layers,
                                kind=args.model, static_dim=static_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    bs, best, best_ep = args.batch, 1e9, 0

    for epoch in range(args.epochs):
        model.train()
        tot, n_seen = 0.0, 0
        for o in ORIGINS:
            sq = f"data/sequences/seq_{o}_L{args.window}.npz"
            sp = f"data/static/static_{o}.npz"
            Xo, yo, uo = _load_seq(sq, args.window)
            So = transform(_load_static(sp, uo))
            X_o, S_o, y_o = torch.tensor(Xo), torch.tensor(So), torch.tensor(yo)
            del Xo, So, yo, uo
            perm = torch.randperm(len(X_o))
            for i in range(0, len(X_o), bs):
                b = perm[i:i + bs]
                xb, sb, yb = X_o[b].to(dev), S_o[b].to(dev), y_o[b].to(dev)
                opt.zero_grad()
                loss = lossf(model(xb, sb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            n_seen += len(X_o)
            del X_o, S_o, y_o
        sched.step()
        model.eval()
        with torch.no_grad():
            ps = torch.cat([model(X_va[i:i + 512].to(dev), S_va[i:i + 512].to(dev))
                            for i in range(0, len(X_va), 512)])
            rmsle = float(torch.sqrt(((ps.cpu() - y_va) ** 2).mean()))
        if rmsle < best:
            best, best_ep = rmsle, epoch + 1
        print(f"epoch {epoch + 1:2d}: train_mse={tot / n_seen:.4f} foldB_rmsle={rmsle:.4f}", flush=True)

    print(f"RESULT {args.model} h{args.hidden} l{args.layers} lr{args.lr} w{args.window} "
          f"norm={args.norm}: BEST fold-B {best:.4f} @ ep {best_ep} (16-feat baseline 1.6702)")
    if best < args.save_thresh:
        save_dir = Path("models/neural")
        save_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{args.model}_h{args.hidden}_l{args.layers}_lr{args.lr}_w{args.window}"
                f"_st{static_dim}_n{args.norm}_r{best:.4f}")
        torch.save(model.state_dict(), save_dir / f"{name}.pt")
        (save_dir / f"{name}.json").write_text(json.dumps({
            "model": args.model, "hidden": args.hidden, "layers": args.layers,
            "lr": args.lr, "window": args.window, "static_dim": static_dim,
            "norm": args.norm, "best_rmsle": best, "best_epoch": best_ep,
            "heavy_cols": heavy.tolist(),
        }, indent=2))
        print(f"SAVED models/neural/{name}.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
