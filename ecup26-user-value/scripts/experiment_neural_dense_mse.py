"""T3.9.8: DENSE daily tokens + GRU + MSE on log1p (NOT ZILN).

The one untried neural variant: dense 90-day calendar tokens (which the sparse
GRU lacked) with the log1p MSE loss (which the ZILN head broke). References:
LGBM fold B 1.6861, sparse-GRU 1.7153, dense-ZILN 2.5788.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_neural_ziln import (  # noqa: E402
    DEVICE,
    N_FEAT,
    N_GLOBAL,
    SEED,
    TARGET_DAYS,
    VAL_FRAC,
    D,
    _build_dense,
    _load,
    _next30_targets,
)

HIDDEN = 64
BATCH = 1024
LR = 3e-4
EPOCHS = 20
PATIENCE = 5
N_ORIGINS = 4


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class GRUNet(nn.Module):
    """Single-output GRU over dense tokens; MSE on log1p target."""

    def __init__(self, n_feat=N_FEAT, n_global=N_GLOBAL, hidden=HIDDEN, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + n_global, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, mask, g):
        B = x.shape[0]
        h = torch.relu(self.proj(x)) * mask.unsqueeze(-1)
        out, _ = self.gru(h)
        lengths = mask.sum(1).long().clamp(min=1)
        last = out[torch.arange(B, device=x.device), lengths - 1]
        pooled = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).float()
        return self.head(torch.cat([last, pooled, g], dim=1)).squeeze(-1)


def _batches(X, M, G, Y, bs):
    n = len(Y)
    perm = np.random.permutation(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        yield (torch.from_numpy(X[idx]).float().to(DEVICE),
               torch.from_numpy(M[idx]).to(DEVICE),
               torch.from_numpy(G[idx]).to(DEVICE),
               torch.from_numpy(Y[idx]).float().to(DEVICE))


def _predict(model, X, M, G):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH * 4):
            xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(DEVICE)
            mb = torch.from_numpy(M[i:i + BATCH * 4]).to(DEVICE)
            gb = torch.from_numpy(G[i:i + BATCH * 4]).to(DEVICE)
            outs.append(model(xb, mb, gb).cpu().numpy())
    return np.concatenate(outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    df, tok = _load(args.data)
    uids = df["user_id"].values
    dates = df["event_date"].values.astype("datetime64[ns]")
    print(f"[data] {len(tok):,} rows, n_feat={N_FEAT}, D={D}, device={DEVICE}")

    max_date = pd.Timestamp(dates.max())
    eval_cutoff = max_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [eval_cutoff - pd.Timedelta(days=TARGET_DAYS + i * 30)
               for i in range(N_ORIGINS)]
    Xs, Ms, Gs, Us, Ys = [], [], [], [], []
    for o in origins:
        seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(o))
        yl = _next30_targets(df, o).reindex(u).fillna(0.0).values.astype(np.float32)
        Xs.append(seqs)
        Ms.append(mask)
        Gs.append(globs)
        Us.append(u)
        Ys.append(yl)
    X = np.concatenate(Xs)
    M = np.concatenate(Ms)
    G = np.concatenate(Gs)
    Y = np.concatenate(Ys)
    del Xs, Ms, Gs, Ys

    all_users = np.unique(np.concatenate(Us))
    rng = np.random.default_rng(SEED)
    val_users = set(rng.choice(all_users, size=int(VAL_FRAC * len(all_users)), replace=False))
    val_mask = np.isin(np.concatenate(Us), np.array(list(val_users)))
    tr = ~val_mask
    print(f"[split] train={tr.sum():,} val={val_mask.sum():,}")

    model = GRUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for xb, mb, gb, yb in _batches(X[tr], M[tr], G[tr], Y[tr], BATCH):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb, mb, gb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        preds = _predict(model, X[val_mask], M[val_mask], G[val_mask])
        val_rmsle = float(np.sqrt(np.mean((preds - Y[val_mask]) ** 2)))
        print(f"  epoch {ep+1}/{args.epochs}: train_mse={np.mean(losses):.4f} "
              f"val_rmsle={val_rmsle:.4f}")
        if val_rmsle < best_val - 1e-5:
            best_val, bad = val_rmsle, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep+1}")
                break
    model.load_state_dict(best_state)

    seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(eval_cutoff))
    pred_log = _predict(model, seqs, mask, globs)
    y_true = _next30_targets(df, eval_cutoff).reindex(u).fillna(0.0).values
    rmsle = float(np.sqrt(np.mean((pred_log - y_true) ** 2)))
    print("=" * 60)
    print(f"[EVAL] dense+MSE fold-B proxy (users={len(u):,}) RMSLE = {rmsle:.4f}")
    print("  refs: LGBM 1.6861 | sparse-GRU 1.7153 | dense-ZILN 2.5788")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
