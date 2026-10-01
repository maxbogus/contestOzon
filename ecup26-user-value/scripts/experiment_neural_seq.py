"""T3.8.14: neural sequence baseline over tokenized daily user behavior.

The task statement hints ("проверить, не пришло ли время нейросетей") that
tokenizing daily behavior + a neural sequence model may beat the GBM plateau.

Design:
- Tokenize each user's ACTIVE days into a fixed-length sequence (last 40):
  6 activity flags + 7 log1p counts + 3 log1p GMVs + day-of-week + log1p gap.
- Small GRU (1 layer, hidden 64), aggregation = last-valid-hidden + mean-pool,
  plus sequence globals (recency, n_active, has_activity).
- Target: log1p(30d GMV) -> MSE in log-space == minimizing RMSLE.
- Walk-forward: 4 origins (Dec15/Nov15/Oct16/Sep16 2025), user-level 90/10 split.
- Eval: users with activity <= 2026-01-14 (fold-B proxy), truth = Jan14-Feb13 GMV.

References: LGBM fold B 1.6881, LGBM CV 1.7155.

Dev-only experiment (GPU, torch in dev group; not part of the submission).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SEQ_LEN = 40
HIDDEN = 64
N_LAYERS = 1
DROPOUT = 0.1
BATCH = 1024
LR = 3e-4
EPOCHS = 15
PATIENCE = 4
TARGET_DAYS = 30
N_ORIGINS = 4
VAL_FRAC = 0.1

FLAG_COLS = ["search", "cat", "has_search_to_cart", "has_search_to_ord",
             "has_cat_to_cart", "has_cat_to_ord"]
COUNT_COLS = ["searches", "search_to_cart", "search_to_ord", "cat_to_cart",
              "cat_to_ord", "to_cart", "to_ord"]
GMV_COLS = ["gmv_search", "gmv_cat", "gmv"]
N_FEAT = len(FLAG_COLS) + len(COUNT_COLS) + len(GMV_COLS) + 2  # + dow, + gap
N_GLOBAL = 3  # recency_log, n_active_log, has_activity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_and_tokenize(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load + sort rows; token matrix (N, N_FEAT) of per-day features."""
    cols = ["user_id", "event_date"] + FLAG_COLS + COUNT_COLS + GMV_COLS
    df = pd.read_parquet(path, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(["user_id", "event_date"]).reset_index(drop=True)
    tok = np.zeros((len(df), N_FEAT), dtype=np.float32)
    off = 0
    for i, c in enumerate(FLAG_COLS):
        tok[:, off + i] = df[c].astype(np.float32).values
    off += len(FLAG_COLS)
    for i, c in enumerate(COUNT_COLS):
        tok[:, off + i] = np.log1p(df[c].values.astype(np.float64)).astype(np.float32)
    off += len(COUNT_COLS)
    for i, c in enumerate(GMV_COLS):
        tok[:, off + i] = np.log1p(df[c].values.astype(np.float64)).astype(np.float32)
    off += len(GMV_COLS)
    tok[:, off] = (df["event_date"].dt.dayofweek / 7.0).astype(np.float32).values
    prev = df.groupby("user_id")["event_date"].shift(1)
    gap = (df["event_date"] - prev).dt.days.fillna(0).astype(np.float64)
    tok[:, off + 1] = np.log1p(np.maximum(gap, 0)).astype(np.float32).values
    return df, tok


def _build_sequences(
    tok: np.ndarray, uids: np.ndarray, dates: np.ndarray, cutoff: np.datetime64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last-`SEQ_LEN` active days per user <= cutoff. seqs, mask, uids, globals."""
    sel = dates <= cutoff
    s_tok, s_uid = tok[sel], uids[sel]
    u, starts = np.unique(s_uid, return_index=True)
    ends = np.append(starts[1:], len(s_uid))
    n = len(u)
    seqs = np.zeros((n, SEQ_LEN, N_FEAT), dtype=np.float32)
    mask = np.zeros((n, SEQ_LEN), dtype=np.float32)
    globs = np.zeros((n, N_GLOBAL), dtype=np.float32)
    last_act = dates[sel][ends - 1]
    cutoff_ts = cutoff.astype("datetime64[D]")
    for i in range(n):
        lo, hi = starts[i], ends[i]
        n_act = int(hi - lo)
        take = min(n_act, SEQ_LEN)
        seg = s_tok[hi - take:hi]
        seqs[i, SEQ_LEN - take:] = seg
        mask[i, SEQ_LEN - take:] = 1.0
        recency = max(int((cutoff_ts - last_act[i]).astype("int64")), 0)
        globs[i, 0] = np.log1p(recency)
        globs[i, 1] = np.log1p(n_act)
        globs[i, 2] = 1.0 if n_act > 0 else 0.0
    return seqs, mask, u, globs


def _next30_targets(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    d = df[
        (df["event_date"] > cutoff)
        & (df["event_date"] <= cutoff + pd.Timedelta(days=TARGET_DAYS))
    ]
    return np.log1p(d.groupby("user_id")["gmv"].sum())


class GRUNet(nn.Module):
    """GRU over daily tokens; last-hidden + mean-pool + globals -> log1p(30d GMV)."""

    def __init__(self, n_feat: int = N_FEAT, n_global: int = N_GLOBAL,
                 hidden: int = HIDDEN, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + n_global, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
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
        yield (torch.from_numpy(X[idx]).to(DEVICE),
               torch.from_numpy(M[idx]).to(DEVICE),
               torch.from_numpy(G[idx]).to(DEVICE),
               torch.from_numpy(Y[idx]).to(DEVICE))



def _train(model, X, M, G, Y, Xv, Mv, Gv, Yv, epochs):
    """Train with MSE(log1p) == RMSLE minimization; early stop on val."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, mb, gb, yb in _batches(X, M, G, Y, BATCH):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb, mb, gb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            val_errs = []
            for xb, mb, gb, yb in _batches(Xv, Mv, Gv, Yv, BATCH * 4):
                val_errs.append(nn.functional.mse_loss(model(xb, mb, gb), yb).item())
        val_rmsle = float(np.sqrt(np.mean(val_errs)))
        print(f"  epoch {ep+1}/{epochs}: train_mse={np.mean(losses):.4f} "
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
    return model, best_val


def _predict(model, X, M, G) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH * 4):
            xb = torch.from_numpy(X[i:i + BATCH * 4]).to(DEVICE)
            mb = torch.from_numpy(M[i:i + BATCH * 4]).to(DEVICE)
            gb = torch.from_numpy(G[i:i + BATCH * 4]).to(DEVICE)
            outs.append(model(xb, mb, gb).cpu().numpy())
    return np.concatenate(outs)




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--origins", type=int, default=N_ORIGINS)
    args = ap.parse_args()

    t0 = time.time()
    df, tok = _load_and_tokenize(args.data)
    uids = df["user_id"].values
    dates = df["event_date"].values.astype("datetime64[ns]")
    print(f"[data] {len(tok):,} rows tokenized ({time.time()-t0:.1f}s), "
          f"n_feat={N_FEAT}, device={DEVICE}")

    # Fold-B proxy: cutoff = max - 30 = 2026-01-14; origins step back 30d.
    max_date = pd.Timestamp(dates.max())
    eval_cutoff = max_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [eval_cutoff - pd.Timedelta(days=TARGET_DAYS + i * 30)
               for i in range(args.origins)]
    print(f"[origins] eval_cutoff={eval_cutoff.date()}: "
          f"{[str(o.date()) for o in origins]}")

    Xs, Ms, Gs, Us, Ys = [], [], [], [], []
    for o in origins:
        seqs, mask, u, globs = _build_sequences(tok, uids, dates, np.datetime64(o))
        y = _next30_targets(df, o).reindex(u).fillna(0.0).values.astype(np.float32)
        Xs.append(seqs)
        Ms.append(mask)
        Gs.append(globs)
        Us.append(u)
        Ys.append(y)
        print(f"  origin {o.date()}: {len(u):,} users")

    X = np.concatenate(Xs)
    M = np.concatenate(Ms)
    G = np.concatenate(Gs)
    Y = np.concatenate(Ys)
    del Xs, Ms, Gs, Ys

    # user-level train/val split (no same-user leakage into val)
    all_users = np.unique(np.concatenate(Us))
    rng = np.random.default_rng(SEED)
    val_users = set(rng.choice(all_users, size=int(VAL_FRAC * len(all_users)), replace=False))
    val_mask = np.isin(np.concatenate(Us), np.array(list(val_users)))
    tr = ~val_mask
    print(f"[split] train={tr.sum():,} val={val_mask.sum():,}")

    model = GRUNet().to(DEVICE)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} params")
    model, val_rmsle = _train(model, X[tr], M[tr], G[tr], Y[tr],
                              X[val_mask], M[val_mask], G[val_mask], Y[val_mask],
                              args.epochs)
    print(f"[train] best val RMSLE={val_rmsle:.4f} ({time.time()-t0:.0f}s total)")

    # --- eval on fold-B proxy: users with activity <= eval_cutoff ---
    seqs, mask, u, globs = _build_sequences(tok, uids, dates, np.datetime64(eval_cutoff))
    pred_log = _predict(model, seqs, mask, globs)
    y_true = _next30_targets(df, eval_cutoff).reindex(u).fillna(0.0).values
    rmsle = float(np.sqrt(np.mean((pred_log - y_true) ** 2)))
    np.save("/tmp/gru_foldB_preds.npy", np.column_stack([u, pred_log]))
    print("=" * 60)
    print(f"[EVAL] fold-B proxy (users={len(u):,}) RMSLE = {rmsle:.4f}")
    print("  reference LGBM fold B: 1.6881 | LGBM CV mean: 1.7155")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
