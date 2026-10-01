"""T3.9.6: neural ZILN over DENSE daily tokens (task hint: tokenize daily behavior).

Upgrades the sparse-active-days GRU:
1. DENSE daily sequence: one token per calendar day for the last D=90 days,
   zero-token + active-flag=0 for inactive days -> the model sees the absolute
   calendar (weekday, holiday periods 23.02 / 08.03) that sparse tokens lost.
2. ZILN head: P(buy) + lognormal(mu, sigma) -> explicit zero-mass + positive
   amount. Prediction = P(buy) * E[lognormal].

Target: log1p(30d GMV). Walk-forward 4 origins, user-level split. Eval on the
fold-B proxy. References: LGBM fold B 1.6874, sparse-GRU 1.7153.
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

D = 90                 # dense window (calendar days)
HIDDEN = 64
N_LAYERS = 1
DROPOUT = 0.1
BATCH = 1024
LR = 3e-4
EPOCHS = 20
PATIENCE = 5
TARGET_DAYS = 30
N_ORIGINS = 4
VAL_FRAC = 0.1

FLAG_COLS = ["search", "cat", "has_search_to_cart", "has_search_to_ord",
             "has_cat_to_cart", "has_cat_to_ord"]
COUNT_COLS = ["searches", "search_to_cart", "search_to_ord", "cat_to_cart",
              "cat_to_ord", "to_cart", "to_ord"]
GMV_COLS = ["gmv_search", "gmv_cat", "gmv"]
# per-day token: dow + active + flags + log1p counts + log1p gmvs
N_FEAT = 2 + len(FLAG_COLS) + len(COUNT_COLS) + len(GMV_COLS)
N_GLOBAL = 2  # n_active_log, has_activity
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    cols = ["user_id", "event_date"] + FLAG_COLS + COUNT_COLS + GMV_COLS
    df = pd.read_parquet(path, columns=cols)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(["user_id", "event_date"]).reset_index(drop=True)
    tok = np.zeros((len(df), N_FEAT), dtype=np.float32)
    tok[:, 0] = 1.0                     # active flag (scattered rows are active)
    tok[:, 1] = (df["event_date"].dt.dayofweek / 7.0).astype(np.float32).values
    off = 2
    for i, c in enumerate(FLAG_COLS):
        tok[:, off + i] = df[c].astype(np.float32).values
    off += len(FLAG_COLS)
    for i, c in enumerate(COUNT_COLS):
        tok[:, off + i] = np.log1p(df[c].values.astype(np.float64)).astype(np.float32)
    off += len(COUNT_COLS)
    for i, c in enumerate(GMV_COLS):
        tok[:, off + i] = np.log1p(df[c].values.astype(np.float64)).astype(np.float32)
    return df, tok


def _build_dense(
    tok: np.ndarray, uids: np.ndarray, dates: np.ndarray, cutoff: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense daily sequences: last D calendar days per user <= cutoff.

    Returns seqs (n_users, D, N_FEAT), mask (n_users, D), user ids, globals.
    """
    lo = cutoff - np.timedelta64(D - 1, "D")
    sel = (dates > lo) & (dates <= cutoff)
    s_tok, s_uid, s_date = tok[sel], uids[sel], dates[sel]
    u, starts = np.unique(s_uid, return_index=True)
    n = len(u)
    # global row -> (user position, slot in window)
    ui = np.searchsorted(u, s_uid)
    day_off = ((cutoff - s_date) / np.timedelta64(1, "D")).astype(np.int64)
    slot = D - 1 - day_off
    seqs = np.zeros((n, D, N_FEAT), dtype=np.float16)  # fp16: half RAM (OOM guard)
    mask = np.zeros((n, D), dtype=np.float32)
    # pre-fill dow for every slot (same calendar for all users)
    dow = pd.date_range(end=pd.Timestamp(cutoff), periods=D).dayofweek.values
    seqs[:, :, 1] = (dow / 7.0).astype(np.float32)[None, :]
    seqs[ui, slot, :] = s_tok.astype(np.float16)  # scatter activity tokens
    mask[ui, slot] = 1.0
    # globals per user
    n_active = np.bincount(ui, minlength=n).astype(np.float32)
    globs = np.zeros((n, N_GLOBAL), dtype=np.float32)
    globs[:, 0] = np.log1p(n_active)
    globs[:, 1] = (n_active > 0).astype(np.float32)
    return seqs, mask, u, globs


def _next30_targets(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    d = df[
        (df["event_date"] > cutoff)
        & (df["event_date"] <= cutoff + pd.Timedelta(days=TARGET_DAYS))
    ]
    return np.log1p(d.groupby("user_id")["gmv"].sum())


class ZILNGRU(nn.Module):
    """GRU over dense daily tokens + ZILN head (P(buy) + lognormal params)."""

    def __init__(self, n_feat: int = N_FEAT, n_global: int = N_GLOBAL,
                 hidden: int = HIDDEN, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + n_global, 32), nn.ReLU(), nn.Dropout(dropout))
        self.head_logit_p = nn.Linear(32, 1)
        self.head_mu = nn.Linear(32, 1)
        self.head_log_sigma = nn.Linear(32, 1)

    def forward(self, x, mask, g):
        B = x.shape[0]
        h = torch.relu(self.proj(x)) * mask.unsqueeze(-1)
        out, _ = self.gru(h)
        lengths = mask.sum(1).long().clamp(min=1)
        last = out[torch.arange(B, device=x.device), lengths - 1]
        pooled = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).float()
        f = self.fuse(torch.cat([last, pooled, g], dim=1))
        logit_p = self.head_logit_p(f).squeeze(-1)
        mu = self.head_mu(f).squeeze(-1)
        log_sigma = self.head_log_sigma(f).squeeze(-1).clamp(min=-5.0, max=3.0)
        return logit_p, mu, log_sigma


def ziln_loss(logit_p, mu, log_sigma, y):
    """Zero-inflated lognormal NLL. y is raw (not log) 30d GMV."""
    p_buy = torch.sigmoid(logit_p)
    sigma = torch.exp(log_sigma)
    eps = 1e-6
    zero = y <= 1e-4
    lognorm_nll = (
        0.5 * np.log(2 * np.pi) + torch.log(y.clamp(min=eps))
        + log_sigma + 0.5 * ((torch.log(y.clamp(min=eps)) - mu) / sigma) ** 2
    )
    nll = torch.where(zero, -torch.log(1 - p_buy + eps), -torch.log(p_buy + eps) + lognorm_nll)
    return nll.mean()


def _predict_ziln(model, X, M, G) -> np.ndarray:
    """Expected value P(buy)*E[lognormal] -> log1p space for RMSLE."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH * 4):
            xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(DEVICE)
            mb = torch.from_numpy(M[i:i + BATCH * 4]).to(DEVICE)
            gb = torch.from_numpy(G[i:i + BATCH * 4]).to(DEVICE)
            logit_p, mu, log_sigma = model(xb, mb, gb)
            p_buy = torch.sigmoid(logit_p)
            sigma = torch.exp(log_sigma)
            pred_raw = p_buy * torch.exp(mu + 0.5 * sigma ** 2)
            outs.append(torch.log1p(pred_raw).cpu().numpy())
    return np.concatenate(outs)




def _next30_raw(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    d = df[
        (df["event_date"] > cutoff)
        & (df["event_date"] <= cutoff + pd.Timedelta(days=TARGET_DAYS))
    ]
    return d.groupby("user_id")["gmv"].sum()


def _batches(X, M, G, Y, bs):
    n = len(Y)
    perm = np.random.permutation(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        yield (torch.from_numpy(X[idx]).float().to(DEVICE),
               torch.from_numpy(M[idx]).to(DEVICE),
               torch.from_numpy(G[idx]).to(DEVICE),
               torch.from_numpy(Y[idx]).to(DEVICE))


def _train(model, X, M, G, Y, Xv, Mv, Gv, Yv_log, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, mb, gb, yb in _batches(X, M, G, Y, BATCH):
            opt.zero_grad()
            logit_p, mu, log_sigma = model(xb, mb, gb)
            loss = ziln_loss(logit_p, mu, log_sigma, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = _predict_ziln(model, Xv, Mv, Gv)
        val_rmsle = float(np.sqrt(np.mean((preds - Yv_log) ** 2)))
        print(f"  epoch {ep+1}/{epochs}: train_nll={np.mean(losses):.4f} "
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--origins", type=int, default=N_ORIGINS)
    args = ap.parse_args()

    t0 = time.time()
    df, tok = _load(args.data)
    uids = df["user_id"].values
    dates = df["event_date"].values.astype("datetime64[ns]")
    print(f"[data] {len(tok):,} rows tokenized ({time.time()-t0:.0f}s), "
          f"n_feat={N_FEAT}, D={D}, device={DEVICE}")

    max_date = pd.Timestamp(dates.max())
    eval_cutoff = max_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [eval_cutoff - pd.Timedelta(days=TARGET_DAYS + i * 30)
               for i in range(args.origins)]
    print(f"[origins] eval_cutoff={eval_cutoff.date()}: "
          f"{[str(o.date()) for o in origins]}")

    Xs, Ms, Gs, Us, Ys = [], [], [], [], []
    for o in origins:
        seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(o))
        yr = _next30_raw(df, o).reindex(u).fillna(0.0).values.astype(np.float32)
        Xs.append(seqs)
        Ms.append(mask)
        Gs.append(globs)
        Us.append(u)
        Ys.append(yr)
        print(f"  origin {o.date()}: {len(u):,} users")

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

    # val targets in log1p space for the RMSLE monitor
    Yv_log = np.log1p(Y[val_mask])

    model = ZILNGRU().to(DEVICE)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} params (ZILN)")
    model, val_rmsle = _train(model, X[tr], M[tr], G[tr], Y[tr],
                              X[val_mask], M[val_mask], G[val_mask], Yv_log,
                              args.epochs)
    print(f"[train] best val RMSLE={val_rmsle:.4f} ({time.time()-t0:.0f}s total)")

    # eval on fold-B proxy
    seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(eval_cutoff))
    pred_log = _predict_ziln(model, seqs, mask, globs)
    y_true = _next30_targets(df, eval_cutoff).reindex(u).fillna(0.0).values
    rmsle = float(np.sqrt(np.mean((pred_log - y_true) ** 2)))
    print("=" * 60)
    print(f"[EVAL] fold-B proxy (users={len(u):,}) RMSLE = {rmsle:.4f}")
    print("  references: LGBM fold B 1.6874 | sparse-GRU 1.7153")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
