"""T3.10.24: Multitask GRU - shared encoder + 2 heads (P(y>0) + regression).

The one neural variant never tried: a SHARED encoder trained on BOTH a binary
P(y>0) task and a log1p regression task (hard parameter sharing). All prior
neural runs were single-headed and rejected (dense-ZILN 2.58, dense+MSE 1.7262,
sparse-GRU 1.7153, transformer). The shared representation may regularize the
zero-mass / whale-tail structure better than a single MSE head.

Gate: fold-B proxy RMSLE < 1.70, else close the ticket (neural path exhausted,
confidence 0.15 per ticket). Reference: LGBM fold B ~1.686, dense+MSE 1.7262.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
GATE = 1.70


class MultitaskGRU(nn.Module):
    """Shared GRU encoder + binary P(y>0) head + log1p regression head."""

    def __init__(self, n_feat=N_FEAT, n_global=N_GLOBAL, hidden=HIDDEN, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, 1, batch_first=True)
        rep = hidden * 2 + n_global
        self.clf_head = nn.Sequential(
            nn.Linear(rep, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.reg_head = nn.Sequential(
            nn.Linear(rep, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def _repr(self, x, mask, g):
        B = x.shape[0]
        h = torch.relu(self.proj(x)) * mask.unsqueeze(-1)
        out, _ = self.gru(h)
        lengths = mask.sum(1).long().clamp(min=1)
        last = out[torch.arange(B, device=x.device), lengths - 1]
        pooled = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1).float()
        return torch.cat([last, pooled, g], dim=1)

    def forward(self, x, mask, g):
        rep = self._repr(x, mask, g)
        return self.reg_head(rep).squeeze(-1), self.clf_head(rep).squeeze(-1)

    def predict(self, x, mask, g):
        reg, clf = self.forward(x, mask, g)
        return reg, torch.sigmoid(clf)


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best, self.bad, self.state = float("inf"), 0, None

    def step(self, val, model):
        if val < self.best - 1e-5:
            self.best, self.bad = val, 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience


def _batches(X, M, G, Yb, Yc, bs):
    n = len(Yb)
    perm = np.random.permutation(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        yield (torch.from_numpy(X[idx]).float().to(DEVICE),
               torch.from_numpy(M[idx]).to(DEVICE),
               torch.from_numpy(G[idx]).to(DEVICE),
               torch.from_numpy(Yb[idx]).float().to(DEVICE),
               torch.from_numpy(Yc[idx]).float().to(DEVICE))


def _predict(model, X, M, G, with_clf=False):
    model.eval()
    regs, clfs = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH * 4):
            xb = torch.from_numpy(X[i:i + BATCH * 4]).float().to(DEVICE)
            mb = torch.from_numpy(M[i:i + BATCH * 4]).to(DEVICE)
            gb = torch.from_numpy(G[i:i + BATCH * 4]).to(DEVICE)
            r, c = model.predict(xb, mb, gb)
            regs.append(r.cpu().numpy())
            clfs.append(c.cpu().numpy())
    regs = np.concatenate(regs)
    if with_clf:
        return regs, np.concatenate(clfs)
    return regs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    df, tok = _load(args.data)
    uids = df["user_id"].values
    dates = df["event_date"].values.astype("datetime64[ns]")
    print(f"[data] {len(tok):,} rows, n_feat={N_FEAT}, D={D}, device={DEVICE}")

    max_date = pd.Timestamp(dates.max())
    eval_cutoff = max_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [eval_cutoff - pd.Timedelta(days=TARGET_DAYS + i * 30)
               for i in range(N_ORIGINS)]
    Xs, Ms, Gs, Us, Yb, Yc = [], [], [], [], [], []
    for o in origins:
        seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(o))
        yl = _next30_targets(df, o).reindex(u).fillna(0.0).values.astype(np.float32)
        Xs.append(seqs)
        Ms.append(mask)
        Gs.append(globs)
        Us.append(u)
        Yb.append(yl)
        Yc.append((yl > 0).astype(np.float32))
    X = np.concatenate(Xs)
    M = np.concatenate(Ms)
    G = np.concatenate(Gs)
    Yb = np.concatenate(Yb)
    Yc = np.concatenate(Yc)
    del Xs, Ms, Gs

    all_users = np.unique(np.concatenate(Us))
    rng = np.random.default_rng(SEED)
    val_users = set(rng.choice(all_users, size=int(VAL_FRAC * len(all_users)), replace=False))
    val_mask = np.isin(np.concatenate(Us), np.array(list(val_users)))
    tr = ~val_mask
    print(f"[split] train={tr.sum():,} val={val_mask.sum():,} zero_frac={float((Yb == 0).mean()):.3f}")

    model = MultitaskGRU().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    es = EarlyStopping(PATIENCE)
    for ep in range(args.epochs):
        model.train()
        losses, clf_l, reg_l = [], [], []
        for xb, mb, gb, yb, yc in _batches(X[tr], M[tr], G[tr], Yb[tr], Yc[tr], BATCH):
            opt.zero_grad()
            reg, clf = model(xb, mb, gb)
            loss_reg = F.mse_loss(reg, yb)
            loss_clf = F.binary_cross_entropy_with_logits(clf, yc)
            loss = loss_reg + loss_clf
            loss.backward()
            opt.step()
            losses.append(loss.item())
            reg_l.append(loss_reg.item())
            clf_l.append(loss_clf.item())
        sched.step()
        preds = _predict(model, X[val_mask], M[val_mask], G[val_mask])
        val_rmsle = float(np.sqrt(np.mean((preds - Yb[val_mask]) ** 2)))
        print(f"  epoch {ep+1}/{args.epochs}: loss={np.mean(losses):.4f} "
              f"(reg {np.mean(reg_l):.4f} / clf {np.mean(clf_l):.4f}) "
              f"val_rmsle={val_rmsle:.4f}")
        if es.step(val_rmsle, model):
            print(f"  early stop at epoch {ep+1}")
            break
    model.load_state_dict(es.state)

    seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(eval_cutoff))
    pred_log, p_pos = _predict(model, seqs, mask, globs, with_clf=True)
    y_true = _next30_targets(df, eval_cutoff).reindex(u).fillna(0.0).values
    rmsle = float(np.sqrt(np.mean((pred_log - y_true) ** 2)))
    rmsle_hurdle = float(np.sqrt(np.mean((pred_log * p_pos - y_true) ** 2)))
    verdict = "PASS" if rmsle < GATE else "FAIL (gate 1.70)"
    print("=" * 60)
    print(f"[EVAL] multitask-GRU fold-B proxy RMSLE = {rmsle:.4f}  (gate {GATE:.2f}) -> {verdict}")
    print(f"[EVAL] hurdle variant (reg * P(y>0))      RMSLE = {rmsle_hurdle:.4f}")
    print("  refs: LGBM ~1.686 | dense+MSE 1.7262 | sparse-GRU 1.7153 | dense-ZILN 2.5788")
    print("=" * 60)
    return 0 if rmsle < GATE else 1


if __name__ == "__main__":
    sys.exit(main())
