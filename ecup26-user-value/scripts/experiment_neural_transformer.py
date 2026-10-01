"""T3.9.9: Transformer over DENSE daily tokens -> log1p(30d GMV).

Task hint ("tokenize daily behavior, is it NN time?") + no container/time limit.
Dense daily tokens -> linear proj -> learned positional -> TransformerEncoder
-> masked mean-pool -> head (recency globals) -> MSE on log1p. Long training
with tqdm progress bars + logging + best-model checkpoint (torch.save).
References: LGBM fold B 1.6857, sparse-GRU 1.7153.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("transformer")

DMODEL = 128
NHEADS = 8
NLAYERS = 4
DROPOUT = 0.1
BATCH = 256
LR = 1e-3
EPOCHS = 60
PATIENCE = 10
N_ORIGINS = 8
CKPT = "/tmp/transformer_best.pt"


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class DailyTransformer(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_global=N_GLOBAL, d=DMODEL,
                 nheads=NHEADS, nlayers=NLAYERS, dropout=DROPOUT, max_len=D):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nheads, dim_feedforward=d * 2,
                                         dropout=dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(
            nn.Linear(d + n_global, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, mask, g):
        h = self.proj(x) + self.pos[:, :x.shape[1], :]
        key_mask = (mask == 0)
        out = self.encoder(h, src_key_padding_mask=key_mask)
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([pooled, g], dim=1)).squeeze(-1)


def _batches(X, M, G, Y, idx, bs):
    """Yield batches by global row indices (memmap-safe, no full-copy)."""
    perm = np.random.permutation(len(idx))
    for i in range(0, len(perm), bs):
        gidx = idx[perm[i:i + bs]]
        yield (torch.from_numpy(np.asarray(X[gidx])).float().to(DEVICE),
               torch.from_numpy(np.asarray(M[gidx])).to(DEVICE),
               torch.from_numpy(np.asarray(G[gidx])).to(DEVICE),
               torch.from_numpy(np.asarray(Y[gidx])).float().to(DEVICE))


@torch.no_grad()
def _predict(model, X, M, G, idx=None):
    """Predict for rows (all if idx is None, else by global indices)."""
    model.eval()
    outs = []
    rng_rows = np.arange(len(X))[idx] if idx is not None else np.arange(len(X))
    for i in range(0, len(rng_rows), BATCH * 4):
        gidx = rng_rows[i:i + BATCH * 4]
        xb = torch.from_numpy(np.asarray(X[gidx])).float().to(DEVICE)
        mb = torch.from_numpy(np.asarray(M[gidx])).to(DEVICE)
        gb = torch.from_numpy(np.asarray(G[gidx])).to(DEVICE)
        outs.append(model(xb, mb, gb).cpu().numpy())
    return np.concatenate(outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--origins", type=int, default=N_ORIGINS)
    args = ap.parse_args()

    t0 = time.time()
    logger.info("loading + tokenizing data...")
    df, tok = _load(args.data)
    uids = df["user_id"].values
    dates = df["event_date"].values.astype("datetime64[ns]")
    logger.info("tokenized %d rows (%.0fs), D=%d, n_feat=%d, device=%s",
                len(tok), time.time() - t0, D, N_FEAT, DEVICE)

    max_date = pd.Timestamp(dates.max())
    eval_cutoff = max_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [eval_cutoff - pd.Timedelta(days=TARGET_DAYS + i * 30)
               for i in range(args.origins)]

    logger.info("building eval sequences (fold B cutoff %s)...", eval_cutoff.date())
    eval_seqs, eval_mask, eval_u, eval_g = _build_dense(
        tok, uids, dates, np.datetime64(eval_cutoff))
    eval_y = _next30_targets(df, eval_cutoff).reindex(eval_u).fillna(0.0).values
    logger.info("eval users: %d", len(eval_u))

    # Memory-safe: write sequences to memmap on disk (OOM guard: the full
    # training set does not fit RAM alongside VS Code). Batches read on demand.
    logger.info("building dense sequences for %d origins (memmap)...", len(origins))
    tmp = Path("/tmp/transformer_data")
    tmp.mkdir(exist_ok=True)
    size_est = sum(245_000 for _ in origins)
    Xm = np.memmap(tmp / "X.dat", dtype=np.float16, mode="w+", shape=(size_est, D, N_FEAT))
    Mm = np.memmap(tmp / "M.dat", dtype=np.float32, mode="w+", shape=(size_est, D))
    Gm = np.memmap(tmp / "G.dat", dtype=np.float32, mode="w+", shape=(size_est, N_GLOBAL))
    Ym = np.memmap(tmp / "Y.dat", dtype=np.float32, mode="w+", shape=(size_est,))
    Um = np.memmap(tmp / "U.dat", dtype=np.int64, mode="w+", shape=(size_est,))
    off = 0
    for o in tqdm(origins, desc="origins"):
        seqs, mask, u, globs = _build_dense(tok, uids, dates, np.datetime64(o))
        yl = _next30_targets(df, o).reindex(u).fillna(0.0).values.astype(np.float32)
        n = len(u)
        Xm[off:off + n] = seqs
        Mm[off:off + n] = mask
        Gm[off:off + n] = globs
        Ym[off:off + n] = yl
        Um[off:off + n] = u
        off += n
    Xm.flush()
    Mm.flush()
    Gm.flush()
    Ym.flush()
    Um.flush()
    del tok, df
    n_total = off
    X = Xm[:n_total]
    M = Mm[:n_total]
    G = Gm[:n_total]
    Y = Ym[:n_total]
    U = Um[:n_total]
    logger.info("sequences: %d x %d x %d (memmap on disk, fp16)", n_total, D, N_FEAT)

    all_users = np.unique(np.asarray(U))
    rng = np.random.default_rng(SEED)
    val_users = set(rng.choice(all_users, size=int(VAL_FRAC * len(all_users)), replace=False))
    val_mask = np.isin(np.asarray(U), np.array(list(val_users)))
    tr = ~val_mask
    train_idx = np.where(tr)[0]
    val_idx = np.where(val_mask)[0]
    logger.info("split: train=%d val=%d", len(train_idx), len(val_idx))

    model = DailyTransformer().to(DEVICE)
    logger.info("model: %d params, %d layers, d=%d",
                sum(p.numel() for p in model.parameters()), NLAYERS, DMODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    n_batches = int(np.ceil(len(train_idx) / BATCH))

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        t_ep = time.time()
        model.train()
        losses = []
        bar = tqdm(_batches(X, M, G, Y, train_idx, BATCH), total=n_batches,
                   desc=f"epoch {ep+1}/{args.epochs}", leave=False, unit="b",
                   mininterval=60)  # limit log spam in non-tty
        for xb, mb, gb, yb in bar:
            opt.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                loss = nn.functional.mse_loss(model(xb, mb, gb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        preds = _predict(model, X, M, G, idx=val_idx)
        val_rmsle = float(np.sqrt(np.mean((preds - np.asarray(Y[val_idx])) ** 2)))
        logger.info("epoch %d/%d: train_mse=%.4f val_rmsle=%.4f (%.0fs, lr=%.2e)",
                    ep + 1, args.epochs, float(np.mean(losses)), val_rmsle,
                    time.time() - t_ep, sched.get_last_lr()[0])
        if val_rmsle < best_val - 1e-5:
            best_val, bad = val_rmsle, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CKPT)
            logger.info("  saved best checkpoint (val %.4f) -> %s", best_val, CKPT)
        else:
            bad += 1
            if bad >= PATIENCE:
                logger.info("early stop at epoch %d", ep + 1)
                break
    model.load_state_dict(best_state)

    pred_log = _predict(model, eval_seqs, eval_mask, eval_g)
    rmsle = float(np.sqrt(np.mean((pred_log - eval_y) ** 2)))
    np.save("/tmp/transformer_foldB_preds.npy", np.column_stack([eval_u, pred_log]))
    logger.info("=" * 60)
    logger.info("EVAL transformer fold-B (users=%d) RMSLE = %.4f", len(eval_u), rmsle)
    logger.info("  best val RMSLE = %.4f | refs: LGBM 1.6857 | sparse-GRU 1.7153", best_val)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
