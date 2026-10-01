#!/usr/bin/env python3
"""T3.19/E2: GRU + attention-pooling on daily sequences.

Main model: Embedding(16->64), GRU(2 x hidden 64, causal), attention pooling
(learnable query over timesteps), MLP head -> log1p(GMV_30d). MSE(log1p).
Trained on the fold-B-origin sequence cache (seq_B_L180.npz), 80/20 split,
only batches to the GPU (8GB RTX 5060).

Usage: uv run python scripts/experiment_neural_gru.py [--window 180]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def load(window: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path("data/sequences/seq_B_L180.npz"))
    X = z["X"][:, -window:, :].astype(np.float32)
    y = z["y"].astype(np.float32)
    X[:, :, LOG1P_IDX] = np.log1p(X[:, :, LOG1P_IDX])
    return X, y


class GRUModel(nn.Module):
    def __init__(self, d_in: int = 16, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.emb = nn.Linear(d_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.query = nn.Parameter(torch.randn(hidden) * 0.05)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.emb(x))                 # [B, L, H]
        out, _ = self.gru(h)                        # [B, L, H]
        w = F.softmax(torch.einsum("blh,h->bl", out, self.query), dim=1)
        pooled = torch.einsum("bl,blh->bh", w, out)  # attention-weighted mean
        return self.head(pooled).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = load(args.window)
    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_tr = int(n * 0.8)
    X_tr = torch.tensor(X[idx[:n_tr]])  # keep on CPU
    y_tr = torch.tensor(y[idx[:n_tr]])
    X_va = torch.tensor(X[idx[n_tr:]])
    y_va = torch.tensor(y[idx[n_tr:]])
    print(f"device={dev} | train {n_tr:,} / val {n - n_tr:,} | L={args.window} d=16 "
          f"(data on CPU, batches to GPU - VRAM budget 8GB)")

    model = GRUModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()
    bs = 256
    best, best_ep = 1e9, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr)
        tot = 0.0
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            xb, yb = X_tr[b].to(dev), y_tr[b].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            ps = torch.cat([model(X_va[i : i + 512].to(dev)) for i in range(0, len(X_va), 512)])
            rmsle = float(torch.sqrt(((ps.cpu() - y_va) ** 2).mean()))
        if rmsle < best:
            best, best_ep = rmsle, epoch + 1
        print(f"epoch {epoch + 1:2d}: train_mse={tot / n_tr:.4f} val_rmsle={rmsle:.4f}")
    print(f"BEST val RMSLE = {best:.4f} @ epoch {best_ep} (MLP 1.7794 | tabular ~1.68 | random ~1.9)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
