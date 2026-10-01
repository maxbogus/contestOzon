#!/usr/bin/env python3
"""T3.19/E1: sanity check - flattened-MLP on the daily sequence.

If even a flat MLP on the last 90 days (flattened) cannot approach the tabular
fold-B (~1.68), the neural direction is dead (cheap kill). If val RMSLE is well
below random (~1.9-2.0), we proceed to GRU.

Uses the fold-B-origin sequence cache (seq_B_L180.npz), 80/20 user split.
GPU: RTX 5060 (8GB) - only batches to GPU.

Usage: uv run python scripts/experiment_neural_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VAL_LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # counts/value cols to log1p


def load(window: int = 90) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path("data/sequences/seq_B_L180.npz"))
    X = z["X"][:, -window:, :].astype(np.float32)   # last `window` days
    y = z["y"].astype(np.float32)
    X[:, :, VAL_LOG1P_IDX] = np.log1p(X[:, :, VAL_LOG1P_IDX])
    return X, y


class MLP(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> int:
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    X, y = load(window=90)
    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_tr = int(n * 0.8)
    X_tr = torch.tensor(X[idx[:n_tr]].reshape(n_tr, -1), device=dev)
    y_tr = torch.tensor(y[idx[:n_tr]], device=dev)
    X_va = torch.tensor(X[idx[n_tr:]].reshape(n - n_tr, -1), device=dev)
    y_va = torch.tensor(y[idx[n_tr:]], device=dev)
    print(f"train {n_tr:,} / val {n - n_tr:,} | d_in={X_tr.shape[1]:,}")

    model = MLP(X_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    lossf = nn.MSELoss()
    bs = 512
    best = 1e9
    for epoch in range(15):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            loss = lossf(model(X_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            p = model(X_va)
            rmsle = float(torch.sqrt(((p - y_va) ** 2).mean()))
        best = min(best, rmsle)
        print(f"epoch {epoch + 1:2d}: train_mse={tot / n_tr:.4f} val_rmsle={rmsle:.4f}")
    print(f"BEST val RMSLE = {best:.4f} (tabular fold-B single ~1.68, random ~1.9-2.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
