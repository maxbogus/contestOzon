#!/usr/bin/env python3
"""T3.19/E0: build dense daily-sequence tensors [n_users, window, D].

For a ref_date (walk-forward origin) builds per-user dense sequences of the
last `window` CALENDAR days before ref (zeros for inactive days = honest gaps).
Each day = 16 raw activity columns + 4 structural (day_of_week, days_to_ref,
is_holiday_ru, days_to_next_holiday) => D=20. Label = log1p(GMV (ref, ref+30]).

Chunked over users (RAM budget: chunk x L x D x 4 bytes; L=365 ~ 7.7GB full).
Leak-free: only rows <= ref_date enter the sequence; label uses (ref, ref+30].

Usage: uv run python scripts/build_sequences.py --ref 2026-01-14 --window 180 \
       --out data/sequences/seq_B_L180.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecup26_user_value.features.build_features import (  # noqa: E402
    _RU_HOLIDAYS,
    _days_until_next,
    _is_holiday,
)
from scripts import benchmark_t3 as bench  # noqa: E402

RAW_COLS = [
    "search", "cat", "searches", "to_cart", "to_ord",
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
    "gmv_search", "gmv_cat", "gmv",
]
N_RAW = len(RAW_COLS)      # 12 (has_* flags dropped: counts subsume them; _load_clean drops them too)
N_STRUCT = 4               # day_of_week, days_to_ref, is_holiday_ru, days_to_next_holiday
D = N_RAW + N_STRUCT       # 16


def _structural(dates: pd.DatetimeIndex, ref_date: pd.Timestamp) -> np.ndarray:
    """Per-date structural features [len(dates), N_STRUCT] (leak-free, public calendar)."""
    out = np.zeros((len(dates), N_STRUCT), dtype=np.float32)
    out[:, 0] = dates.dayofweek
    out[:, 1] = (ref_date - dates).days  # 0 at ref, grows backwards
    out[:, 2] = [_is_holiday(d) for d in dates]
    out[:, 3] = [_days_until_next(_RU_HOLIDAYS, d) for d in dates]
    return out


def build_sequences(df: pd.DataFrame, ref_date: pd.Timestamp, window: int = 180,
                    chunk: int = 50000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X[n,L,D] float32, y[n] log1p-GMV, user_ids[n])."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    start = ref_date - pd.Timedelta(days=window - 1)
    dates = pd.date_range(start, ref_date, freq="D")
    past = df[df["event_date"] <= ref_date]
    past = past[(past["event_date"] >= start)]
    past["_pos"] = (past["event_date"] - start).dt.days
    users = np.sort(past["user_id"].unique())
    y = np.zeros(len(users), dtype=np.float32)
    price_col = "price" if "price" in df.columns else "gmv"
    target = df[(df["event_date"] > ref_date) & (df["event_date"] <= ref_date + pd.Timedelta(days=30))]
    g = target.groupby("user_id")[price_col].sum()
    y = np.log1p(g.reindex(users).fillna(0.0).to_numpy()).astype(np.float32)
    struct = _structural(dates, ref_date)
    X = np.zeros((len(users), window, D), dtype=np.float32)
    uid_to_i = {u: i for i, u in enumerate(users)}
    uid_arr = np.asarray(users)
    pos_arr = past["_pos"].to_numpy(dtype=np.int64)
    uid_arr2 = past["user_id"].to_numpy()
    # map user_id -> global index (vectorized via searchsorted, users sorted)
    gidx = np.searchsorted(uid_arr, uid_arr2)
    feat = past[RAW_COLS].to_numpy(dtype=np.float32)
    # advanced-indexing fill (vectorized, no per-row loop)
    X[gidx, pos_arr, :N_RAW] = feat
    X[:, :, N_RAW:] = struct[np.newaxis, :, :]
    return X, y, users


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=str, default="2026-01-14")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--out", type=str, default="data/sequences/seq.npz")
    args = ap.parse_args()
    ref = pd.Timestamp(args.ref)
    df = bench._load_clean(Path("data/train.parquet"))
    print(f"building sequences ref={ref.date()} window={args.window} (raw D={D})...")
    X, y, users = build_sequences(df, ref, window=args.window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, user_ids=users)
    print(f"saved {out}: X={X.shape} y={y.shape} ({out.stat().st_size/1e6:.0f} MB)")
    print(f"zero_frac={float((y == 0).mean()):.3f} mean_y={float(np.expm1(y).mean()):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
