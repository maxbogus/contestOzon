#!/usr/bin/env python3
"""Validate the 30-day-target pipeline on a proxy holdout (last N train days).

Simulates the leaderboard exactly: trains on data up to `proxy_now` (= last
train date minus holdout_days), predicts GMV for the next `holdout_days`, and
scores with RMSLE against the real per-user GMV in that window (zeros filled
for inactive users, as the organizers do).

Reports the naive constant-mean RMSLE as the reference: the pipeline must beat
it (previously the lifetime-target model scored 3.96 vs naive 3.18).

Usage:
    uv run python scripts/validate_pipeline_30d.py \
        [--data data/train.parquet] [--holdout-days 30] [--cache-dir /tmp/t3val_cache]

Per .clinerules/02-ecup-rules.md:
- R6: reproducible (seeded LightGBM)
- R8: dev-only script, writes only to --cache-dir
- R10: user_ids unique
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.run_pipeline import predict_for_users  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_pipeline_30d")


def _load_clean(data_path: Path) -> pd.DataFrame:
    """Load train.parquet and normalize to the T3 pipeline schema.

    Keeps only the columns used by feature engineering and downcasts numeric
    dtypes (float32/int32/int8): the full 18-column float64/int64 frame blows
    up RAM during groupby feature building on 30M rows.
    """
    df = pd.read_parquet(data_path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"].astype(np.float32)
    _keep = [
        "user_id",
        "event_date",
        "price",
        "gmv",
        "search",
        "cat",
        "searches",
        "to_cart",
        "to_ord",
        "search_to_cart",
        "search_to_ord",
        "cat_to_cart",
        "cat_to_ord",
        "gmv_search",
        "gmv_cat",
    ]
    df = df[[c for c in _keep if c in df.columns]]
    for c in ("price", "gmv", "gmv_search", "gmv_cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    for c in (
        "searches",
        "to_cart",
        "to_ord",
        "search_to_cart",
        "search_to_ord",
        "cat_to_cart",
        "cat_to_ord",
    ):
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    for c in ("search", "cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.int8)
    return df


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE with log1p, matching the contest metric (negatives clipped)."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.parquet",
        help="Path to train.parquet (default: data/train.parquet)",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=30,
        help="Length of the proxy future window in days (default: 30)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/t3val_cache",
        help="Cache dir for the dev-resume cache (default: /tmp/t3val_cache)",
    )
    args = parser.parse_args()

    df = _load_clean(Path(args.data))
    max_date = df["event_date"].max()
    proxy_now = max_date - pd.Timedelta(days=args.holdout_days)

    train_df = df[df["event_date"] <= proxy_now].copy()
    truth_df = df[(df["event_date"] > proxy_now) & (df["event_date"] <= max_date)]
    truth = truth_df.groupby("user_id")["price"].sum()

    all_users = np.sort(
        np.unique(np.concatenate([train_df["user_id"].unique(), truth.index.values]))
    )
    logger.info(
        "proxy_now=%s, train rows=%d, truth users=%d, predict for %d users",
        proxy_now.date(),
        len(train_df),
        truth.nunique(),
        len(all_users),
    )

    pred = predict_for_users(train_df, all_users, cache_dir=Path(args.cache_dir))
    assert len(pred) == len(all_users)
    y_true = truth.reindex(all_users).fillna(0.0).values

    model_rmsle = _rmsle(y_true, pred)
    naive_rmsle = _rmsle(y_true, np.full(len(y_true), float(y_true.mean())))

    print("=" * 62)
    print(f"proxy holdout: ({proxy_now.date()}, {max_date.date()}]  {args.holdout_days} days")
    print(f"model RMSLE (30d-target pipeline)  = {model_rmsle:.4f}")
    print(f"naive RMSLE (constant mean)        = {naive_rmsle:.4f}")
    print(
        f"lift vs naive                      = {100 * (naive_rmsle - model_rmsle) / naive_rmsle:+.2f}%"
    )
    print(
        f"pred : mean={pred.mean():9.2f} median={np.median(pred):9.2f} zeros={int((pred == 0).sum())}"
    )
    print(
        f"truth: mean={y_true.mean():9.2f} median={np.median(y_true):9.2f} zeros={int((y_true == 0).sum())}"
    )
    print(f"rmsle < naive (PASS): {model_rmsle < naive_rmsle}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
