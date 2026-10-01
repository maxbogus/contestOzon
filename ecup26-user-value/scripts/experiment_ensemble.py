#!/usr/bin/env python3
"""A/B harness: LightGBM hyperparameters + multi-seed ensemble on the walk-forward proxy.

Builds/caches the walk-forward training set once (parquet in /tmp), then trains
configs with custom params + N seeds and scores RMSLE on the proxy holdout.

Usage:
  # baseline (should reproduce the production proxy ~1.7070)
  uv run python scripts/experiment_ensemble.py --seeds 42 --n-estimators 50 \\
      --num-leaves 15 --lr 0.1 --feature-fraction 1.0 --bagging-fraction 1.0
  # tuned multi-seed
  uv run python scripts/experiment_ensemble.py --n-estimators 150 --num-leaves 31 \\
      --lr 0.05 --feature-fraction 0.8 --bagging-fraction 0.8 --seeds 42,43,44,45

Per .clinerules/02-ecup-rules.md: R6 (seeded), R8 (dev-only), R1 (lightgbm Apache-2.0).
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

from ecup26_user_value.features.build_features import build_features  # noqa: E402
from ecup26_user_value.run_pipeline import (  # noqa: E402
    _build_walk_forward_training,
    _default_num_threads,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiment")

EXP_CACHE = "/tmp/exp_walkforward_train.parquet"


_KEEP = [
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


def load_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"].astype(np.float32)
    df = df[[c for c in _KEEP if c in df.columns]]
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


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--holdout-days", type=int, default=30)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-child", type=int, default=20)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--seeds", type=str, default="42,43,44,45")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    df = load_clean(Path(args.data))
    max_date = df["event_date"].max()
    proxy_now = max_date - pd.Timedelta(days=args.holdout_days)
    train_df = df[df["event_date"] <= proxy_now].copy()
    truth_df = df[(df["event_date"] > proxy_now) & (df["event_date"] <= max_date)]
    truth = truth_df.groupby("user_id")["price"].sum()
    all_users = np.sort(
        np.unique(np.concatenate([train_df["user_id"].unique(), truth.index.values]))
    )
    del df, truth_df

    cache = Path(EXP_CACHE)
    if cache.exists() and not args.rebuild:
        X = pd.read_parquet(cache)
        y = X.pop("y_log").astype(float)
        logger.info("Loaded cached walk-forward train: %d rows x %d", len(X), X.shape[1] - 1)
    else:
        logger.info("Building walk-forward train (cached to %s)...", EXP_CACHE)
        X, y = _build_walk_forward_training(train_df, proxy_now)
        out = X.copy()
        out["y_log"] = np.asarray(y, dtype=np.float64)
        out.to_parquet(cache, index=False)
        logger.info("Saved %d-row train cache", len(out))
    # user_id is an identifier, not a feature: move it out of X (must match
    # the inference feature matrix which is indexed by user_id).
    X = X.set_index("user_id")
    X = X[[c for c in X.columns if c != "user_id"]]

    logger.info("Building inference features at %s...", proxy_now.date())
    X_infer = build_features(train_df, ref_date=proxy_now).set_index("user_id")
    Xv = X_infer.reindex(all_users).fillna(0.0)

    seeds = [int(s) for s in args.seeds.split(",")]
    logger.info("Training %d seeds: %s", len(seeds), seeds)
    raw_preds = []
    for seed in seeds:
        params = {
            "objective": "regression",
            "num_leaves": args.num_leaves,
            "learning_rate": args.lr,
            "min_child_samples": args.min_child,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": seed,
            "num_threads": _default_num_threads(),
        }
        model = _fit(X, y, params, args.n_estimators)
        raw_preds.append(np.asarray(model.predict(Xv), dtype=np.float64))

    raw = np.mean(raw_preds, axis=0)
    pred = np.expm1(raw)
    y_true = truth.reindex(all_users).fillna(0.0).values
    naive = rmsle(y_true, np.full(len(y_true), float(y_true.mean())))
    print("=" * 60)
    print(
        f"config: n_est={args.n_estimators} leaves={args.num_leaves} lr={args.lr} "
        f"min_child={args.min_child} ff={args.feature_fraction} bag={args.bagging_fraction} "
        f"seeds={seeds}"
    )
    print(f"model RMSLE = {rmsle(y_true, pred):.4f}   (naive = {naive:.4f})")
    print(f"pred mean={pred.mean():.2f} median={np.median(pred):.2f} zeros={(pred == 0).sum()}")
    print("=" * 60)
    return 0


def _fit(X: pd.DataFrame, y: pd.Series, params: dict, n_estimators: int):
    import lightgbm as lgb

    Xt = X.fillna(0.0)
    dset = lgb.Dataset(Xt, label=np.asarray(y, dtype=np.float64))
    return lgb.train(params, dset, num_boost_round=n_estimators)


if __name__ == "__main__":
    sys.exit(main())
