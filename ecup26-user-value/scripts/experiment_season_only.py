"""T3.8.13 diagnostic: train ONLY on the 2025 gift season, predict fold B.

Hypothesis (user): the target window (Feb 14 - Mar 15 2026) is the gift season
(23 Feb / 8 Mar / Ozon promos); a model trained exclusively on the analogous
period last year might capture it best.

Reality check (documented):
- Data starts 2025-01-01, so "last year's gift season" exists exactly ONCE
  (Feb 14 - Mar 15 2025) -> training set is a SINGLE origin (~215k rows).
- At that origin the feature history is only ~45 days (data starts Jan 1 2025),
  while the 2026 target users have ~13 months of history -> feature
  distribution shift (T, recency, totals are systematically smaller).
- No Feb-Mar 2026 labels exist locally (private LB), so the fold-B proxy
  (Jan 14 - Feb 13 2026) is the only available evaluation.

Usage:
    uv run python scripts/experiment_season_only.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.features.build_features import build_features  # noqa: E402
from ecup26_user_value.run_pipeline import _default_num_threads  # noqa: E402
from scripts import benchmark_t3 as bench  # noqa: E402

SEED = 42
GIFT_ORIGIN = "2025-02-14"  # features as of Feb 14 2025
GIFT_LABEL_END = "2025-03-15"  # label = GMV over (Feb 14, Mar 15] 2025


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--fold-cache-dir", type=str, default="data/folds/h")
    args = ap.parse_args()

    df = bench._load_clean(Path(args.data))

    # --- training set: single gift-season origin (Feb 14 - Mar 15 2025) ---
    origin_ts = pd.Timestamp(GIFT_ORIGIN)
    train = df[df["event_date"] <= origin_ts]
    X = build_features(train, ref_date=origin_ts)
    y30 = (
        df[(df["event_date"] > origin_ts) & (df["event_date"] <= pd.Timestamp(GIFT_LABEL_END))]
        .groupby("user_id")["price"]
        .sum()
    )
    X["y_log"] = X["user_id"].map(np.log1p(y30)).fillna(0.0)
    feats = [c for c in X.columns if c not in ("user_id", "y_log")]
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": SEED,
        "num_threads": _default_num_threads(),
    }
    model = lgb.train(
        params,
        lgb.Dataset(X[feats], label=X["y_log"].astype(float)),
        num_boost_round=300,
    )

    # --- evaluate on the fold-B proxy (Jan 14 - Feb 13 2026) ---
    cache = Path(args.fold_cache_dir)
    infer = pd.read_parquet(cache / "fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(cache / "fold_B_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    Xv = infer.reindex(all_users).fillna(0.0)
    pred_log = np.clip(np.asarray(model.predict(Xv[feats]), dtype=np.float64), -50.0, None)
    pred = np.expm1(pred_log)
    y_true = truth.reindex(all_users).fillna(0.0).values
    rmsle = _rmsle(y_true, pred)

    # --- also: predict the real target (Feb 14 - Mar 15 2026) if requested ---
    print("=" * 60)
    print(f"season-only model: train {len(X)} rows (single origin {GIFT_ORIGIN})")
    print(f"  train label = GMV over {GIFT_ORIGIN}..{GIFT_LABEL_END} 2025 (gift season)")
    print(f"  fold B proxy RMSLE: {rmsle:.4f}")
    print("  reference full walk-forward fold B: 1.6881")
    print("  reference full walk-forward CV mean: 1.7155")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
