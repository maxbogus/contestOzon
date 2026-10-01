#!/usr/bin/env python3
"""T3.18: Permutation Null Importance - separate signal from noise features.

For sparse data many features are 0/NaN for 70%+ users. Shuffle the target,
retrain, and compare REAL vs NULL gain importance. A feature whose real
importance does NOT beat its shuffled-target (null) importance is a
noise-magnet (the model overfits its sparse structure), not signal.

Usage: uv run python scripts/null_importance.py --cache data/folds/actor3_calv3 --fold B --shuffles 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts import benchmark_t3 as bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default="data/folds/actor3_calv3")
    ap.add_argument("--fold", type=str, default="B")
    ap.add_argument("--shuffles", type=int, default=3)
    args = ap.parse_args()
    cache = Path(args.cache)

    X = pd.read_parquet(cache / f"fold_{args.fold}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id").fillna(0.0)
    cols = list(X.columns)
    yv = y.to_numpy(dtype=np.float64)

    def fit(label):
        m = lgb.train(
            bench.make_params(bench.BASE_CONFIG, bench.SEED),
            lgb.Dataset(X, label=label),
            num_boost_round=int(bench.BASE_CONFIG["n_estimators"]),
        )
        return m.feature_importance("gain")

    print(f"fitting real + {args.shuffles} null models on fold {args.fold}...")
    real = fit(yv)
    rng = np.random.default_rng(bench.SEED)
    nulls = []
    for k in range(args.shuffles):
        nulls.append(fit(rng.permutation(yv)))
    null_mean = np.mean(nulls, axis=0)
    null_std = np.std(nulls, axis=0)
    # score: real gain / (null gain + 1); <= 1 means null >= real (noise)
    score = real / (null_mean + 1.0)
    rows = sorted(zip(cols, real, null_mean, score), key=lambda r: r[3])
    print("=== noise features (score <= 1: real does not beat null) ===")
    for c, r, n, s in rows:
        if s <= 1.0:
            print(f"  score={s:5.2f} real={r:>10.0f} null={n:>10.0f}  {c}")
    print("=== top signal features ===")
    for c, r, n, s in reversed(rows[-15:]):
        print(f"  score={s:6.1f} real={r:>10.0f} null={n:>8.0f}  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
