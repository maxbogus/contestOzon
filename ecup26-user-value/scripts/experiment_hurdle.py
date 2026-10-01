#!/usr/bin/env python3
"""T3.17: hurdle (two-stage) model vs single-stage on cached folds.

The user's multi-stage idea: first classify WHO will be active in the window
(P(y>0)), then predict GMV growth ONLY for the active ones, zeros for the rest.

    pred = P(active) x expm1(E[log1p(GMV) | active])

Because P x E predicts the conditional MEAN (which over-shoots the RMSLE-optimal
point for a zero-inflated target), we sweep a log-space shrink c on the hurdle
prediction. Compared against the single-stage LGBM baseline on the SAME folds.

Usage: uv run python scripts/experiment_hurdle.py --cache data/folds/actor3_calv3 --folds B,G
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

CLF = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_child_samples": 50,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": bench.SEED,
}


def _load(cache: Path, name: str):
    X = pd.read_parquet(cache / f"fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id").fillna(0.0)
    infer = pd.read_parquet(cache / f"fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(cache / f"fold_{name}_truth.parquet")["y_true"]
    au = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    Xv = infer.reindex(au).fillna(0.0)
    yt = truth.reindex(au).fillna(0.0).values
    return X, y, Xv, yt


def _rmsle(yt: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(p)) ** 2)))



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default="data/folds/actor3_calv3")
    ap.add_argument("--folds", type=str, default="B,G")
    args = ap.parse_args()
    cache = Path(args.cache)

    for name in [f.strip() for f in args.folds.split(",")]:
        X, y, Xv, yt = _load(cache, name)
        y_arr = y.to_numpy(dtype=np.float64)
        active = np.expm1(np.clip(y_arr, 0.0, None)) > 0.0  # positional mask
        n_pos = int(active.sum())
        zf = 1.0 - n_pos / len(y_arr)

        # --- single-stage baseline ---
        m = lgb.train(
            bench.make_params(bench.BASE_CONFIG, bench.SEED),
            lgb.Dataset(X, label=np.asarray(y)),
            num_boost_round=int(bench.BASE_CONFIG["n_estimators"]),
        )
        pred_base = np.expm1(np.clip(np.asarray(m.predict(Xv)), 0.0, None))
        r_base = _rmsle(yt, pred_base)

        # --- hurdle: stage 1 P(active) ---
        clf = lgb.train(
            {**CLF, "scale_pos_weight": (len(y) - n_pos) / max(n_pos, 1)},
            lgb.Dataset(X, label=active.astype(np.int32)),
            num_boost_round=400,
        )
        p_active = np.asarray(clf.predict(Xv))

        # --- hurdle: stage 2 regression on active rows only (positional mask) ---
        Xa = X.iloc[active]
        ya = y_arr[active]
        m2 = lgb.train(
            bench.make_params(bench.BASE_CONFIG, bench.SEED),
            lgb.Dataset(Xa, label=ya),
            num_boost_round=int(bench.BASE_CONFIG["n_estimators"]),
        )
        pred_active = np.expm1(np.clip(np.asarray(m2.predict(Xv)), 0.0, None))

        hurdle = p_active * pred_active
        best_r, best_c = 1e9, 1.0
        for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r = _rmsle(yt, np.expm1(c * np.log1p(hurdle)))
            if r < best_r:
                best_r, best_c = r, c

        print(f"fold {name}: zero_frac={zf:.3f} | single={r_base:.4f} | "
              f"hurdle(c*={best_c:.1f})={best_r:.4f} | delta={best_r - r_base:+.4f} | "
              f"P(active) mean={p_active.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
