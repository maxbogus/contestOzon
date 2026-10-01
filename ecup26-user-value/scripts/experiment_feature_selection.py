"""T3.9.1: feature selection - LightGBM gain-rank + top-K pruning.

Answers: does pruning 114 -> top-K features help fold-B RMSLE, and which
features/windows carry the signal? Uses cached walk-forward folds (no rebuild).
References: fold B 1.6881, CV 1.7155 (full 114 features).
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

from scripts import benchmark_t3 as bench  # noqa: E402

CACHE = "data/folds/h"


def _load_fold(name: str):
    X = pd.read_parquet(f"{CACHE}/fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_{name}_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    return X, y, infer, truth, all_users


def _rmsle(y_log_pred: np.ndarray, y_true_raw: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_log_pred - np.log1p(y_true_raw)) ** 2)))


def _fit_predict(X, y, infer, all_users, feats, params, n_est):
    Xv = infer.reindex(all_users).fillna(0.0)
    model = lgb.train(params, lgb.Dataset(X[feats].fillna(0.0), label=np.asarray(y)),
                      num_boost_round=n_est)
    return model, np.asarray(model.predict(Xv[feats]), dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default=CACHE)
    ap.add_argument("--ks", type=str, default="10,20,30,40,60,80,100,114")
    args = ap.parse_args()

    params = bench.make_params(bench.BASE_CONFIG, bench.SEED)
    n_est = int(bench.BASE_CONFIG["n_estimators"])

    X, y, infer, truth, all_users = _load_fold("B")
    feats_all = list(X.columns)
    model_full, _ = _fit_predict(X, y, infer, all_users, feats_all, params, n_est)
    gain = dict(zip(feats_all, model_full.feature_importance("gain"), strict=False))
    ranked = sorted(feats_all, key=lambda f: -gain[f])

    print("=" * 72)
    print("T3.9.1 feature selection (fold B proxy)")
    print("top-30 by gain:")
    for i, f in enumerate(ranked[:30], 1):
        print(f"  {i:2d}. {f:40s} gain={gain[f]:,.0f}")
    print("-" * 72)
    print(f"{'K':>6s} {'RMSLE':>10s} {'vs full':>10s}")
    for ks in args.ks.split(","):
        k = int(ks)
        feats = ranked[:k]
        _, pred = _fit_predict(X, y, infer, all_users, feats, params, n_est)
        y_true = truth.reindex(all_users).fillna(0.0).values
        r = _rmsle(pred, y_true)
        print(f"{k:6d} {r:10.4f} {r - 1.6881:10.4f}")
    print("=" * 72)

    # window gain aggregation (T3.9.2a): which rolling windows carry signal
    print("\nWindow gain aggregation (sum of gain by window suffix):")
    windows = {}
    for f, g in gain.items():
        for w in ("90d", "60d", "30d", "14d", "7d"):
            if f.endswith(w):
                windows.setdefault(w, 0.0)
                windows[w] += g
                break
    total = sum(gain.values())
    for w, g in sorted(windows.items(), key=lambda kv: -kv[1]):
        print(f"  {w}: {g:,.0f} ({100*g/total:.1f}% of total gain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
