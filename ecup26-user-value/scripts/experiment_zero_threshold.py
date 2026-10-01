"""Zero-inflation threshold: pred = 0 if P(y>0) < t else keep regression.

The RMSLE error decomposition showed 45.9% zeros contribute 50% of error^2 (the
model predicts ~7.7 GMV for them). For zero-inflated targets the squared-error
optimal predictor is a HARD ZERO, not the soft conditional mean. Here we sweep
the threshold over a P(y>0) classifier, applied to LGBM / XGB / CatBoost
regressions. References: fold B 1.6857 (refined LGBM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecup26_user_value import hybrid  # noqa: E402
from scripts import benchmark_t3 as bench  # noqa: E402

CACHE = "data/folds/w180"
REFINED = {"n_estimators": 550, "num_leaves": 97, "learning_rate": 0.02,
          "feature_fraction": 0.66, "bagging_fraction": 0.8, "min_child_samples": 21}


def _load(name: str):
    X = pd.read_parquet(f"{CACHE}/fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_{name}_truth.parquet")["y_true"]
    au = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    Xf = X.fillna(0.0)
    Xv = infer.reindex(au).fillna(0.0)
    yt = np.log1p(truth.reindex(au).fillna(0.0).values)
    return Xf, np.asarray(y, dtype=np.float64), Xv, yt, au


def _fit_reg(name: str, Xf, y, Xv):
    if name == "lgb":
        p = bench.make_params({**bench.BASE_CONFIG, **REFINED}, bench.SEED)
        m = lgb.train(p, lgb.Dataset(Xf, label=y), num_boost_round=REFINED["n_estimators"])
    elif name == "xgb":
        m = xgb.XGBRegressor(n_estimators=550, learning_rate=0.02, max_depth=7,
                             min_child_weight=10, subsample=0.8, colsample_bytree=0.66,
                             reg_lambda=1.0, tree_method="hist", random_state=bench.SEED,
                             n_jobs=bench._default_num_threads())
        m.fit(Xf, y)
    else:
        m = CatBoostRegressor(iterations=550, learning_rate=0.02, depth=7,
                              min_data_in_leaf=20, l2_leaf_reg=1.0, subsample=0.8,
                              random_seed=bench.SEED, thread_count=bench._default_num_threads(),
                              verbose=0)
        m.fit(Xf, y)
    return np.asarray(m.predict(Xv), dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=str, default="B")
    args = ap.parse_args()

    results = {}
    for fold in args.folds.split(","):
        Xf, y, Xv, yt, au = _load(fold)
        y_bin = (y > 0).astype(np.int32)

        regs = {}
        for rn in ("lgb", "xgb", "cat"):
            regs[rn] = _fit_reg(rn, Xf, y, Xv)

        # P(y>0) classifier (full-data; hybrid.py single source)
        clf = hybrid.fit_purchase_prob(Xf.values, y_bin, bench.SEED)
        P = np.asarray(clf.predict(Xv.values), dtype=np.float64)
        yt_bin = (np.expm1(yt) > 0).astype(np.int32)
        print(f"[fold {fold}] P(y>0) AUC={roc_auc_score(yt_bin, P):.4f}")

        for rn, pred in regs.items():
            base_r = float(np.sqrt(np.mean((pred - yt) ** 2)))
            best_t, best_r = 0.0, base_r
            for t in np.arange(0.0, 0.61, 0.02):
                pt = np.where(t <= P, pred, 0.0)
                r = float(np.sqrt(np.mean((pt - yt) ** 2)))
                if r < best_r:
                    best_t, best_r = t, r
            results[(fold, rn)] = (best_t, best_r, base_r)
            print(f"  {rn:4s} base {base_r:.4f} -> threshold t={best_t:.2f}: {best_r:.4f} "
                  f"({best_r-base_r:+.4f})")
        # blend thresholding (mean of the three log-space preds)
        blend = np.mean([regs[rn] for rn in ("lgb", "xgb", "cat")], axis=0)
        base_r = float(np.sqrt(np.mean((blend - yt) ** 2)))
        best_t, best_r = 0.0, base_r
        for t in np.arange(0.0, 0.61, 0.02):
            pt = np.where(t <= P, blend, 0.0)
            r = float(np.sqrt(np.mean((pt - yt) ** 2)))
            if r < best_r:
                best_t, best_r = t, r
        results[(fold, "blend")] = (best_t, best_r, base_r)
        print(f"  blend base {base_r:.4f} -> threshold t={best_t:.2f}: {best_r:.4f} "
              f"({best_r-base_r:+.4f})")
    print("=" * 60)
    for k, (t, r, b) in sorted(results.items()):
        print(f"  {k[0]}/{k[1]:5s}: base {b:.4f} | t={t:.2f} -> {r:.4f} ({r-b:+.4f})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
