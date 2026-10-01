"""T3.9.10: quick coordinate-descent tuning for XGBoost + CatBoost.

We tuned only LGBM (T3.9.5). XGB was already 1.6860 at base config. Quick
1-D coordinate search over the sensitive params on an 800k subsample, then
verify the winner on full fold B (and blend with LGBM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_t3 as bench  # noqa: E402

SEED = 42
CACHE = "data/folds/w180"
N_SAMPLE = 800_000

XGB_GRID = {
    "max_depth": [5, 6, 7, 9],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "min_child_weight": [5, 10, 20, 50],
    "colsample_bytree": [0.5, 0.66, 0.8, 1.0],
    "n_estimators": [400, 550, 700],
}
CAT_GRID = {
    "depth": [5, 6, 7, 9],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "min_data_in_leaf": [10, 20, 50],
    "l2_leaf_reg": [0.5, 1.0, 3.0],
    "iterations": [400, 550, 700],
}


def _load():
    X = pd.read_parquet(f"{CACHE}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_B_truth.parquet")["y_true"]
    au = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
    Xf = X.iloc[idx].fillna(0.0)
    y_s = np.asarray(y.iloc[idx], dtype=np.float64)
    Xv = infer.reindex(au).fillna(0.0)
    yt = np.log1p(truth.reindex(au).fillna(0.0).values)
    return Xf, y_s, Xv, yt, list(X.columns)


def _xgb_fit_eval(Xf, y, Xv, yt, cfg):
    m = xgb.XGBRegressor(n_estimators=cfg.get("n_estimators", 550),
                         learning_rate=cfg.get("learning_rate", 0.02),
                         max_depth=cfg.get("max_depth", 7),
                         min_child_weight=cfg.get("min_child_weight", 10),
                         subsample=0.8, colsample_bytree=cfg.get("colsample_bytree", 0.66),
                         reg_lambda=1.0, tree_method="hist", random_state=SEED,
                         n_jobs=bench._default_num_threads())
    m.fit(Xf, y)
    return float(np.sqrt(np.mean((m.predict(Xv) - yt) ** 2)))


def _cat_fit_eval(Xf, y, Xv, yt, cfg):
    m = CatBoostRegressor(iterations=cfg.get("iterations", 550),
                          learning_rate=cfg.get("learning_rate", 0.02),
                          depth=cfg.get("depth", 7),
                          min_data_in_leaf=cfg.get("min_data_in_leaf", 20),
                          l2_leaf_reg=cfg.get("l2_leaf_reg", 1.0),
                          subsample=0.8, random_seed=SEED,
                          thread_count=bench._default_num_threads(), verbose=0)
    m.fit(Xf, y)
    return float(np.sqrt(np.mean((m.predict(Xv) - yt) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-verify", action="store_true")
    args = ap.parse_args()

    Xf, y, Xv, yt, feats = _load()
    print(f"[data] subsample {len(Xf):,} x {Xf.shape[1]} feats")

    xbest = {"max_depth": 7, "learning_rate": 0.02, "min_child_weight": 10,
             "colsample_bytree": 0.66, "n_estimators": 550}
    cbest = {"depth": 7, "learning_rate": 0.02, "min_data_in_leaf": 20,
             "l2_leaf_reg": 1.0, "iterations": 550}

    print("--- XGBoost ---")
    for param, vals in XGB_GRID.items():
        row = []
        for v in vals:
            cfg = dict(xbest)
            cfg[param] = v
            row.append((v, _xgb_fit_eval(Xf, y, Xv, yt, cfg)))
        xbest[param] = min(row, key=lambda kv: kv[1])[0]
        print(f"  {param:18s} -> best {xbest[param]}: "
              + ", ".join(f"{v}:{r:.4f}" for v, r in row))
    print(f"  XGB final: {xbest}")

    print("--- CatBoost ---")
    for param, vals in CAT_GRID.items():
        row = []
        for v in vals:
            cfg = dict(cbest)
            cfg[param] = v
            row.append((v, _cat_fit_eval(Xf, y, Xv, yt, cfg)))
        cbest[param] = min(row, key=lambda kv: kv[1])[0]
        print(f"  {param:18s} -> best {cbest[param]}: "
              + ", ".join(f"{v}:{r:.4f}" for v, r in row))
    print(f"  Cat final: {cbest}")

    if args.full_verify:
        XfF = pd.read_parquet(f"{CACHE}/fold_B_train.parquet")
        yF = XfF.pop("y_log").astype(float)
        if "origin" in XfF.columns:
            XfF = XfF.drop(columns=["origin"])
        XfF = XfF.set_index("user_id").fillna(0.0)
        XvF = pd.read_parquet(f"{CACHE}/fold_B_infer.parquet").set_index("user_id")
        truth = pd.read_parquet(f"{CACHE}/fold_B_truth.parquet")["y_true"]
        au = np.sort(np.unique(np.concatenate([XvF.index.values, truth.index.values])))
        XvF = XvF.reindex(au).fillna(0.0)
        ytF = np.log1p(truth.reindex(au).fillna(0.0).values)
        r_x = _xgb_fit_eval(XfF, np.asarray(yF), XvF, ytF, xbest)
        r_c = _cat_fit_eval(XfF, np.asarray(yF), XvF, ytF, cbest)
        # blend of tuned XGB + Cat happens in the final ensemble script
        print(f"  full fold B: XGB {r_x:.4f} | Cat {r_c:.4f} | LGBM ref 1.6857")
    print("=" * 60)
    print(f"XGB tuned: {xbest}")
    print(f"Cat tuned: {cbest}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
