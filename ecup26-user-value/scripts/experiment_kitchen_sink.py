"""Kitchen-sink ensemble: blend LGBM + XGBoost + CatBoost + GRU on fold B.

All models' log-space predictions are blended with a weight grid. Expected to
be flat (log-space corr 0.97-0.99), but settles the "stack the noise gains"
question with a fact. Uses the extended-window cache (138 features).
References: fold B 1.6874 (LGBM, w180).
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_t3 as bench  # noqa: E402

CACHE = "data/folds/w180"
GRU_PREDS = "/tmp/gru_foldB_preds.npy"


def main() -> int:
    X = pd.read_parquet(f"{CACHE}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_B_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    y_true = np.log1p(truth.reindex(all_users).fillna(0.0).values)
    feats = list(X.columns)
    Xf = X[feats].fillna(0.0)
    Xv = infer.reindex(all_users)[feats].fillna(0.0)

    # LGBM
    params = bench.make_params(bench.BASE_CONFIG, bench.SEED)
    lgbm = lgb.train(params, lgb.Dataset(Xf, label=np.asarray(y, dtype=np.float64)),
                     num_boost_round=int(bench.BASE_CONFIG["n_estimators"]))
    pred_lgb = np.asarray(lgbm.predict(Xv), dtype=np.float64)

    # XGBoost
    xgbm = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=bench.SEED, n_jobs=bench._default_num_threads(),
    )
    xgbm.fit(Xf, y)
    pred_xgb = np.asarray(xgbm.predict(Xv), dtype=np.float64)

    # CatBoost
    catm = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=5, min_data_in_leaf=20,
        l2_leaf_reg=1.0, subsample=0.8, random_seed=bench.SEED,
        thread_count=bench._default_num_threads(), verbose=0,
    )
    catm.fit(Xf, y)
    pred_cat = np.asarray(catm.predict(Xv), dtype=np.float64)

    # GRU (raw daily tokens; windows irrelevant)
    g = np.load(GRU_PREDS)
    pred_gru = pd.DataFrame(g, columns=["user_id", "pred_log"]).set_index("user_id")
    pred_gru = pred_gru.reindex(all_users).fillna(0.0)["pred_log"].values

    def rmsle(p: np.ndarray) -> float:
        return float(np.sqrt(np.mean((p - y_true) ** 2)))

    preds = {"lgb": pred_lgb, "xgb": pred_xgb, "cat": pred_cat, "gru": pred_gru}
    print("=" * 60)
    print("individual fold-B RMSLE:")
    for name, p in preds.items():
        print(f"  {name:5s} {rmsle(p):.4f}")
    print("  log-space corr with lgb:")
    for name, p in preds.items():
        print(f"    {name:5s} {np.corrcoef(pred_lgb, p)[0,1]:.4f}")

    # weight grid over (gru, xgb, cat); lgb = 1 - rest
    best_w, best_r = None, 1e9
    for w_gru in np.arange(0.0, 0.26, 0.05):
        for w_xgb in np.arange(0.0, 0.31, 0.1):
            for w_cat in np.arange(0.0, 0.31, 0.1):
                if w_gru + w_xgb + w_cat > 0.5:
                    continue
                w_lgb = 1.0 - w_gru - w_xgb - w_cat
                blended = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cat * pred_cat + w_gru * pred_gru
                r = rmsle(blended)
                if r < best_r:
                    best_r, best_w = r, (w_lgb, w_xgb, w_cat, w_gru)
    print("-" * 60)
    print(f"BEST blend (w_lgb,w_xgb,w_cat,w_gru)={best_w}: RMSLE {best_r:.4f}")
    print(f"  vs LGBM alone: 1.6874 -> delta {best_r-1.6874:+.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
