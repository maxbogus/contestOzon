"""T3.8.14: blend GRU sequence preds with LGBM fold-B preds (log-space).

The GRU (tokenized daily behavior) and LGBM (engineered aggregates) use
different signal; blending in log-space may beat both. Grid over weight.

References: LGBM fold B 1.6881, GRU fold B ~1.715.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_t3 as bench  # noqa: E402

CACHE = "data/folds/h"
GRU_PREDS = "/tmp/gru_foldB_preds.npy"


def main() -> int:
    # LGBM on cached fold B (log-space predictions)
    X = pd.read_parquet(f"{CACHE}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_B_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    params = bench.make_params(bench.BASE_CONFIG, bench.SEED)
    model = lgb.train(params, lgb.Dataset(X.fillna(0.0), label=np.asarray(y)),
                      num_boost_round=int(bench.BASE_CONFIG["n_estimators"]))
    Xv = infer.reindex(all_users).fillna(0.0)
    lgb_log = np.asarray(model.predict(Xv), dtype=np.float64)  # log1p space
    y_true = np.log1p(truth.reindex(all_users).fillna(0.0).values)

    # GRU preds (user_id, pred_log)
    g = np.load(GRU_PREDS)
    gru = pd.DataFrame(g, columns=["user_id", "pred_log"]).set_index("user_id")
    gru_log = gru.reindex(all_users).fillna(0.0)["pred_log"].values

    print("=" * 60)
    print(f"LGBM alone: {float(np.sqrt(np.mean((lgb_log - y_true)**2))):.4f}")
    print(f"GRU alone:  {float(np.sqrt(np.mean((gru_log - y_true)**2))):.4f}")
    print("  log-space corr(lgb, gru):",
          f"{float(np.corrcoef(lgb_log, gru_log)[0, 1]):.4f}")
    best_w, best_r = 0.0, 1e9
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * gru_log + (1 - w) * lgb_log
        r = float(np.sqrt(np.mean((blended - y_true) ** 2)))
        if r < best_r:
            best_w, best_r = w, r
    print(f"BEST blend: w_gru={best_w:.2f} -> RMSLE {best_r:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
