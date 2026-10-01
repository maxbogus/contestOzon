"""T3.9.12: per-bucket (predicted-GMV) bias calibration.

The whale analysis showed systematic bias: whales under-predicted 6x, zeros
over-predicted (~7.7). Global bias b~0 cancels these. Here we compute a
per-bucket bias (by PREDICTED GMV) on one fold and apply it to the OTHER fold
(cross-fold, no leakage), targeting the systematic tail bias.
References: LGBM fold B 1.6857 (refined config).
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

CACHE = "data/folds/w180"
REFINED = {"n_estimators": 550, "num_leaves": 97, "learning_rate": 0.02,
          "feature_fraction": 0.66, "bagging_fraction": 0.8, "min_child_samples": 21}
EDGES = [0.0, 10.0, 100.0, 1000.0, 1e12]


def _fit_predict(fold: str):
    X = pd.read_parquet(f"{CACHE}/fold_{fold}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_{fold}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_{fold}_truth.parquet")["y_true"]
    au = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    p = bench.make_params({**bench.BASE_CONFIG, **REFINED}, bench.SEED)
    m = lgb.train(p, lgb.Dataset(X.fillna(0.0), label=np.asarray(y)), num_boost_round=550)
    pred_log = np.asarray(m.predict(infer.reindex(au).fillna(0.0)))
    yt_log = np.log1p(truth.reindex(au).fillna(0.0).values)
    pred_raw = np.expm1(pred_log)
    return pred_log, yt_log, pred_raw


def main() -> int:
    predA, ytA, rawA = _fit_predict("A")
    predB, ytB, rawB = _fit_predict("B")

    def rmsle(p, yt):
        return float(np.sqrt(np.mean((p - yt) ** 2)))

    print("=" * 64)
    print(f"base: fold A {rmsle(predA, ytA):.4f} | fold B {rmsle(predB, ytB):.4f}")

    for src_name, (src_pred, src_yt, src_raw), (tgt_pred, tgt_yt, tgt_raw) in (
        ("A->B", (predA, ytA, rawA), (predB, ytB, rawB)),
        ("B->A", (predB, ytB, rawB), (predA, ytA, rawA)),
    ):
        biases = []
        for i in range(len(EDGES) - 1):
            msk = (src_raw >= EDGES[i]) & (src_raw < EDGES[i + 1])
            if msk.sum() < 10:
                biases.append(0.0)
            else:
                biases.append(float(np.mean(src_yt[msk] - src_pred[msk])))
        corr = tgt_pred.copy()
        for i in range(len(EDGES) - 1):
            msk = (tgt_raw >= EDGES[i]) & (tgt_raw < EDGES[i + 1])
            corr[msk] += biases[i]
        print(f"  {src_name} biases {[f'{b:+.3f}' for b in biases]}")
        tgt = "B" if src_name == "A->B" else "A"
        print(f"  {src_name}: calibrated fold {tgt} "
              f"{rmsle(corr, tgt_yt):.4f} (base {rmsle(tgt_pred, tgt_yt):.4f})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
