"""T3.9.5: LightGBM re-tuning on the CURRENT 138-feature model.

We tuned when there were ~90 features; after +120/180d windows (138 feats) the
optimum may have shifted. Systematic coordinate descent (one param at a time,
1/5-step across boundaries) then a pairwise 2D grid on the most sensitive two.
Runs on a subsample of fold B for speed; verifies the winner on the full fold.

References: fold B 1.6874 (base config 300/31/0.05/0.8/0.8, w180 features).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_t3 as bench  # noqa: E402

SEED = 42
CACHE = "data/folds/w180"
N_SAMPLE = 800_000

#: (name, values) - 1/5-step boundaries for coordinate descent.
SEARCH = {
    "n_estimators": [200, 300, 400, 500, 600],
    "learning_rate": [0.02, 0.04, 0.05, 0.06, 0.08],
    "num_leaves": [15, 23, 31, 47, 63, 95],
    "min_child_samples": [5, 10, 20, 50, 100],
    "feature_fraction": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l2": [0.0, 1.0, 3.0, 5.0, 10.0],
}


def _load(sample: bool):
    X = pd.read_parquet(f"{CACHE}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_B_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    if sample:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    Xf = X.fillna(0.0)
    Xv = infer.reindex(all_users).fillna(0.0)
    y_true = np.log1p(truth.reindex(all_users).fillna(0.0).values)
    return Xf, np.asarray(y, dtype=np.float64), Xv, y_true


def _params(base: dict, overrides: dict) -> dict:
    p = dict(base)
    p.update(overrides)
    return p


def _rmsle(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y_true) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-verify", action="store_true",
                    help="also verify the winner on the full fold B")
    args = ap.parse_args()

    Xf, y, Xv, y_true = _load(sample=True)
    base = {k: v for k, v in bench.BASE_CONFIG.items() if k in SEARCH}
    print(f"[data] subsample {len(Xf):,} rows x {Xf.shape[1]} feats")

    # coordinate descent
    best = dict(base)
    t0 = time.time()
    for param, values in SEARCH.items():
        row = []
        for v in values:
            over = dict(best)
            over[param] = v
            rounds = int(v) if param == "n_estimators" else int(best.get("n_estimators", 300))
            m = lgb.train(
                bench.make_params(_params(bench.BASE_CONFIG, over), SEED),
                lgb.Dataset(Xf, label=y), num_boost_round=rounds,
            )
            r = _rmsle(m.predict(Xv), y_true)
            row.append((v, r))
        v_best = min(row, key=lambda kv: kv[1])[0]
        best[param] = v_best
        print(f"  {param:18s} -> best {v_best} (row: "
              + ", ".join(f"{v}:{r:.4f}" for v, r in row) + ")")
    print(f"[cd] coordinate descent done in {time.time()-t0:.0f}s; "
          f"config={best}")

    # pairwise on the two most sensitive (largest spread): lr x leaves, mcs x ff
    pairs = [
        ("learning_rate", "num_leaves"),
        ("min_child_samples", "feature_fraction"),
    ]
    for p1, p2 in pairs:
        vals1, vals2 = SEARCH[p1], SEARCH[p2]
        grid = []
        for v1 in vals1:
            for v2 in vals2:
                over = dict(best)
                over[p1], over[p2] = v1, v2
                rounds = int(best.get("n_estimators", 300))
                m = lgb.train(
                    bench.make_params(_params(bench.BASE_CONFIG, over), SEED),
                    lgb.Dataset(Xf, label=y), num_boost_round=rounds,
                )
                grid.append(((v1, v2), _rmsle(m.predict(Xv), y_true)))
        best_v, best_r = min(grid, key=lambda kv: kv[1])
        best[p1], best[p2] = best_v
        print(f"  pairwise {p1}x{p2}: best ({p1}={best_v[0]}, {p2}={best_v[1]}) "
              f"RMSLE {best_r:.4f}")

    print("=" * 60)
    print(f"FINAL tuned config: {best}")
    if args.full_verify:
        XfF, yF, XvF, yTF = _load(sample=False)
        mF = lgb.train(
            bench.make_params(_params(bench.BASE_CONFIG, best), SEED),
            lgb.Dataset(XfF, label=yF),
            num_boost_round=int(best.get("n_estimators", 300)),
        )
        r_full = _rmsle(mF.predict(XvF), yTF)
        print(f"  full fold B verify: {r_full:.4f} (base 1.6874) -> "
              f"{r_full-1.6874:+.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
