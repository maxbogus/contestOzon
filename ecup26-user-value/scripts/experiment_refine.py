"""T3.9.7: fine local search (+-10%, 1/10 steps) around the tuned optimum.

The coordinate descent + pairwise (T3.9.5) found (500/95/0.02/0.6/0.8/20).
Landscape is flat, so this is a fine +-10% refinement (11 points/param) plus a
fine learning_rate x num_leaves pairwise, with the discipline that the winner
must improve BOTH full folds A and B (guards fold-B overfitting).

References: tuned fold A 1.7415 / fold B 1.6861 / CV 1.7138.
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

OPT = {
    "n_estimators": 500,
    "num_leaves": 95,
    "learning_rate": 0.02,
    "min_child_samples": 20,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "lambda_l2": 0.0,
}


def _fine_grid(name: str) -> list:
    v = OPT[name]
    if name == "num_leaves":
        return list(range(int(v - 10), int(v + 11), 2))
    if name == "min_child_samples":
        return list(range(int(v - 2), int(v + 3)))
    if name == "n_estimators":
        return list(range(int(v - 50), int(v + 51), 10))
    lo, hi = v * 0.9, v * 1.1
    step = (hi - lo) / 10
    return [round(lo + i * step, 6) for i in range(11)]


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


def _fit_eval(Xf, y, Xv, y_true, over: dict, rounds: int) -> float:
    p = bench.make_params({**bench.BASE_CONFIG, **over}, SEED)
    m = lgb.train(p, lgb.Dataset(Xf, label=y), num_boost_round=rounds)
    return float(np.sqrt(np.mean((m.predict(Xv) - y_true) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-verify", action="store_true")
    args = ap.parse_args()

    Xf, y, Xv, y_true = _load(sample=True)
    print(f"[data] subsample {len(Xf):,} rows x {Xf.shape[1]} feats")
    best = dict(OPT)
    t0 = time.time()

    # 1-D fine sweep (+-10%, 1/10 steps)
    for param in ("n_estimators", "num_leaves", "learning_rate",
                  "min_child_samples", "feature_fraction", "bagging_fraction"):
        row = []
        for v in _fine_grid(param):
            over = dict(best)
            over[param] = v
            rounds = int(over["n_estimators"])
            row.append((v, _fit_eval(Xf, y, Xv, y_true, over, rounds)))
        v_best = min(row, key=lambda kv: kv[1])[0]
        best[param] = v_best
        r_min = min(r for _, r in row)
        r_max = max(r for _, r in row)
        print(f"  {param:18s} opt={OPT[param]} -> best {v_best} "
              f"(min {r_min:.4f}, range {r_min:.4f}..{r_max:.4f})")
    print(f"[1d] fine sweep done in {time.time()-t0:.0f}s; config={best}")

    # fine pairwise lr x leaves (5x5) in the neighborhood
    lrs = sorted({round(v, 4) for v in np.linspace(best["learning_rate"] * 0.9,
                                                   best["learning_rate"] * 1.1, 5)})
    leaves = sorted({int(v) for v in np.linspace(best["num_leaves"] * 0.9,
                                                 best["num_leaves"] * 1.1, 5)})
    grid = []
    for lr in lrs:
        for nl in leaves:
            over = dict(best)
            over["learning_rate"], over["num_leaves"] = lr, nl
            rounds = int(best["n_estimators"])
            grid.append(((lr, nl), _fit_eval(Xf, y, Xv, y_true, over, rounds)))
    (blr, bnl), br = min(grid, key=lambda kv: kv[1])
    best["learning_rate"], best["num_leaves"] = blr, bnl
    print(f"  fine pairwise lr x leaves: best (lr={blr}, leaves={bnl}) RMSLE {br:.4f}")

    print("=" * 60)
    print(f"FINAL refined config: {best}")
    if args.full_verify:
        for name in ("A", "B"):
            XfF, yF, XvF, yTF = _load_full(name)
            r = _fit_eval(XfF, yF, XvF, yTF, best, int(best["n_estimators"]))
            ref = 1.7415 if name == "A" else 1.6861
            print(f"  fold {name} full verify: {r:.4f} (tuned ref {ref}) -> {r-ref:+.4f}")
    print("=" * 60)
    return 0


def _load_full(name: str):
    X = pd.read_parquet(f"{CACHE}/fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{CACHE}/fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{CACHE}/fold_{name}_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    return X.fillna(0.0), np.asarray(y, dtype=np.float64), \
        infer.reindex(all_users).fillna(0.0), \
        np.log1p(truth.reindex(all_users).fillna(0.0).values)


if __name__ == "__main__":
    sys.exit(main())
