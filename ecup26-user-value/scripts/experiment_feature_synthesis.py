"""T3.9.3: automated feature synthesis - residual-guided interaction search.

Mathematical (not logical) search for non-obvious feature combinations:
1. Fit base LGBM on a 300k subsample of fold B -> residual r = y_log - pred.
2. Generate products / ratios / differences of the top-40 gain features.
3. Fast filter by |corr(r, candidate)|; then score survivors by the gain of a
   single LightGBM tree (num_leaves=8) fitted on the residual.
4. Top-30 candidates -> add to the full fold B -> A/B vs 1.6881.

Dev-only. gplearn is installed but the residual-guided search is the practical
core (gplearn SymbolicRegressor is too slow on this scale).
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
CACHE = "data/folds/h"
N_SAMPLE = 300_000
TOP_FEATS = 40
CORR_THRESH = 0.02
MAX_SCORED = 400
N_ADD = 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default=CACHE)
    ap.add_argument("--n-add", type=int, default=N_ADD)
    args = ap.parse_args()
    params = bench.make_params(bench.BASE_CONFIG, bench.SEED)
    n_est = int(bench.BASE_CONFIG["n_estimators"])

    # fold B
    X = pd.read_parquet(f"{args.cache}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(f"{args.cache}/fold_B_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(f"{args.cache}/fold_B_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    Xf = X.fillna(0.0)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xf), size=min(N_SAMPLE, len(Xf)), replace=False)
    Xs, ys = Xf.iloc[idx].values, np.asarray(y, dtype=np.float64)[idx]

    # base residual
    model = lgb.train(params, lgb.Dataset(Xs, label=ys), num_boost_round=n_est)
    resid = ys - np.asarray(model.predict(Xs), dtype=np.float64)
    gain = np.asarray(model.feature_importance("gain"), dtype=np.float64)
    feats = list(Xf.columns)
    top = [f for f, _ in sorted(zip(feats, gain, strict=False), key=lambda kv: -kv[1])][:TOP_FEATS]
    top_idx = [feats.index(f) for f in top]
    Xtop = Xs[:, top_idx].astype(np.float32)
    resid32 = resid.astype(np.float32)
    print(f"[base] fold-B sample {len(Xs):,}; top-5 feats: {top[:5]}")

    # candidate generation + fast corr filter
    t0 = time.time()
    scored: list[tuple[float, str, np.ndarray, int, int, str]] = []
    n_cand = 0
    eps = 1e-3
    for i in range(TOP_FEATS):
        for j in range(i + 1, TOP_FEATS):
            for op, name, vec in (
                ("prod", f"prod_{top[i]}_{top[j]}", Xtop[:, i] * Xtop[:, j]),
                ("diff", f"diff_{top[i]}_{top[j]}", Xtop[:, i] - Xtop[:, j]),
            ):
                n_cand += 1
                c = abs(np.corrcoef(vec, resid32)[0, 1])
                if c > CORR_THRESH:
                    scored.append((c, name, vec, i, j, op))
            n_cand += 1
            vec = Xtop[:, i] / (Xtop[:, j] + eps)
            c = abs(np.corrcoef(vec, resid32)[0, 1])
            if c > CORR_THRESH:
                scored.append((c, f"ratio_{top[i]}_{top[j]}", vec, i, j, "ratio"))
    scored.sort(key=lambda kv: -kv[0])
    scored = scored[:MAX_SCORED]
    print(f"[cand] {n_cand:,} candidates; {len(scored)} pass corr>{CORR_THRESH} "
          f"({time.time()-t0:.0f}s)")

    # single-tree gain scoring on the residual
    t1 = time.time()
    results: list[tuple[float, str, int, int, str]] = []
    for _c, name, vec, _i, _j, _op in scored:
        d = lgb.Dataset(np.ascontiguousarray(vec.reshape(-1, 1)), label=resid)
        m = lgb.train({"objective": "regression", "num_leaves": 8, "learning_rate": 0.3,
                       "verbose": -1, "random_state": SEED, "num_threads": bench._default_num_threads()},
                      d, num_boost_round=1)
        results.append((float(m.feature_importance("gain")[0]), name, _i, _j, _op))
    results.sort(key=lambda kv: -kv[0])
    print(f"[score] single-tree gains ({time.time()-t1:.0f}s)")
    print("  top-15 candidates by residual gain:")
    for g, name, _i, _j, _op in results[:15]:
        print(f"    {name:60s} gain={g:,.0f}")

    # A/B: add top-N candidates to the FULL fold B
    top_n = results[: args.n_add]
    Xf_new = Xf.copy()
    Xv = infer.reindex(all_users).fillna(0.0)
    for _g, name, i, j, op in top_n:
        a, b = top[i], top[j]
        if op == "prod":
            Xf_new[name] = Xf_new[a] * Xf_new[b]
            Xv[name] = Xv[a] * Xv[b]
        elif op == "diff":
            Xf_new[name] = Xf_new[a] - Xf_new[b]
            Xv[name] = Xv[a] - Xv[b]
        else:
            Xf_new[name] = Xf_new[a] / (Xf_new[b] + eps)
            Xv[name] = Xv[a] / (Xv[b] + eps)
    model2 = lgb.train(params, lgb.Dataset(Xf_new.fillna(0.0), label=np.asarray(y, dtype=np.float64)),
                       num_boost_round=n_est)
    pred = np.asarray(model2.predict(Xv[Xf_new.columns]), dtype=np.float64)
    y_true = np.log1p(truth.reindex(all_users).fillna(0.0).values)
    r = float(np.sqrt(np.mean((pred - y_true) ** 2)))
    print("=" * 60)
    print(f"[A/B] base fold B: 1.6881 | +{args.n_add} synth feats: {r:.4f} "
          f"({r-1.6881:+.4f})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
