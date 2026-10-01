#!/usr/bin/env python3
"""T3.14/T3.15: fast feature-subset A/B on cached folds (NO fold rebuild).

The cached fold train matrices (data/folds/actor3_calv3) hold all 180 features;
feature-selection configs are just column subsets, so each LGBM fit is ~1-2 min
instead of a ~10 min fold rebuild per config. Screens configs on fold B, then
promotes the winner to A/P/G.

Configs (feature groups, see build_features.py env gates):
  full       = 180 (base+actor+calv2 + behavioral14 + norm6)
  no_beh     = 166 (drop behavioral gift-timing 14)
  no_norm    = 174 (drop history-normalization 6)
  no_both    = 160 (pure calv2 = the LB winner's set)
  topK       = top-K columns by LightGBM gain importance (fold-B fit)

Usage:
  uv run python scripts/fold_feature_ab.py [--cache data/folds/actor3_calv3] [--folds B A P G]
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

BEHAVIORAL = [
    "flag_adv_buyer_2025", "flag_late_buyer_2025", "flag_lm_buyer_2025",
    "gift_vs_mass_2025", "gmv_adv_feb23_2025", "gmv_adv_mar8_2025",
    "gmv_late_2025", "gmv_lm_feb23_2025", "gmv_lm_mar8_2025",
    "gmv_pre_ny_2025", "gmv_pre_sep1_2025", "lastmin_share_2025",
    "late_share_2025", "planner_share_2025",
]
NORM = [
    "orders_per_active_day", "gmv_per_active_day", "activity_rate",
    "gmv_per_day_alive", "orders_per_day_alive", "recency_rate",
]


def _load_fold(cache: Path, name: str):
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


def _fit_eval(X, y, Xv, yt, cols: list[str], top: int = 0) -> tuple[float, float]:
    """Train on `cols` (or top-K by gain if top>0), return (rmsle, pred_mean)."""
    use = cols[:]
    if top:
        m = lgb.train(
            bench.make_params(bench.BASE_CONFIG, 42),
            lgb.Dataset(X[use], label=np.asarray(y)),
            num_boost_round=int(bench.BASE_CONFIG["n_estimators"]),
        )
        imp = m.feature_importance("gain")
        use = [c for _, c in sorted(zip(imp, use), reverse=True)[:top]]
    m = lgb.train(
        bench.make_params(bench.BASE_CONFIG, 42),
        lgb.Dataset(X[use], label=np.asarray(y)),
        num_boost_round=int(bench.BASE_CONFIG["n_estimators"]),
    )
    p = np.expm1(np.asarray(m.predict(Xv[use])))
    return _rmsle(yt, p), float(p.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=str, default="data/folds/actor3_calv3")
    ap.add_argument("--folds", type=str, default="B")
    args = ap.parse_args()
    cache = Path(args.cache)
    all_cols = [c for c in pd.read_parquet(cache / "fold_B_train.parquet").columns
                if c not in ("user_id", "origin", "y_log")]
    beh = [c for c in all_cols if c in BEHAVIORAL]
    norm = [c for c in all_cols if c in NORM]
    base = [c for c in all_cols if c not in beh and c not in norm]
    configs = {
        "full180": all_cols,
        "no_beh166": base + norm,
        "no_norm174": base + beh,
        "no_both160": base,
        "top100": all_cols,   # top=100 by gain
        "top120": all_cols,   # top=120 by gain
    }
    for name in args.folds.split(","):
        X, y, Xv, yt = _load_fold(cache, name)
        for cfg, cols in configs.items():
            top = 100 if cfg == "top100" else (120 if cfg == "top120" else 0)
            r, pm = _fit_eval(X, y, Xv, yt, cols, top=top)
            print(f"fold {name} [{cfg:10s} n={len(cols) if not top else top}]: RMSLE={r:.4f} mean={pm:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
