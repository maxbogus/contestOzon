#!/usr/bin/env python3
"""T3.10.16 - SHAP feature-importance diagnostic (fold B, production features).

Trains the production LightGBM part (blend_retuned LGBM config) on fold B, then
computes SHAP values (TreeExplainer) on a ~100k validation subsample. Outputs:

  E1: top-20 features by mean |SHAP| (with mean SHAP on positive users = direction)
  E2: two influence profiles - mean SHAP for zero-users vs whale-users (top-5% by
      truth GMV), plus the features that diverge most (input for T3.10.14 experts).

Compliance: R1 (shap MIT, lightgbm MIT), R6 (seeded), R8 (dev-only script, not
imported by run.py). Forbidden: SHAP is diagnostic ONLY - not used for feature
selection (pruning was already rejected in T3.9.1).

Usage:
  uv run python scripts/shap_diagnostic.py --output docs/results/shap_t3_2026-08-20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import ab_validate as av  # noqa: E402

CACHE_ACTOR = Path("data/folds/actor3")


def rank_features(shap_matrix, feature_names, top_k: int = 20) -> pd.DataFrame:
    """Rank features by mean |SHAP| across all rows."""
    ma = np.abs(np.asarray(shap_matrix, dtype=np.float64)).mean(axis=0)
    order = np.argsort(ma)[::-1][:top_k]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": ma[order],
        }
    )


def group_profile(shap_matrix, feature_names, mask) -> pd.DataFrame:
    """Mean SHAP per feature for the selected user group."""
    sv = np.asarray(shap_matrix, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    return pd.DataFrame({"feature": feature_names, "mean_shap": sv[m].mean(axis=0)})


def divergence(prof_a: pd.DataFrame, prof_b: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Features where the two group profiles differ most (|mean_b - mean_a|)."""
    d = prof_a.merge(prof_b, on="feature", suffixes=("_zeros", "_whales"))
    d["diff"] = d["mean_shap_whales"] - d["mean_shap_zeros"]
    return d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(top_k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="B")
    ap.add_argument("--cache", type=Path, default=CACHE_ACTOR)
    ap.add_argument("--n-sub", type=int, default=100_000, help="SHAP subsample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--whale-frac", type=float, default=0.05)
    ap.add_argument("--output", type=Path, default=Path("docs/results/shap_t3_2026-08-20"))
    args = ap.parse_args()

    os.environ.setdefault("ECUP26_NUM_THREADS", "4")
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading fold {args.fold} from {args.cache} ...")
    X, y, Xv, yt_log = av._load(args.fold, args.cache)
    print(f"train {X.shape}, infer {Xv.shape}")

    model = av._train_lgb(
        X, y, {"n_estimators": 650, "num_leaves": 120, "learning_rate": 0.02,
                "feature_fraction": 0.6, "bagging_fraction": 0.8, "min_child_samples": 21},
        args.seed,
    )
    feature_names = list(X.columns)
    del X

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xv), size=min(args.n_sub, len(Xv)), replace=False)
    X_sub = Xv.iloc[idx].fillna(0.0)
    yt_log_sub = np.asarray(yt_log)[idx]
    y_raw_sub = np.clip(np.expm1(yt_log_sub), 0.0, None)

    print("computing SHAP (TreeExplainer) ...")
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    print(f"shap_values shape: {shap_values.shape}")

    # E1: top-k features by mean |SHAP|.
    top = rank_features(shap_values, feature_names, args.top_k)
    print(f"\nE1: top-{args.top_k} features by mean |SHAP|:")
    print(top.to_string(index=False))

    # E2: zero-users vs whale-users influence profiles.
    is_zero = y_raw_sub == 0
    whale_thr = float(np.quantile(y_raw_sub, 1.0 - args.whale_frac))
    is_whale = y_raw_sub >= whale_thr
    print(
        f"\nE2: zeros={int(is_zero.sum())} ({is_zero.mean():.3f}), "
        f"whales(top-{args.whale_frac:.0%} by truth)={int(is_whale.sum())}"
    )
    prof_zero = group_profile(shap_values, feature_names, is_zero)
    prof_whale = group_profile(shap_values, feature_names, is_whale)
    div = divergence(prof_zero, prof_whale, args.top_k)
    print("\nfeatures with the largest zeros-vs-whales SHAP divergence (whale - zero):")
    print(div.to_string(index=False))

    top.to_csv(out_dir / "top_features.csv", index=False)
    prof_zero.to_csv(out_dir / "profile_zeros.csv", index=False)
    prof_whale.to_csv(out_dir / "profile_whales.csv", index=False)
    div.to_csv(out_dir / "divergence_zeros_vs_whales.csv", index=False)
    print(f"\nCSVs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
