#!/usr/bin/env python3
"""CLI for T3.6.12 ablation matrix.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lifetimes, lightgbm, scikit-learn - all Apache 2.0/BSD)
- R4: no internet at runtime
- R6: reproducible (seeded)
- R7: CPU tabular models
- R8: writes docs/results/ablation_v1.csv, NOT submit.csv; lightgbm in dev deps

Usage:
    uv run python scripts/ablation_matrix.py --synthetic --folds 2
    uv run python scripts/ablation_matrix.py --data path/to/train.parquet --folds 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ on path when run as standalone script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecup26_user_value.ablation.runner import run_ablation  # noqa: E402

SEED = 42

# Active models for v2 (T3.6.18). lgbm_ziln is now a 3-sub-model composite.
ACTIVE_MODELS: tuple[str, ...] = ("btyd_global", "btyd_segmented", "lgbm_mse", "lgbm_ziln")


def _make_synthetic(n_users: int = 60) -> pd.DataFrame:
    """Build a synthetic transaction df for quick testing."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        n_orders = int(rng.integers(3, 12)) if seg == "search" else int(rng.integers(1, 5))
        for _o in range(n_orders):
            price = float(rng.uniform(5, 30)) if seg == "search" else float(rng.uniform(80, 250))
            rows.append(
                {
                    "user_id": uid,
                    "event_date": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 30))),
                    "type": seg,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


def _holdout_target(df: pd.DataFrame, holdout_days: int = 30) -> pd.Series:
    cutoff = df["event_date"].max() - pd.Timedelta(days=holdout_days)
    return (
        df[df["event_date"] > cutoff]
        .groupby("user_id")["price"]
        .sum()
        .reindex(df["user_id"].unique())
        .fillna(0.0)
    )


def _load_real(path: Path) -> pd.DataFrame:
    """Load real train.parquet. Schema per docs/tasks/task3_user_value_data.md.

    Required columns after mapping: user_id, event_date, type, price.
    Source columns to map:
        user_id -> user_id
        event_date -> event_date
        type -> derived from 'search'/'cat' flags (1 if search>0 else 'cat')
        price -> 'gmv' if available else 0.0
    """
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    # Adapt to expected schema
    if "type" not in df.columns and "search" in df.columns and "cat" in df.columns:  # noqa: SIM102
        df["type"] = np.where(df["search"] > 0, "search", "cat")
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"]
    keep = ["user_id", "event_date", "type", "price"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; have {df.columns.tolist()}")
    return df[keep]


def _summarize(result: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RMSLE across folds, mean +/- std."""
    summary = (
        result.groupby(["model", "features"])["rmsle"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rmsle_mean", "std": "rmsle_std", "count": "n_folds"})
    )
    return summary.sort_values("rmsle_mean")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (default: True if --data not provided)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to train.parquet (alternative to --synthetic)",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "results" / "ablation_v1.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "docs" / "results" / "ablation_v1_summary.csv",
    )
    args = parser.parse_args()

    if args.data:
        df = _load_real(args.data)
    else:
        print("[ablation] No --data provided, using synthetic (n_users=60)")
        df = _make_synthetic(n_users=60)

    target = _holdout_target(df, holdout_days=args.holdout_days)
    print(
        f"[ablation] df rows={len(df)}, unique users={df['user_id'].nunique()}, folds={args.folds}"
    )

    result = run_ablation(
        df=df,
        target=target,
        folds=args.folds,
        holdout_days=args.holdout_days,
        model_names=list(ACTIVE_MODELS),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"[ablation] Wrote {len(result)} rows -> {args.output}")

    summary = _summarize(result)
    summary.to_csv(args.summary_output, index=False)
    print(
        f"[ablation] Wrote summary ({len(summary)} model-feature combos) -> {args.summary_output}"
    )
    print()
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
