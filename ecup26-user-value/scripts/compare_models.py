#!/usr/bin/env python3
"""Compare baseline vs segmented BTYD on real held-out data (T3.7.2).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lifetimes, our own code)
- R6: reproducible (seeded synthetic, deterministic BTYD)
- R8: NOT imported by run.py (dev-only comparison tool)
- R10: no submit.csv writes

Splits train.parquet chronologically (last 30 days as holdout), trains
both models on the train portion, predicts for each user_id with activity
in the holdout window, and computes all Tier 1 + Tier 2/3 metrics.

Usage:
    uv run python scripts/compare_models.py \\
        --data data/train.parquet \\
        --output docs/results/compare_baseline_vs_segmented_<date>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.eval.diagnostics import (  # noqa: E402
    adversarial_breakdown,
    compute_diagnostics,
)
from ecup26_user_value.eval.metrics import compute_all_metrics  # noqa: E402
from ecup26_user_value.models.btyd_baseline import BTYDBaseline  # noqa: E402
from ecup26_user_value.models.btyd_segmented import SegmentedBTYD  # noqa: E402

# Holdout window: last 30 days of train data (simulates E-CUP private split).
HOLDOUT_DAYS = 30
SEED = 42


def _load_transactions(path: Path) -> pd.DataFrame:
    """Load train parquet, adapt to BTYD schema."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "type" not in df.columns and "search" in df.columns and "cat" in df.columns:  # noqa: SIM102
        df["type"] = np.where(df["search"] > 0, "search", "cat")
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"].astype(float)
    keep = ["user_id", "event_date", "type", "price"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}")
    return df[keep]


def _train_predict(
    model_name: str,
    train_df: pd.DataFrame,
    val_users: np.ndarray,
) -> np.ndarray:
    """Train the named model on train_df, predict per-user for val_users."""
    if model_name == "btyd_global":
        model = BTYDBaseline(penalizer_coef=0.01, time_period_days=30).fit(
            train_df,
            customer_id_col="user_id",
            order_date_col="event_date",
        )
        pred_all = model.predict(
            train_df,
            customer_id_col="user_id",
            order_date_col="event_date",
        )
        # Aggregate per-user from per-transaction predictions
        s = pd.Series(pred_all, index=train_df["user_id"].values)
        per_user = s.groupby(level=0).sum().clip(lower=0.0)
    elif model_name == "btyd_segmented":
        model = SegmentedBTYD(penalizer_coef=0.01, time_period_days=30).fit(
            train_df,
            order_date_col="event_date",
        )
        pred_all = model.predict(train_df, order_date_col="event_date")
        s = pd.Series(pred_all, index=train_df["user_id"].values)
        per_user = s.groupby(level=0).sum().clip(lower=0.0)
    else:
        raise ValueError(f"unknown model {model_name!r}")

    return per_user.reindex(val_users).fillna(0.0).values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "compare_baseline_vs_segmented_latest.md",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=HOLDOUT_DAYS,
        help="Length of held-out window in days (default 30).",
    )
    args = parser.parse_args()

    print(f"[compare] Loading {args.data}...")
    df = _load_transactions(args.data)
    print(f"[compare] {len(df)} transactions, {df['user_id'].nunique()} users")

    # Time-based split: train on first N - holdout_days, validate on last holdout_days
    max_date = df["event_date"].max()
    cutoff = max_date - pd.Timedelta(days=args.holdout_days)
    train_df = df[df["event_date"] <= cutoff].copy()
    val_df = df[(df["event_date"] > cutoff) & (df["event_date"] <= max_date)].copy()
    print(f"[compare] Train: {len(train_df)} rows (<= {cutoff})")
    print(f"[compare] Val:   {len(val_df)} rows ({cutoff} .. {max_date})")

    # Ground truth: sum(price) per user in holdout window
    y_true_series = val_df.groupby("user_id")["price"].sum()
    val_users = y_true_series.index.values
    y_true = y_true_series.values
    print(f"[compare] {len(val_users)} users with activity in holdout")

    # Train both models
    results: dict[str, dict[str, float]] = {}
    for model_name in ("btyd_global", "btyd_segmented"):
        print(f"[compare] Training {model_name}...")
        y_pred = _train_predict(model_name, train_df, val_users)
        tier1 = compute_all_metrics(y_true, y_pred)
        tier23 = compute_diagnostics(y_true, y_pred, n_boot=200)
        all_metrics = {**tier1, **tier23}
        # Adversarial breakdown
        adv = adversarial_breakdown(
            pd.DataFrame({"user_id": val_users, "y_true": y_true, "y_pred": y_pred}),
            y_true=y_true,
            y_pred=y_pred,
        )
        all_metrics["adversarial_n_slices"] = int(len(adv))
        if len(adv) > 0:
            all_metrics["adversarial_max_rmsle"] = float(adv["rmsle"].max())
        results[model_name] = all_metrics

    # Comparison table
    print(f"\n=== Comparison ({len(val_users)} users) ===")
    keys_to_show = [
        "model_rmsle",
        "naive_rmsle_mean",
        "lift_vs_mean",
        "mae",
        "rmse",
        "bias_mean",
        "spearman",
        "quantile_calibration",
        "coverage_q10_q90",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "adversarial_max_rmsle",
    ]
    header = "| Metric | " + " | ".join(results.keys()) + " |"
    sep = "|---|" + "|".join(["---"] * len(results)) + "|"
    rows = [header, sep]
    for k in keys_to_show:
        cells = []
        for m in results:
            v = results[m].get(k)
            if v is None:
                cells.append("n/a")
            elif isinstance(v, bool):
                cells.append(str(v))
            else:
                try:
                    cells.append(f"{float(v):.4f}")
                except (ValueError, TypeError):
                    cells.append(str(v))
        rows.append(f"| `{k}` | " + " | ".join(cells) + " |")
    table = "\n".join(rows)
    print(table)

    # Winner recommendation
    rmsle_global = results["btyd_global"]["model_rmsle"]
    rmsle_segmented = results["btyd_segmented"]["model_rmsle"]
    if rmsle_segmented < rmsle_global:
        improvement = (rmsle_global - rmsle_segmented) / rmsle_global * 100
        winner = "btyd_segmented"
        recommendation = f"SWITCH to SegmentedBTYD: {improvement:.1f}% RMSLE improvement."
    elif rmsle_global < rmsle_segmented:
        improvement = (rmsle_segmented - rmsle_global) / rmsle_global * 100
        winner = "btyd_global"
        recommendation = f"KEEP btyd_global: SegmentedBTYD is {improvement:.1f}% WORSE."
    else:
        winner = "tie"
        recommendation = "TIE: identical RMSLE."

    print(f"\n[compare] Winner: {winner}")
    print(f"[compare] {recommendation}")

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""# Baseline vs Segmented BTYD comparison ({pd.Timestamp.now().date()})

## Setup

- Data: `{args.data}`
- Train: {len(train_df)} rows (<= {cutoff})
- Val: {len(val_df)} rows ({cutoff} .. {max_date}, {args.holdout_days} days)
- Users with holdout activity: {len(val_users)}
- Target: sum(price) per user in holdout window
- Models:
  - **btyd_global**: BTYDBaseline (single global model)
  - **btyd_segmented**: SegmentedBTYD (per-search-vs-cat + global fallback)

## Results

{table}

## Recommendation

**{recommendation}**

Winner: `{winner}`

## Notes

- Holdout window simulates E-CUP private split (last 30 days).
- This is **not** the actual leaderboard - the real test set
  (2026-02-14 .. 2026-03-15) is held by organizers.
- Tier 1 metrics: model_rmsle, naive_rmsle, lift, mae, rmse,
  bias_mean, spearman, quantile_calibration.
- Tier 2/3 metrics: coverage, PIT, bootstrap CI, adversarial.

## Compliance

- R1: open-source (lifetimes Apache 2.0)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py
- R10: no submit.csv writes
""",
        encoding="utf-8",
    )
    print(f"\n[compare] Wrote report to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
