"""GMV distribution EDA for E-CUP T3 user-value prediction.

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible) and
ticket T3.6.13.

RMSLE is sensitive to outliers. This EDA characterizes the GMV distribution
before modeling. Expected (per ticket):
- ~45.9% zeros (zero-inflation)
- Heavy right tail (log-normal-like)
- Outliers > 99.9th percentile

Provides:
- compute_gmv_distribution(gmv_series) -> dict with stats
- save_report(report, path) -> JSON file
- main() CLI for full dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: Percentiles to report (per ticket spec).
EDA_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0, 99.9)


def compute_gmv_distribution(gmv: pd.Series) -> dict[str, Any]:
    """Compute GMV distribution statistics.

    Args:
        gmv: pandas Series of GMV values (per user).

    Returns:
        Dict with n_total, n_zero, percent_zero, percentiles, outlier_count.
    """
    gmv_clean = gmv.dropna()
    if gmv_clean.empty:
        return {
            "n_total": 0,
            "n_zero": 0,
            "percent_zero": 0.0,
            "percentiles": {str(p): 0.0 for p in EDA_PERCENTILES},
            "outlier_count": 0,
        }

    n_total = len(gmv_clean)
    n_zero = int((gmv_clean == 0).sum())
    percent_zero = n_zero / n_total

    percentiles = {str(p): float(gmv_clean.quantile(p / 100.0)) for p in EDA_PERCENTILES}

    # Outliers: values > 99.9th percentile
    p99_9 = percentiles["99.9"]
    outlier_count = int((gmv_clean > p99_9).sum())

    return {
        "n_total": n_total,
        "n_zero": n_zero,
        "percent_zero": percent_zero,
        "percentiles": percentiles,
        "outlier_count": outlier_count,
    }


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    """Save EDA report to JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out


def main() -> None:
    """CLI: run on train.parquet gmv column."""
    import argparse

    parser = argparse.ArgumentParser(description="GMV distribution EDA")
    parser.add_argument("--input", default="data/train.parquet", help="Path to train.parquet")
    parser.add_argument("--output", default="reports/eda_gmv.json", help="Output JSON path")
    parser.add_argument("--gmv-col", default="gmv", help="GMV column name (per-user aggregate)")
    args = parser.parse_args()

    import polars as pl

    print(f"Loading {args.input}...")
    df = pl.read_parquet(args.input)
    # Aggregate GMV per user (sum over days)
    user_gmv = df.group_by("user_id").agg(pl.col(args.gmv_col).sum().alias("gmv_total"))
    gmv_series = user_gmv["gmv_total"].to_pandas()

    print(f"Computing distribution on {len(gmv_series)} users...")
    report = compute_gmv_distribution(gmv_series)

    save_report(report, args.output)

    print("\n=== GMV Distribution ===")
    print(f"Total users: {report['n_total']}")
    print(f"Zero users: {report['n_zero']} ({report['percent_zero']:.2%})")
    print("\nPercentiles:")
    for p, v in report["percentiles"].items():
        print(f"  {p}%: {v:.2f}")
    print(f"\nOutliers (> 99.9%): {report['outlier_count']}")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
