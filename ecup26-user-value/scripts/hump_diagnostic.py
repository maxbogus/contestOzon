#!/usr/bin/env python3
"""T3.17: hump diagnostic - weekly GMV aggregate to verify the seasonality.

The user's mechanics hypothesis (from Ozon financials):
  Q1'25 x1 -> Q2'25 x1.25 -> Q3'25 x1.33 -> Q4'25 x2 (peak) ->
  Q1'26 first half decline -> Q1'26 second half (test window) = gift hump.

This script prints weekly GMV (normalized to the first full week) + quarterly
totals, so we can SEE whether the hump/peak structure exists in the training
data and confirm the walk-forward regime-gap (no Feb-Mar 2026 label).

Usage: uv run python scripts/hump_diagnostic.py [--weeks 8]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts import benchmark_t3 as bench  # noqa: E402


def main() -> int:
    df = bench._load_clean(Path("data/train.parquet"))
    df["event_date"] = pd.to_datetime(df["event_date"])
    total_gmv = float(df["price"].sum())
    # weekly aggregate (Mon-start)
    wk = df.set_index("event_date").resample("W-MON")["price"].sum()
    wk.index.name = "week_start"
    first_full = wk[wk.index >= "2025-01-06"].iloc[0] if (wk.index >= "2025-01-06").any() else wk.iloc[0]
    norm = wk / first_full

    # quarterly totals
    q = df.assign(q=df["event_date"].dt.to_period("Q")).groupby("q")["price"].sum()
    q1_25 = float(q.get("2025Q1", np.nan))
    print("=== Quarterly GMV (normalized to Q1'25 = x1) ===")
    for period, v in q.items():
        mult = v / q1_25 if q1_25 else float("nan")
        print(f"  {period}: {v/1e12:.3f} трлн | x{mult:.2f}")

    print(f"\nTotal GMV in train (2025-01-01..2026-02-13): {total_gmv/1e12:.3f} трлн")
    print("Test window is 14.02-15.03.2026 (NOT in train - that's the target).")

    print("\n=== Weekly GMV (normalized, first full week 2025-01-06 = x1) ===")
    # print around quarter boundaries + the 2025 gift window
    print("  week_start  mult   (window marker)")
    markers = {
        (2025, 2, 14): "2025 gift window starts (Valentine)",
        (2025, 2, 23): "Feb-23 gift",
        (2025, 3, 8): "Mar-8 gift",
        (2025, 11, 3): "Nov promos (Q4 peak)",
        (2025, 12, 15): "New Year",
        (2026, 1, 12): "Q1'26 decline",
    }
    for ts, mult in norm.items():
        mark = ""
        for (y, m, d), label in markers.items():
            if ts <= pd.Timestamp(y, m, d) < ts + pd.Timedelta(days=7):
                mark = f"  <- {label}"
                break
        print(f"  {ts.date()}  {mult:.2f}{mark}")

    out = Path("docs/results/hump_weekly.csv")
    pd.DataFrame({"week_start": wk.index, "gmv": wk.values,
                  "norm_to_q1_25_week": norm.values}).to_csv(out, index=False)
    print(f"\nSaved weekly table -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
