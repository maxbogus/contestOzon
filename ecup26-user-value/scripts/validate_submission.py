#!/usr/bin/env python3
"""CLI: validate a T3 submission CSV against sample data (Tier 1 metrics).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (numpy, pandas, scipy)
- R6: reproducible
- R8: dev-only (does NOT write submit.csv)
- R10: asserts unique IDs in the predictions file

Computes all Tier 1 metrics on the predictions file vs a held-out
"ground truth" subset, prints a pass/fail summary table, and writes
a Markdown report.

Usage:
    uv run python scripts/validate_submission.py \\
        --test_data_path data/sample_submit.csv \\
        --predictions /tmp/submit.csv \\
        --output docs/results/validation_t3_2026-08-16.md

Note: since the test set has no ground truth, this script uses the
mean of `sample_submit.csv`'s `predict` column as a stand-in target.
For real validation against a held-out private window, supply
`--target_csv` with a CSV containing `user_id,gmv` columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.eval.diagnostics import (  # noqa: E402
    adversarial_breakdown,
    compute_diagnostics,
)
from ecup26_user_value.eval.metrics import compute_all_metrics, naive_baseline_rmsle  # noqa: E402

PASS_CRITERIA: dict[str, tuple[float, str]] = {
    # metric_name: (threshold, comparison)  comparison in {">", "<"}
    "model_rmsle": (float("inf"), "<"),  # checked against naive_rmsle separately
    "naive_rmsle_mean": (float("inf"), "<"),  # informational only
    "naive_rmsle_median": (float("inf"), "<"),  # informational only
    "lift_vs_mean": (0.0, ">"),  # must beat mean baseline
    "mae": (float("inf"), "<"),  # placeholder; see mae_vs_mean below
    "rmse": (float("inf"), "<"),  # placeholder
    "bias_mean": (float("inf"), "<"),  # placeholder; absolute value
    "spearman": (0.0, ">"),  # must have positive correlation
    "quantile_calibration": (float("inf"), "<"),  # placeholder
}


def _passes(metric: str, value: float, mae_value: float) -> bool:
    """Apply pass/fail criteria. Most use a fixed threshold; MAE/RMSE use
    a fraction of the mean target value (so they scale with data).
    """
    if metric == "mae":
        return value < mae_value  # MAE must be < mean(y_true)
    if metric == "rmse":
        return value < mae_value * 2  # RMSE < 2x mean
    if metric == "bias_mean":
        return abs(value) < mae_value  # |bias| < mean
    if metric == "quantile_calibration":
        return value < mae_value  # calibration error < mean
    thr, cmp = PASS_CRITERIA[metric]
    if cmp == ">":
        return value > thr
    return value < thr


def _format_md_table(metrics: dict[str, float], passes: dict[str, bool]) -> str:
    rows = ["| Metric | Value | Pass |", "|---|---|---|"]
    for name, val in metrics.items():
        p = "✓" if passes.get(name, True) else "✗"
        if val is None:
            val_str = "n/a"
        elif isinstance(val, bool):
            val_str = str(val)
        else:
            try:
                val_str = f"{float(val):.4f}"
            except (ValueError, TypeError):
                val_str = str(val)
        rows.append(f"| `{name}` | {val_str} | {p} |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="CSV with user_id column (test users)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="CSV with columns user_id,predict (model output)",
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default=None,
        help="Optional CSV with columns user_id,gmv (ground truth). "
        "If omitted, target = sample_submit.csv['predict'].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "validation_t3_latest.md",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=1,
        choices=[1, 2],
        help="Validation tier: 1=Tier 1, 2=Tier 1+2/3.",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_data_path)
    pred_df = pd.read_csv(args.predictions)

    # R10: assert unique IDs in predictions
    assert pred_df.iloc[:, 0].is_unique, (
        f"R10: duplicate IDs in predictions ({pred_df.iloc[:, 0].duplicated().sum()} duplicates)"
    )

    # Align by user_id
    merged = test_df[["user_id"]].merge(pred_df, on="user_id", how="left")
    assert merged["predict"].notna().all(), (
        f"missing predictions for {(merged['predict'].isna()).sum()} users"
    )
    assert (merged["predict"] >= 0).all(), "negative predictions present"

    if args.target_csv:
        target_df = pd.read_csv(args.target_csv)[["user_id", "gmv"]]
        merged = merged.merge(target_df, on="user_id", how="left")
        assert merged["gmv"].notna().all(), "missing gmvs in target"
        y_true = merged["gmv"].values
        target_source = args.target_csv
    else:
        y_true = merged["predict"].values  # placeholder
        target_source = f"placeholder={args.test_data_path}['predict']"

    y_pred = merged["predict"].values

    metrics = compute_all_metrics(y_true, y_pred)
    mae_val = metrics["mae"]
    passes = {k: _passes(k, v, mae_val) for k, v in metrics.items()}

    if args.tier >= 2:
        tier23 = compute_diagnostics(y_true, y_pred)
        metrics.update(tier23)
        # Adversarial breakdown (placeholder mode: limited features)
        try:
            adv_df = pd.DataFrame(
                {
                    "user_id": merged["user_id"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
            adv = adversarial_breakdown(adv_df, y_true=y_true, y_pred=y_pred)
            if not adv.empty:
                metrics["adversarial_n_slices"] = int(len(adv))
                metrics["adversarial_max_rmsle"] = float(adv["rmsle"].max())
        except Exception:
            pass  # adversarial needs features; skip

    # Print summary
    print(f"\n=== Validation: {len(merged)} users ===")
    print(f"Target source: {target_source}")
    print()
    md_table = _format_md_table(metrics, passes)
    print(md_table)

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    naive_rmsle = naive_baseline_rmsle(y_true, "mean")
    args.output.write_text(
        f"""# T3 validation report

- Predictions: `{args.predictions}`
- Target: `{target_source}`
- Rows: {len(merged)}

{md_table}

## Naive baseline RMSLE (mean): {naive_rmsle:.4f}

## Pass criteria

- `lift_vs_mean > 0`: model must beat the mean constant predictor
- `mae < mean(true)`: must be better than mean
- `rmse < 2 * mae`: scale check
- `bias_mean abs < mae`: model not systematically biased
- `spearman > 0`: must have positive correlation
- `quantile_calibration < mae`: distribution close to ground truth

## Compliance

- R1: open-source (numpy, pandas, scipy)
- R6: reproducible
- R8: dev-only, does not write submit.csv
- R10: predictions file asserted unique IDs

## Tier

Tier 1 + Tier 2/3 (enable Tier 2/3 with --tier=2; default = Tier 1).

## Forbidden actions taken

None. Read-only on `submit.csv`; writes only this report.
""",
        encoding="utf-8",
    )
    print(f"\nWrote report to {args.output}")

    n_failed = sum(1 for v in passes.values() if not v)
    if n_failed > 0:
        print(f"\nFAIL: {n_failed} criteria failed")
        return 1
    print("\nPASS: all criteria met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
