"""Tier 2/3 diagnostics for E-CUP T3 user-value predictions (T3.7.2).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (stdlib + numpy + pandas + scipy; NO matplotlib)
- R6: reproducible (seeded bootstrap, deterministic math)
- R8: NOT imported by run.py (dev-only evaluation tool)
- R10: no submit.csv writes

Provides Tier 2 metrics:
- rmse (re-export from metrics)
- percentiles_alignment(y_true, y_pred, q) -> dict
- coverage_at_quantile(y_true, y_pred, q) -> float
- stability_seed_variance(fit_predict_fn, df, n_seeds) -> dict

Provides Tier 3 metrics:
- pit_histogram(y_true, y_pred, n_bins) -> (str, np.ndarray) - ASCII histogram
- adversarial_breakdown(df_with_features) -> pd.DataFrame
- bootstrap_ci(y_true, y_pred, n_boot, alpha) -> (low, high)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ecup26_user_value.eval.metrics import rmse as _rmse

logger = logging.getLogger(__name__)


# Re-export rmse for unified diagnostic API
rmse = _rmse


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE with the T3 convention: log1p, max(0, pred)."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def percentiles_alignment(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, float]:
    """Side-by-side quantile values for predictions and ground truth.

    Returns a dict like {"q10_pred": ..., "q10_true": ..., "q50_pred": ..., ...}.

    Useful for detecting calibration drift that mean-based metrics hide.
    A well-calibrated model has pred_q == true_q for every quantile.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    result: dict[str, float] = {}
    for quantile in q:
        key = f"q{int(round(quantile * 100))}"
        result[f"{key}_true"] = float(np.quantile(y_true, quantile))
        result[f"{key}_pred"] = float(np.quantile(y_pred, quantile))
    return result


def coverage_at_quantile(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    q: tuple[float, float] = (0.1, 0.9),
) -> float:
    """Fraction of users whose pred and true both fall in the same [q_lo, q_hi] band.

    Useful as a "sharpness" metric: a well-calibrated model has high coverage.
    Returns a value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    q_lo, q_hi = q
    lo = float(np.quantile(y_true, q_lo))
    hi = float(np.quantile(y_true, q_hi))
    if hi <= lo:
        return 0.0
    in_band_true = (y_true >= lo) & (y_true <= hi)
    in_band_pred = (y_pred >= lo) & (y_pred <= hi)
    # Coverage = fraction where BOTH are in band
    overlap = in_band_true & in_band_pred
    return float(overlap.sum() / y_true.size)


def stability_seed_variance(
    fit_predict_fn: Callable[[pd.DataFrame | None, int], np.ndarray],
    df: pd.DataFrame | None,
    n_seeds: int = 5,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Run fit_predict_fn with N different seeds, return RMSLE mean/std.

    Args:
        fit_predict_fn: callable (df, seed) -> predictions array. Must accept seed.
        df: passed through to fit_predict_fn as first arg.
        n_seeds: number of distinct seeds to run.
        y_true: ground truth. If None, fit_predict_fn is assumed to return
            predictions of length equal to the first call's output; we then
            compute RMSLE against the predictions themselves (useful only
            for measuring variance, not absolute quality).

    Returns:
        dict with "rmsle_mean", "rmsle_std", "n_seeds", "rmsle_values".
    """
    rmsle_values: list[float] = []
    for seed in range(n_seeds):
        pred = fit_predict_fn(df, seed)
        pred_arr = np.asarray(pred, dtype=np.float64)
        if y_true is not None and len(y_true) == len(pred_arr):
            rmsle_values.append(_rmsle(y_true, pred_arr))
        # If y_true is None or lengths mismatch, skip RMSLE for this seed.
    if not rmsle_values:
        return {
            "rmsle_mean": float("nan"),
            "rmsle_std": float("nan"),
            "n_seeds": 0,
            "rmsle_values": [],
        }
    return {
        "rmsle_mean": float(np.mean(rmsle_values)),
        "rmsle_std": float(np.std(rmsle_values)),
        "n_seeds": len(rmsle_values),
        "rmsle_values": rmsle_values,
    }


def pit_histogram(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> tuple[str, np.ndarray]:
    """Probability Integral Transform histogram (ASCII rendering).

    For each observation, compute P(true <= pred) using the empirical CDF of pred.
    Under perfect calibration, these PIT values are uniformly distributed on [0, 1].
    Returns (ascii_text, bin_counts).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return "", np.zeros(n_bins, dtype=int)

    # Compute PIT values: for each true value, fraction of pred <= true
    sorted_pred = np.sort(y_pred)
    pit_values = np.searchsorted(sorted_pred, y_true, side="right") / len(sorted_pred)
    pit_values = np.clip(pit_values, 0.0, 1.0)

    counts, _ = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))

    # ASCII rendering: max bar = 40 chars
    max_count = max(counts.max(), 1)
    lines: list[str] = []
    bin_width = 1.0 / n_bins
    for i, c in enumerate(counts):
        lo_pct = int(100 * i * bin_width)
        hi_pct = int(100 * (i + 1) * bin_width)
        bar_len = int(round(40 * c / max_count))
        bar = "*" * bar_len
        lines.append(f"[{lo_pct:3d}-{hi_pct:3d}%] | {bar} {c}")
    text = "\n".join(lines)
    return text, counts


def adversarial_breakdown(
    df: pd.DataFrame,
    y_true: np.ndarray | pd.Series | None = None,
    y_pred: np.ndarray | pd.Series | None = None,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    n_transactions_col: str = "n_transactions",
    last_active_col: str = "last_active_day",
    gmv_total_col: str = "gmv_total",
) -> pd.DataFrame:
    """Per-slice RMSLE breakdown for adversarial sub-populations.

    Slices:
    - cold-start: users with very few transactions (< median / 2)
    - churned: users with no recent activity (> 90 days since last_active)
    - whales: top 1% by gmv_total
    - mainstream: everyone else

    If y_true/y_pred are not given, expects columns y_true_col and y_pred_col
    in df. Either way, returns a DataFrame with columns: slice, n_users, rmsle.
    """
    df = df.copy()
    if y_true is None:
        if y_true_col not in df.columns:
            raise ValueError(f"missing {y_true_col} in df")
        y_true = df[y_true_col].values
    if y_pred is None:
        if y_pred_col not in df.columns:
            raise ValueError(f"missing {y_pred_col} in df")
        y_pred = df[y_pred_col].values

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["slice", "n_users", "rmsle"])

    rows: list[dict[str, Any]] = []

    # Cold-start: n_transactions in lower half of population
    if n_transactions_col in df.columns:
        median_n = float(np.median(df[n_transactions_col]))
        cold_mask = df[n_transactions_col] <= median_n / 2
        if cold_mask.any():
            rows.append(
                {
                    "slice": "cold-start",
                    "n_users": int(cold_mask.sum()),
                    "rmsle": _rmsle(
                        np.asarray(y_true)[cold_mask.values], np.asarray(y_pred)[cold_mask.values]
                    ),
                }
            )

    # Churned: last_active > 270 days from end
    if last_active_col in df.columns:
        max_active = float(df[last_active_col].max())
        churn_threshold = max_active - 270  # 9 months inactive
        churn_mask = df[last_active_col] <= churn_threshold
        if churn_mask.any():
            rows.append(
                {
                    "slice": "churned",
                    "n_users": int(churn_mask.sum()),
                    "rmsle": _rmsle(
                        np.asarray(y_true)[churn_mask.values], np.asarray(y_pred)[churn_mask.values]
                    ),
                }
            )

    # Whales: top 1% by gmv_total
    if gmv_total_col in df.columns:
        whale_threshold = float(np.quantile(df[gmv_total_col], 0.99))
        whale_mask = df[gmv_total_col] >= whale_threshold
        if whale_mask.any():
            rows.append(
                {
                    "slice": "whales (top 1%)",
                    "n_users": int(whale_mask.sum()),
                    "rmsle": _rmsle(
                        np.asarray(y_true)[whale_mask.values], np.asarray(y_pred)[whale_mask.values]
                    ),
                }
            )

    # Mainstream: complement of all above
    all_special = np.zeros(n, dtype=bool)
    if n_transactions_col in df.columns:
        median_n = float(np.median(df[n_transactions_col]))
        all_special |= (df[n_transactions_col] <= median_n / 2).values
    if last_active_col in df.columns:
        max_active = float(df[last_active_col].max())
        all_special |= (df[last_active_col] <= max_active - 270).values
    if gmv_total_col in df.columns:
        whale_threshold = float(np.quantile(df[gmv_total_col], 0.99))
        all_special |= (df[gmv_total_col] >= whale_threshold).values

    mainstream_mask = ~all_special
    if mainstream_mask.any():
        rows.append(
            {
                "slice": "mainstream",
                "n_users": int(mainstream_mask.sum()),
                "rmsle": _rmsle(
                    np.asarray(y_true)[mainstream_mask], np.asarray(y_pred)[mainstream_mask]
                ),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI on RMSLE.

    Resamples (y_true, y_pred) pairs with replacement n_boot times,
    computes RMSLE each time, returns (alpha/2, 1-alpha/2) percentiles.

    Args:
        y_true: ground truth.
        y_pred: predictions.
        n_boot: number of bootstrap samples.
        alpha: significance level (0.05 = 95% CI).
        seed: random seed for reproducibility (R6).

    Returns:
        (low, high) tuple of CI bounds.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    rmsle_samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmsle_samples[i] = _rmsle(y_true[idx], y_pred[idx])

    low = float(np.quantile(rmsle_samples, alpha / 2))
    high = float(np.quantile(rmsle_samples, 1 - alpha / 2))
    return (low, high)


def compute_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_boot: int = 1000,
    n_pit_bins: int = 10,
) -> dict[str, Any]:
    """Run all Tier 2/3 diagnostics in one call.

    Returns a flat dict suitable for direct DataFrame construction.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    pit_text, pit_counts = pit_histogram(y_true_arr, y_pred_arr, n_bins=n_pit_bins)
    ci_low, ci_high = bootstrap_ci(y_true_arr, y_pred_arr, n_boot=n_boot)
    return {
        "rmse": _rmse(y_true_arr, y_pred_arr),
        "spearman_global": None,  # filled by metrics.compute_all_metrics
        "percentiles_q10_pred": float(np.quantile(y_pred_arr, 0.1)),
        "percentiles_q50_pred": float(np.quantile(y_pred_arr, 0.5)),
        "percentiles_q90_pred": float(np.quantile(y_pred_arr, 0.9)),
        "percentiles_q10_true": float(np.quantile(y_true_arr, 0.1)),
        "percentiles_q50_true": float(np.quantile(y_true_arr, 0.5)),
        "percentiles_q90_true": float(np.quantile(y_true_arr, 0.9)),
        "coverage_q10_q90": coverage_at_quantile(y_true_arr, y_pred_arr, (0.1, 0.9)),
        "pit_total_observations": int(pit_counts.sum()),
        "pit_max_bin_count": int(pit_counts.max()),
        "pit_min_bin_count": int(pit_counts.min()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }
