"""Tier 1 validation metrics for E-CUP T3 user-value predictions.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (numpy, pandas, scipy)
- R6: reproducible (deterministic math, no random seeds)
- R8: NOT imported by run.py (dev-only evaluation tool)

Provides:
- naive_baseline_rmsle(y_true, strategy="mean"|"median"|"zero") -> float
- lift_vs_naive(y_true, y_pred, strategy="mean") -> float
- mae(y_true, y_pred) -> float
- rmse(y_true, y_pred) -> float
- bias_mean(y_true, y_pred) -> float
- spearman_global(y_true, y_pred) -> float  (NaN-safe)
- quantile_calibration_error(y_true, y_pred, quantiles=(0.1, 0.5, 0.9)) -> float
- public_private_split(df, ref_date, private_days=30) -> (train_df, val_df)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _rmsle_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE with the T3 convention: log1p, max(0, pred)."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def naive_baseline_rmsle(
    y_true: np.ndarray | pd.Series,
    strategy: Literal["mean", "median", "zero"] = "mean",
) -> float:
    """RMSLE achieved by a constant predictor.

    Useful for "are we better than nothing?" sanity check.

    Args:
        y_true: ground truth GMV per user (non-negative).
        strategy: which constant to predict.
            - "mean": predict mean(y_true)
            - "median": predict median(y_true)
            - "zero": predict 0 for everyone

    Returns:
        Scalar RMSLE of the constant predictor against y_true.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    if strategy == "mean":
        const = float(np.mean(y_true))
    elif strategy == "median":
        const = float(np.median(y_true))
    elif strategy == "zero":
        const = 0.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return _rmsle_from_arrays(y_true, np.full_like(y_true, const))


def lift_vs_naive(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    strategy: Literal["mean", "median", "zero"] = "mean",
) -> float:
    """Fractional improvement of model RMSLE over naive baseline.

    Returns a value in (-inf, 1]:
    - 1.0 = perfect (model RMSLE = 0)
    - 0.0 = same as naive baseline
    - negative = worse than naive baseline

    If naive_rmsle is 0 (e.g. all targets are zero), returns 0.0
    (degenerate case; the model cannot meaningfully improve over 0).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    model_rmsle = _rmsle_from_arrays(y_true, y_pred)
    naive_rmsle = naive_baseline_rmsle(y_true, strategy=strategy)
    if naive_rmsle == 0.0:
        return 0.0
    return (naive_rmsle - model_rmsle) / naive_rmsle


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias_mean(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean signed error: positive = overprediction, negative = underprediction.

    Defined as mean(pred - true). 0 means the model is unbiased on average.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(y_pred - y_true))


def spearman_global(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Spearman correlation between predictions and ground truth.

    NaN-safe: returns 0.0 for degenerate inputs (constant pred, length<2)
    instead of raising.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return 0.0
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else 0.0


def quantile_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> float:
    """RMS difference between predicted and true quantile values.

    Captures calibration drift that mean-based metrics (MAE/RMSE) hide.
    A well-calibrated model has this near 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    diffs = []
    for q in quantiles:
        q_true = float(np.quantile(y_true, q))
        q_pred = float(np.quantile(y_pred, q))
        diffs.append((q_true - q_pred) ** 2)
    return float(np.sqrt(np.mean(diffs)))


def public_private_split(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | str | None = None,
    private_days: int = 30,
    date_col: str = "event_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into train (everything before private window)
    and val (the private window).

    Mimics the E-CUP public/private 20/80 split (per task3_user_value.md).
    Time-ordered; no shuffling.

    Args:
        df: transaction DataFrame with a date column.
        ref_date: end of train / start of val. Default: max date in df.
        private_days: length of the validation window in days.
        date_col: name of the date column.

    Returns:
        (train_df, val_df) tuple.
    """
    if ref_date is None:
        ref_date = df[date_col].max()
    if isinstance(ref_date, str):
        ref_date = pd.Timestamp(ref_date)

    train = df[df[date_col] <= ref_date].copy()
    val_end = ref_date + pd.Timedelta(days=private_days)
    val = df[(df[date_col] > ref_date) & (df[date_col] <= val_end)].copy()
    return train, val


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Run all Tier 1 metrics in one call.

    Returns a flat dict suitable for direct DataFrame construction.
    """
    return {
        "model_rmsle": _rmsle_from_arrays(y_true, y_pred),
        "naive_rmsle_mean": naive_baseline_rmsle(y_true, "mean"),
        "naive_rmsle_median": naive_baseline_rmsle(y_true, "median"),
        "lift_vs_mean": lift_vs_naive(y_true, y_pred, "mean"),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias_mean": bias_mean(y_true, y_pred),
        "spearman": spearman_global(y_true, y_pred),
        "quantile_calibration": quantile_calibration_error(y_true, y_pred),
    }
