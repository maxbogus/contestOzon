"""Per-bucket diagnostics for LTV predictions (T3.6.17).

Idea from docs/research/agent-ltv.md (T3.6.3 note, \u00a74.1):
split predictions into zero / quintiles / top 1%, report per-bucket RMSLE
and Spearman. Catches calibration drift that global RMSLE hides.

Provides:
- bucket_diagnostics(y_true, y_pred, n_buckets=5) -> pd.DataFrame

Buckets returned (in this order):
1. "zero"            - users with y_true == 0
2. "1-25%" ... "n_buckets-1 buckets"  - quintiles of y_true > 0
3. "top 1%"          - top percentile of y_true (separate, always present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE per task3_user_value.md: log1p of max(0, pred)."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation; returns NaN for degenerate buckets (<2 unique preds)."""
    if len(y_true) < 2:
        return float("nan")
    if len(np.unique(y_pred)) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else float("nan")


def bucket_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Compute per-bucket RMSLE + Spearman for LTV predictions.

    Args:
        y_true: ground truth GMV per user (non-negative).
        y_pred: predicted GMV per user (will be clipped to >=0).
        n_buckets: number of quintile buckets for y_true > 0 users.

    Returns:
        DataFrame with columns: bucket, n_users, mean_true, mean_pred,
        rmsle_bucket, spearman_bucket. One row per bucket.
    """
    y_true = np.asarray(y_true, dtype=np.float64).copy()
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0).copy()

    rows: list[dict[str, object]] = []

    # Bucket 1: zero users
    zero_mask = y_true == 0
    if zero_mask.any():
        rows.append(
            {
                "bucket": "zero",
                "n_users": int(zero_mask.sum()),
                "mean_true": 0.0,
                "mean_pred": float(np.mean(y_pred[zero_mask])),
                "rmsle_bucket": _rmsle(y_true[zero_mask], y_pred[zero_mask]),
                "spearman_bucket": _safe_spearman(y_true[zero_mask], y_pred[zero_mask]),
            }
        )

    # Buckets 2..n+1: quintiles of positive y_true
    pos_mask = y_true > 0
    if pos_mask.any():
        y_pos = y_true[pos_mask]
        yp_pos = y_pred[pos_mask]
        # Percentile-based bucket assignment
        try:
            bucket_ids = pd.qcut(y_pos, q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            # Too few unique positive values for n_buckets quintiles
            bucket_ids = np.zeros(len(y_pos), dtype=int)
        unique_ids = sorted(np.unique(bucket_ids).tolist())
        n_unique = len(unique_ids)
        for bid in unique_ids:
            mask = bucket_ids == bid
            pct_lo = int(round(100 * bid / n_unique))
            pct_hi = int(round(100 * (bid + 1) / n_unique))
            label = f"{pct_lo}-{pct_hi}%"
            rows.append(
                {
                    "bucket": label,
                    "n_users": int(mask.sum()),
                    "mean_true": float(np.mean(y_pos[mask])),
                    "mean_pred": float(np.mean(yp_pos[mask])),
                    "rmsle_bucket": _rmsle(y_pos[mask], yp_pos[mask]),
                    "spearman_bucket": _safe_spearman(y_pos[mask], yp_pos[mask]),
                }
            )

    # Final bucket: top 1% of y_true (separate from quintiles)
    n_total = len(y_true)
    n_top = max(1, n_total // 100)
    # Top-N by y_true (regardless of bucket above)
    top_idx = np.argsort(y_true)[-n_top:]
    rows.append(
        {
            "bucket": "top 1%",
            "n_users": int(n_top),
            "mean_true": float(np.mean(y_true[top_idx])),
            "mean_pred": float(np.mean(y_pred[top_idx])),
            "rmsle_bucket": _rmsle(y_true[top_idx], y_pred[top_idx]),
            "spearman_bucket": _safe_spearman(y_true[top_idx], y_pred[top_idx]),
        }
    )

    return pd.DataFrame(rows)
