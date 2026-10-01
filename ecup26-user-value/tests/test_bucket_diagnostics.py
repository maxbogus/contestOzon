"""RED tests for T3.6.17 - bucket-level diagnostics.

Per docs/research/agent-ltv.md §4.1 (T3.6.3 note):
- Split predictions into zero / quintiles / top 1%
- Report per-bucket RMSLE + Spearman
- Catches calibration drift hidden by global metric
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestBucketDiagnostics:
    """bucket_diagnostics() returns a per-bucket DataFrame."""

    def test_basic_quintiles(self) -> None:
        from ecup26_user_value.eval.bucket_diagnostics import bucket_diagnostics

        rng = np.random.default_rng(0)
        y_true = np.concatenate(
            [
                np.zeros(50),  # zero bucket
                rng.uniform(10, 100, 100),  # low
                rng.uniform(100, 500, 100),  # mid
                rng.uniform(500, 2000, 100),  # high
                rng.uniform(2000, 10000, 50),  # top
            ]
        )
        y_pred = y_true * rng.uniform(0.5, 1.5, size=len(y_true))

        result = bucket_diagnostics(y_true, y_pred, n_buckets=4)
        assert isinstance(result, pd.DataFrame)
        assert "bucket" in result.columns
        assert "n_users" in result.columns
        assert "rmsle_bucket" in result.columns
        # Should have at least 1 (zero) + 4 quintile buckets = 5 rows
        assert len(result) >= 5

    def test_zero_bucket_separated(self) -> None:
        """Users with y_true == 0 must be in a dedicated 'zero' bucket."""
        from ecup26_user_value.eval.bucket_diagnostics import bucket_diagnostics

        y_true = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 200.0, 300.0])
        y_pred = np.array([5.0, 10.0, 15.0, 20.0, 90.0, 210.0, 280.0])

        result = bucket_diagnostics(y_true, y_pred, n_buckets=3)
        zero_row = result[result["bucket"] == "zero"]
        assert len(zero_row) == 1
        assert zero_row.iloc[0]["n_users"] == 4

    def test_nan_spearman_for_degenerate_buckets(self) -> None:
        """If bucket has <2 unique predictions, spearman must be NaN (not crash)."""
        from ecup26_user_value.eval.bucket_diagnostics import bucket_diagnostics

        y_true = np.array([0.0, 0.0, 0.0, 100.0, 200.0, 300.0])
        y_pred = np.array([5.0, 5.0, 5.0, 90.0, 210.0, 280.0])

        result = bucket_diagnostics(y_true, y_pred, n_buckets=3)
        # Zero bucket has all y_pred identical -> spearman_bucket is NaN
        zero_row = result[result["bucket"] == "zero"]
        assert np.isnan(zero_row.iloc[0]["spearman_bucket"])

    def test_predictions_nonnegative_required(self) -> None:
        """RMSLE convention: negative predictions must be clipped at 0."""
        from ecup26_user_value.eval.bucket_diagnostics import bucket_diagnostics

        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([-50.0, 150.0, 350.0])  # one negative
        # Should not crash; should clip at 0
        result = bucket_diagnostics(y_true, y_pred, n_buckets=2)
        assert not np.isnan(result["rmsle_bucket"]).any()

    def test_does_not_mutate_inputs(self) -> None:
        """Defensive: input arrays must not be modified."""
        from ecup26_user_value.eval.bucket_diagnostics import bucket_diagnostics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        y_true_copy = y_true.copy()
        y_pred_copy = y_pred.copy()
        bucket_diagnostics(y_true, y_pred, n_buckets=2)
        np.testing.assert_array_equal(y_true, y_true_copy)
        np.testing.assert_array_equal(y_pred, y_pred_copy)
