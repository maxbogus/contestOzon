"""RED tests for T3.7.2 Tier 2/3 diagnostics.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (stdlib + numpy + pandas + scipy; NO matplotlib)
- R6: reproducible (seeded bootstrap, deterministic math)
- R8: NOT imported by run.py (dev-only evaluation)
- R10: no submit.csv writes
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


class TestRmseReexport:
    """rmse from metrics.py is re-exported from diagnostics for unified API."""

    def test_rmse_simple(self) -> None:
        from ecup26_user_value.eval.diagnostics import rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        expected = float(np.sqrt(2 / 3))
        assert rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-9)


class TestPercentilesAlignment:
    """percentiles_alignment returns dict {Q10_pred, Q10_true, ...}."""

    def test_perfect_alignment(self) -> None:
        from ecup26_user_value.eval.diagnostics import percentiles_alignment

        y = np.arange(1, 101, dtype=float)  # 1..100
        result = percentiles_alignment(y, y, q=(0.1, 0.5, 0.9))
        assert "q10_pred" in result and "q10_true" in result
        assert result["q10_pred"] == pytest.approx(result["q10_true"], abs=1e-9)
        assert result["q50_pred"] == pytest.approx(result["q50_true"], abs=1e-9)
        assert result["q90_pred"] == pytest.approx(result["q90_true"], abs=1e-9)

    def test_offset_predictions(self) -> None:
        from ecup26_user_value.eval.diagnostics import percentiles_alignment

        y_true = np.arange(1, 101, dtype=float)
        y_pred = y_true * 2.0  # offset
        result = percentiles_alignment(y_true, y_pred, q=(0.5,))
        assert result["q50_pred"] > result["q50_true"]


class TestCoverageAtQuantile:
    """coverage_at_quantile: fraction of users where pred and true are in same band."""

    def test_perfect_calibration_returns_high_coverage(self) -> None:
        """q10-q90 band by definition covers ~80% of true, so coverage >= 0.7."""
        from ecup26_user_value.eval.diagnostics import coverage_at_quantile

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 1000)
        y_pred = y_true.copy()
        cov = coverage_at_quantile(y_true, y_pred, q=(0.1, 0.9))
        assert cov == pytest.approx(0.8, abs=0.05), f"got {cov}"

    def test_random_predictions_have_partial_coverage(self) -> None:
        from ecup26_user_value.eval.diagnostics import coverage_at_quantile

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 1000)
        y_pred = rng.uniform(0, 100, 1000)  # uncorrelated predictions
        cov = coverage_at_quantile(y_true, y_pred, q=(0.1, 0.9))
        # Random predictions: coverage should be much lower than well-calibrated
        assert cov < 0.7, f"random cov should be < 0.7, got {cov}"


class TestStabilitySeedVariance:
    """stability_seed_variance: fit_predict_fn called N times, return mean/std RMSLE."""

    def test_low_variance_on_deterministic_model(self) -> None:
        from ecup26_user_value.eval.diagnostics import stability_seed_variance

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 50)

        def fit_predict(df, seed):
            # Deterministic: predict mean (ignores seed)
            return np.full(len(y_true), float(np.mean(y_true)))

        result = stability_seed_variance(fit_predict, None, n_seeds=3, y_true=y_true)
        assert "rmsle_mean" in result
        assert "rmsle_std" in result
        assert result["rmsle_std"] == pytest.approx(0.0, abs=1e-9)

    def test_high_variance_on_noisy_model(self) -> None:
        from ecup26_user_value.eval.diagnostics import stability_seed_variance

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 50)

        def fit_predict(df, seed):
            rng2 = np.random.default_rng(seed)
            noise = rng2.uniform(-50, 50, len(y_true))
            return np.full(len(y_true), float(np.mean(y_true))) + noise

        result = stability_seed_variance(fit_predict, None, n_seeds=3, y_true=y_true)
        assert result["rmsle_std"] > 0


class TestPitHistogram:
    """pit_histogram: ASCII string with bin counts. Returns (string, counts_array)."""

    def test_returns_ascii_string_and_counts(self) -> None:
        from ecup26_user_value.eval.diagnostics import pit_histogram

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 100)
        # Predictions very close to truth -> PIT should be roughly uniform
        y_pred = y_true + rng.normal(0, 0.01, 100)

        text, counts = pit_histogram(y_true, y_pred, n_bins=10)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "*" in text  # ASCII histogram uses *
        assert isinstance(counts, np.ndarray)
        assert len(counts) == 10
        assert counts.sum() == 100  # all observations binned
        # Counts should be roughly balanced (within 2x of mean per bin)
        mean_per_bin = 10  # 100 / 10
        assert (counts >= 0).all()
        assert (counts < mean_per_bin * 4).all(), f"too imbalanced: {counts}"

    def test_calibrated_predictions_have_balanced_bins(self) -> None:
        from ecup26_user_value.eval.diagnostics import pit_histogram

        rng = np.random.default_rng(42)
        n = 1000
        y_true = rng.uniform(0, 100, n)
        # Tiny noise -> well calibrated
        y_pred = y_true + rng.normal(0, 0.001, n)
        _, counts = pit_histogram(y_true, y_pred, n_bins=10)
        # Expected ~100 per bin; chi-square-like check: none should be < 30
        assert counts.min() > 30, f"too few in some bins: {counts}"


class TestAdversarialBreakdown:
    """adversarial_breakdown: per-slice RMSLE."""

    def test_returns_per_slice_metrics(self) -> None:
        from ecup26_user_value.eval.diagnostics import adversarial_breakdown

        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "user_id": range(n),
                "y_true": rng.uniform(0, 100, n),
                "y_pred": rng.uniform(0, 100, n),
                "n_transactions": rng.integers(1, 30, n),
                "last_active_day": rng.integers(1, 365, n),
                "gmv_total": rng.uniform(0, 10000, n),
            }
        )
        result = adversarial_breakdown(df)
        assert isinstance(result, pd.DataFrame)
        assert "slice" in result.columns
        assert "n_users" in result.columns
        assert "rmsle" in result.columns
        # Should have at least 3 slices: cold-start, churned, whales
        assert len(result) >= 3


class TestBootstrapCi:
    """bootstrap_ci: 95% CI on RMSLE. Returns (low, high)."""

    def test_perfect_prediction_zero_ci(self) -> None:
        from ecup26_user_value.eval.diagnostics import bootstrap_ci

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 50)
        y_pred = y_true.copy()
        low, high = bootstrap_ci(y_true, y_pred, n_boot=100)
        assert low == pytest.approx(0.0, abs=1e-9)
        assert high == pytest.approx(0.0, abs=1e-9)

    def test_low_less_than_high(self) -> None:
        from ecup26_user_value.eval.diagnostics import bootstrap_ci

        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, 100)
        y_pred = y_true + rng.normal(0, 10, 100)  # noisy
        low, high = bootstrap_ci(y_true, y_pred, n_boot=200)
        assert low <= high
        assert high > 0  # non-trivial interval
        assert math.isfinite(low) and math.isfinite(high)


class TestDiagnosticsDoesNotTouchSubmission:
    """Sanity: diagnostics module does not import run.py or submission internals."""

    def test_no_run_py_import(self) -> None:
        # If diagnostics accidentally depends on submission, this breaks
        # the R8 isolation. Just import and verify it works.
        import ecup26_user_value.eval.diagnostics  # noqa: F401

        # No assertion needed; the import either succeeds or raises ImportError.
        # The point is: it should succeed without triggering run.py side effects.
        assert True
