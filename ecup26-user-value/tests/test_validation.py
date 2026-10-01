"""RED tests for T3.7.1 Tier 1 validation metrics.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (numpy, pandas, scipy)
- R6: reproducible (deterministic math, no random)
- R10: no submit.csv writes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_data(n_users: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 1000, n_users)
    y_pred = y_true * rng.uniform(0.5, 1.5, n_users)
    return y_true, y_pred


class TestNaiveBaselineRmsle:
    """naive_baseline_rmsle returns RMSLE of constant predictor (mean/median/zero)."""

    def test_mean_strategy(self) -> None:
        from ecup26_user_value.eval.metrics import naive_baseline_rmsle

        y_true = np.array([100.0, 200.0, 300.0])
        # mean = 200, log1p(100)=4.615, log1p(200)=5.298, log1p(300)=5.704
        # RMSLE = sqrt(mean((log1p(y) - log1p(200))^2))
        result = naive_baseline_rmsle(y_true, strategy="mean")
        expected = float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(200.0)) ** 2)))
        assert abs(result - expected) < 1e-9, f"got {result}, expected {expected}"

    def test_median_strategy(self) -> None:
        from ecup26_user_value.eval.metrics import naive_baseline_rmsle

        y_true = np.array([100.0, 200.0, 300.0])
        result = naive_baseline_rmsle(y_true, strategy="median")
        expected = float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(200.0)) ** 2)))
        assert abs(result - expected) < 1e-9

    def test_zero_strategy(self) -> None:
        from ecup26_user_value.eval.metrics import naive_baseline_rmsle

        y_true = np.array([0.0, 100.0, 200.0])
        result = naive_baseline_rmsle(y_true, strategy="zero")
        # log1p of zero = 0 for all rows; mean of (log1p(y) - 0)^2
        expected = float(np.sqrt(np.mean(np.log1p(y_true) ** 2)))
        assert abs(result - expected) < 1e-9


class TestLiftVsNaive:
    """lift_vs_naive reports % improvement over naive baseline."""

    def test_perfect_prediction_is_huge_lift(self) -> None:
        from ecup26_user_value.eval.metrics import lift_vs_naive

        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = y_true.copy()
        # Perfect prediction: RMSLE=0, naive>0 -> lift = (naive - 0)/naive = 1.0
        lift = lift_vs_naive(y_true, y_pred, strategy="mean")
        assert lift == pytest.approx(1.0, abs=1e-9), f"got {lift}"

    def test_worse_than_naive_is_negative(self) -> None:
        from ecup26_user_value.eval.metrics import lift_vs_naive

        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([10000.0, 10000.0, 10000.0])  # terrible
        # Should be negative (worse than constant predictor)
        lift = lift_vs_naive(y_true, y_pred, strategy="mean")
        assert lift < 0, f"expected negative lift, got {lift}"


class TestMae:
    """mae = mean(|true - pred|)."""

    def test_simple(self) -> None:
        from ecup26_user_value.eval.metrics import mae

        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 35.0])
        # |10-12| + |20-18| + |30-35| = 2 + 2 + 5 = 9, /3 = 3.0
        assert mae(y_true, y_pred) == pytest.approx(3.0, abs=1e-9)


class TestRmse:
    """rmse = sqrt(mean((true - pred)^2))."""

    def test_simple(self) -> None:
        from ecup26_user_value.eval.metrics import rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # (1+0+1)/3 = 2/3, sqrt = 0.8165
        expected = float(np.sqrt(2 / 3))
        assert rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-9)


class TestBiasMean:
    """bias_mean = mean(pred - true). 0 = unbiased, >0 = overprediction."""

    def test_perfect_prediction_is_zero(self) -> None:
        from ecup26_user_value.eval.metrics import bias_mean

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        assert bias_mean(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_constant_overprediction(self) -> None:
        from ecup26_user_value.eval.metrics import bias_mean

        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = y_true + 5.0  # always +5
        assert bias_mean(y_true, y_pred) == pytest.approx(5.0, abs=1e-9)


class TestSpearmanGlobal:
    """spearman_global = Spearman correlation (no NaN on degenerate inputs)."""

    def test_perfect_correlation(self) -> None:
        from ecup26_user_value.eval.metrics import spearman_global

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true * 2.0 + 1.0
        rho = spearman_global(y_true, y_pred)
        assert rho == pytest.approx(1.0, abs=1e-9)

    def test_anti_perfect(self) -> None:
        from ecup26_user_value.eval.metrics import spearman_global

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        rho = spearman_global(y_true, y_pred)
        assert rho == pytest.approx(-1.0, abs=1e-9)

    def test_constant_prediction_returns_zero(self) -> None:
        """No variance in pred -> no correlation possible; return 0 (not NaN)."""
        from ecup26_user_value.eval.metrics import spearman_global

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])
        rho = spearman_global(y_true, y_pred)
        assert rho == 0.0


class TestQuantileCalibrationError:
    """quantile_calibration_error: Q10/Q50/Q90 of pred vs true."""

    def test_perfect_calibration_is_zero(self) -> None:
        from ecup26_user_value.eval.metrics import quantile_calibration_error

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = y_true.copy()
        err = quantile_calibration_error(y_true, y_pred)
        assert err == pytest.approx(0.0, abs=1e-9)

    def test_offset_predictions_have_error(self) -> None:
        from ecup26_user_value.eval.metrics import quantile_calibration_error

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = y_true * 2.0  # predictions are double
        err = quantile_calibration_error(y_true, y_pred)
        assert err > 0


class TestPublicPrivateSplit:
    """public_private_split: 80/20 time-ordered split."""

    def test_split_is_time_ordered(self) -> None:
        from ecup26_user_value.eval.metrics import public_private_split

        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        df = pd.DataFrame({"user_id": range(100), "event_date": dates, "gmv": 1.0})
        train, val = public_private_split(df, ref_date=dates[79], private_days=20)

        assert len(train) == 80, f"expected 80 train rows, got {len(train)}"
        assert len(val) == 20, f"expected 20 val rows, got {len(val)}"
        # Time-ordered: all train dates <= all val dates
        assert train["event_date"].max() <= val["event_date"].min(), "train and val overlap in time"


class TestAssertions:
    """Validations must handle edge cases without crashing."""

    def test_metrics_finite_on_synthetic(self) -> None:
        from ecup26_user_value.eval.metrics import (
            bias_mean,
            lift_vs_naive,
            mae,
            rmse,
            spearman_global,
        )

        y_true, y_pred = _make_data(n_users=50)
        for fn in (mae, rmse, bias_mean, spearman_global, lift_vs_naive):
            result = fn(y_true, y_pred)
            assert np.isfinite(result), f"{fn.__name__} returned {result}"

    def test_metrics_handle_zeros(self) -> None:
        from ecup26_user_value.eval.metrics import bias_mean, mae

        y_true = np.zeros(10)
        y_pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        # Zero target with zero prediction -> MAE=mean
        assert np.isfinite(mae(y_true, y_pred))
        # Bias = mean of positive preds when true=0 -> positive bias
        assert bias_mean(y_true, y_pred) > 0
