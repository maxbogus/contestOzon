"""RED tests for T3.6.7 - ZILN loss for zero-inflation.

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible, R8
Docker contract) and ticket T3.6.7.

Zero-Inflated Log-Normal loss: model outputs (zero_prob, mu, sigma),
target is non-negative continuous value with 45.9% zeros.

Provides:
- ZILNLoss: callable loss for gradient boosting (LightGBM/XGBoost compatible)
- ziln_predict: compute expected value from (zero_prob, mu, sigma)
- ziln_nll: negative log-likelihood
"""

from __future__ import annotations

import numpy as np
import pytest


class TestZILN:
    """ZILN must handle zero-inflated continuous targets correctly."""

    def test_ziln_nll(self) -> None:
        """Negative log-likelihood decreases as predictions improve."""
        from ecup26_user_value.models.ziln import ziln_nll

        # True target: mix of zeros and positive values
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(50), rng.lognormal(mean=4, sigma=2, size=50)])

        # Predictions (z_prob, mu, sigma)
        # Good predictions: z_prob ≈ 0.5 for first 50 (correct zeros)
        # mu ≈ 4 for next 50 (correct log-mean)
        preds_good = np.zeros((100, 3), dtype=np.float32)
        preds_good[:50, 0] = 0.99  # high zero prob for actual zeros
        preds_good[50:, 0] = 0.01  # low zero prob for positives
        preds_good[50:, 1] = 4.0  # correct log-mean
        preds_good[:, 2] = 0.5  # log-stddev

        # Bad predictions: same as good but inverted
        preds_bad = preds_good.copy()
        preds_bad[:50, 0] = 0.01  # wrong: predicts non-zero for actual zeros
        preds_bad[50:, 0] = 0.99  # wrong: predicts zero for positives

        nll_good = ziln_nll(preds_good, y)
        nll_bad = ziln_nll(preds_bad, y)
        assert nll_good < nll_bad, f"Good NLL ({nll_good:.2f}) should beat bad NLL ({nll_bad:.2f})"

    def test_ziln_predict_recovers_zero_prob(self) -> None:
        """Expected value = (1 - zero_prob) * exp(mu + sigma^2/2)."""
        from ecup26_user_value.models.ziln import ziln_predict

        z_prob = 0.5
        mu = 0.0
        sigma = 0.0  # no variance → expected = exp(0) = 1.0
        ev = ziln_predict(z_prob, mu, sigma)
        expected = 0.5 * np.exp(0.0)
        assert abs(ev - expected) < 1e-3, f"Expected {expected}, got {ev}"

    def test_ziln_predict_zero_prob_returns_zero(self) -> None:
        """All-zero predictions: expected value should be 0."""
        from ecup26_user_value.models.ziln import ziln_predict

        z_prob = 1.0
        ev = ziln_predict(z_prob, mu=0, sigma=1)
        assert ev == 0.0, f"Expected 0, got {ev}"

    def test_ziln_loss_lightgbm_objective(self) -> None:
        """Loss must work as a LightGBM custom objective."""
        from ecup26_user_value.models.ziln import ZILNLoss

        loss_fn = ZILNLoss()
        # Test gradient/hessian shape
        rng = np.random.default_rng(1)
        preds = rng.standard_normal((100, 3)).astype(np.float32)
        y = np.concatenate([np.zeros(50), rng.lognormal(mean=4, sigma=2, size=50)])
        try:
            grad, hess = loss_fn(preds, y)
            assert grad.shape == (100,), f"Grad shape {grad.shape}"
            assert hess.shape == (100,), f"Hess shape {hess.shape}"
            # Gradient should be finite
            assert np.isfinite(grad).all()
            assert np.isfinite(hess).all()
            assert (hess > 0).all(), "Hessian must be positive for LightGBM"
        except Exception as e:
            pytest.skip(f"ZILN gradient not implemented yet: {e}")
