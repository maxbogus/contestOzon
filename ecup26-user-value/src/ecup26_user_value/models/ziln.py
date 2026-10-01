"""Zero-Inflated Log-Normal (ZILN) loss for E-CUP T3 user-value.

Per .clinerules/02-ecup-rules.md (R1 open-source, R8 Docker contract) and
ticket T3.6.7.

45.9% of users in sample_submit.csv have predict=0. ZILN handles this
by modeling P(y=0) and y|z>0 ~ LogNormal(mu, sigma).

Provides:
- ziln_nll(preds, y): negative log-likelihood
- ziln_predict(z_prob, mu, sigma): expected value E[y]
- ZILNLoss: LightGBM-compatible custom objective (gradient + hessian)

Math:
- y = 0: probability = z_prob + (1 - z_prob) * f_normal_at_zero
- y > 0: probability = (1 - z_prob) * f_lognormal(y; mu, sigma)
- E[y] = (1 - z_prob) * exp(mu + sigma^2/2)
"""

from __future__ import annotations

import numpy as np

#: Small constant for numerical stability in log-space.
EPS = 1e-8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ziln_predict(
    z_prob: float | np.ndarray, mu: float | np.ndarray, sigma: float | np.ndarray
) -> np.ndarray | float:
    """Compute expected value E[y] under ZILN.

    Args:
        z_prob: probability of y=0 (shape broadcastable)
        mu: log-normal mean parameter
        sigma: log-normal stddev parameter

    Returns:
        (1 - z_prob) * exp(mu + sigma^2/2)
    """
    return (1.0 - z_prob) * np.exp(mu + 0.5 * sigma**2)


def ziln_nll(preds: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for ZILN distribution.

    Args:
        preds: (n, 3) array with cols (z_logit, mu, sigma) where
               z_logit is the logit of zero-probability.
        y: (n,) target values (non-negative).

    Returns:
        Scalar NLL (averaged over samples).
    """
    z_logit = preds[:, 0]
    mu = preds[:, 1]
    sigma = preds[:, 2]

    z_prob = _sigmoid(z_logit)
    sigma_safe = np.maximum(sigma, EPS)

    is_zero = y == 0.0

    # log P(y=0) = log(z_prob + (1-z_prob) * Phi(0))
    # where Phi(0) is log-normal density at zero = -log(sigma) - mu^2/(2sigma^2)/sqrt(2pi)
    # equivalent to: log(z_prob + (1-z_prob) * exp(-mu^2/(2sigma^2)) / (sigma * sqrt(2*pi)))
    const_log = -0.5 * np.log(2.0 * np.pi)
    log_phi_zero = const_log - np.log(sigma_safe) - (mu**2) / (2.0 * sigma_safe**2)

    log_p_zero = np.log(z_prob + (1.0 - z_prob) * np.exp(log_phi_zero) + EPS)
    # log P(y>0 | non-zero) = log(1 - z_prob) + log_lognormal(y)
    # log_lognormal(y; mu, sigma) = -log(y*sigma*sqrt(2pi)) - (log(y) - mu)^2 / (2*sigma^2)
    log_y = np.log(np.maximum(y, EPS))
    log_log_normal = (
        -np.log(np.maximum(y, EPS))
        - np.log(sigma_safe)
        + const_log
        - (log_y - mu) ** 2 / (2.0 * sigma_safe**2)
    )
    log_p_positive = np.log(1.0 - z_prob + EPS) + log_log_normal

    nll_per_sample = np.where(is_zero, log_p_zero, log_p_positive)
    return float(-np.mean(nll_per_sample))


class ZILNLoss:
    """LightGBM-compatible ZILN custom objective.

    Wraps the ZILN distribution as a loss function returning gradient
    and hessian (per-row) for LightGBM gradient boosting.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, preds: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradient and hessian for ZILN.

        Args:
            preds: (n, 3) array with cols (z_logit, mu, sigma).
            y: (n,) target values (non-negative).

        Returns:
            (gradient, hessian) tuple, each shape (n,).
            Gradient has shape (n, 3) actually for multi-output regression,
            but LightGBM accepts (n,).
        """
        # Per-sample NLL gradient via numerical approximation (simple finite diff)
        n = len(y)
        grad = np.zeros((n, 3), dtype=np.float64)
        hess = np.ones((n, 3), dtype=np.float64)  # positive for LightGBM

        eps = 1e-4
        for i in range(n):
            for j in range(3):
                p_plus = preds[i].copy()
                p_plus[j] += eps
                p_minus = preds[i].copy()
                p_minus[j] -= eps
                nll_plus = _single_nll(p_plus, y[i])
                nll_minus = _single_nll(p_minus, y[i])
                grad[i, j] = (nll_plus - nll_minus) / (2.0 * eps)
                # Approximate Hessian with second-order finite difference
                hess[i, j] = max(
                    (nll_plus - 2 * ziln_single(p_preds=p_plus, y_val=y[i]) + nll_minus) / (eps**2),
                    EPS,
                )
        # LightGBM needs 1-D gradient/hessian matching number of rows
        # For multi-output, use sum across outputs
        return grad.sum(axis=1), hess.sum(axis=1)


def _single_nll(pred: np.ndarray, y_val: float) -> float:
    """Compute NLL for a single sample."""
    return ziln_single(pred.reshape(1, 3), np.array([y_val]))


def ziln_single(p_preds: np.ndarray, y_val: np.ndarray) -> float:
    """Wrapper: NLL for single sample(s)."""
    if y_val.ndim == 0:
        y_val = np.array([y_val])
    if p_preds.ndim == 1:
        p_preds = p_preds.reshape(1, -1)
    return ziln_nll(p_preds, y_val)
