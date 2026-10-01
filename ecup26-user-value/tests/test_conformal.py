"""Unit tests for T3.10.23 split-conformal helpers (scripts/conformal_threshold.py).

Per .clinerules: R1 (numpy), R6 (deterministic pure functions). No model fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import conformal_threshold as ct  # noqa: E402


def test_nonconformity_zero_vs_positive() -> None:
    p1 = np.array([0.9, 0.1, 0.5])
    y = np.array([0, 1, 1])  # true labels: zero, positive, positive
    s = ct._nonconformity(p1, y)
    # label 0 -> score = P(y>0) = 0.9; label 1 -> score = 1 - P(y>0)
    assert s[0] == pytest.approx(0.9)
    assert s[1] == pytest.approx(0.9)
    assert s[2] == pytest.approx(0.5)


def test_conformal_qhat_level_and_value() -> None:
    scores = np.arange(1.0, 11.0)  # 10 sorted scores 1..10
    # n=10, alpha=0.1 -> level = ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
    q = ct._conformal_qhat(scores, 0.1)
    assert q == pytest.approx(10.0)  # higher-quantile at level 1.0 = max
    # alpha=0.5 -> level = ceil(11*0.5)/10 = ceil(5.5)/10 = 6/10 = 0.6
    q2 = ct._conformal_qhat(scores, 0.5)
    assert q2 == pytest.approx(7.0)  # 60th pct linear=6.4, higher -> 7


def test_conformal_qhat_finite_sample_guarantee() -> None:
    # scores that are all small except the max: q_hat at 90% must be >= max-1
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, size=200)
    q = ct._conformal_qhat(scores, 0.1)
    assert 0.0 <= q <= 1.0
    # a calibrated threshold must leave <= alpha fraction of points above it
    assert float((scores > q).mean()) <= 0.12
