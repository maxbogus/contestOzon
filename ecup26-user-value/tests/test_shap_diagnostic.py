"""Unit tests for T3.10.16 SHAP aggregation helpers (scripts/shap_diagnostic.py).

Per .clinerules: R1 (numpy/pandas), R6 (deterministic pure functions). Tests only
the pure helpers - no model fit, no shap import (heavy numba startup avoided).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import shap_diagnostic as sd  # noqa: E402


def _make_shap() -> np.ndarray:
    """5 features x 4 users; f0 dominates, f4 weakly opposes f0 for user 0."""
    return np.array(
        [
            [2.0, 0.1, 0.0, 0.0, -0.2],
            [2.5, 0.2, 0.0, 0.0, -0.1],
            [-1.8, 0.3, 0.0, 0.0, 0.2],
            [1.9, -0.1, 0.0, 0.0, 0.1],
        ]
    )


FEATURES = ["gmv_90d", "searches_30d", "noise_a", "noise_b", "recency"]


def test_rank_features_top() -> None:
    top = sd.rank_features(_make_shap(), FEATURES, top_k=3)
    assert list(top["feature"]) == ["gmv_90d", "searches_30d", "recency"]
    assert top.iloc[0]["mean_abs_shap"] == pytest.approx(2.05)


def test_group_profile_sign() -> None:
    sv = _make_shap()
    mask = np.array([True, True, False, True])
    prof = sd.group_profile(sv, FEATURES, mask)
    # gmv_90d mean over users 0,1,3 = (2.0+2.5+1.9)/3 = 2.1333
    assert prof.loc[0, "mean_shap"] == pytest.approx(2.1333333333)


def test_divergence_ordering() -> None:
    sv = _make_shap()
    zeros = np.array([True, True, False, False])
    whales = np.array([False, False, True, True])
    div = sd.divergence(
        sd.group_profile(sv, FEATURES, zeros),
        sd.group_profile(sv, FEATURES, whales),
        top_k=3,
    )
    assert div.iloc[0]["feature"] == "gmv_90d"  # largest |whale - zero| diff
    assert len(div) == 3
