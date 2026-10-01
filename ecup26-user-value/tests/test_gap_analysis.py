"""Unit tests for T3.10.27 gap-analysis decomposition (scripts/gap_analysis.py).

Per .clinerules/02-ecup-rules.md: R1 (numpy), R6 (deterministic, seeded-free pure
functions). These tests exercise only the pure decomposition helpers, no model fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import gap_analysis as ga  # noqa: E402


def _make_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8 users: 4 zeros, 4 positives; one clear whale among the positives."""
    y_raw = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 1000.0])
    yt_log = np.log1p(y_raw)
    # model over-predicts zeros (pred_log 1.0 vs truth 0), under-predicts the whale.
    pred_log = np.array([1.0, 1.0, 1.0, 1.0, 2.4, 3.0, 3.4, 6.0])
    return y_raw, yt_log, pred_log


def test_zero_frac_and_rmsle() -> None:
    y_raw, yt_log, pred_log = _make_preds()
    s = ga.build_segments(y_raw, yt_log, pred_log)
    assert s["zero_frac"] == 0.5
    assert s["n"] == 8
    expected = float(np.sqrt(np.mean((pred_log - yt_log) ** 2)))
    assert s["rmsle"] == pytest.approx(expected)


def test_err2_shares_partition() -> None:
    y_raw, yt_log, pred_log = _make_preds()
    s = ga.build_segments(y_raw, yt_log, pred_log, whale_fracs=(0.25,))
    z = s["segments"]["zeros"]
    p = s["segments"]["positives"]
    assert z["err2_share"] + p["err2_share"] == pytest.approx(1.0, abs=1e-9)
    assert z["n"] + p["n"] == 8


def test_whale_top25_is_largest_users() -> None:
    y_raw, yt_log, pred_log = _make_preds()
    s = ga.build_segments(y_raw, yt_log, pred_log, whale_fracs=(0.25,))
    w = s["segments"]["top25pct"]
    assert w["n"] == 2  # the 30 and 1000 spenders
    assert s["segments"]["top25pct_zeros"]["n"] == 0


def test_bias_sign_zeros_overpredicted() -> None:
    y_raw, yt_log, pred_log = _make_preds()
    s = ga.build_segments(y_raw, yt_log, pred_log)
    # zeros: pred_log 1.0 > truth 0 => bias = truth - pred = -1.0 (over-predicted)
    assert s["segments"]["zeros"]["bias_mean"] == pytest.approx(-1.0)
