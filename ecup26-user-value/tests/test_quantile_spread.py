"""Unit tests for T3.10.22 quantile-spread feature OOF helper.

Per .clinerules: R1 (numpy/pandas), R6 (deterministic). Tests only the pure
`_user_folds` split helper - no model fit (heavy quantile training is the
script's job).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import ab_validate as av  # noqa: E402


def test_same_user_same_fold() -> None:
    # user 1 appears 3x (3 origins), user 2 once, etc.
    u = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5])
    folds = av._user_folds(u, 3)
    assert folds.shape == u.shape
    for uid in np.unique(u):
        vals = set(folds[u == uid].tolist())
        assert len(vals) == 1, f"user {uid} split across folds: {vals}"


def test_deterministic() -> None:
    u = np.array([7, 8, 9, 10, 11, 12, 13])
    a = av._user_folds(u, 3)
    b = av._user_folds(u, 3)
    assert (a == b).all()


def test_all_folds_used_with_enough_users() -> None:
    rng = np.random.default_rng(0)
    u = rng.integers(0, 200, size=2000)
    folds = av._user_folds(u, 3)
    assert set(folds.tolist()) == {0, 1, 2}
