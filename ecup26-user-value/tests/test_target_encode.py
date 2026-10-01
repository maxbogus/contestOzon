"""Unit tests for T3.10.21 OOF target-encoding helper (_oof_group_mean).

Per .clinerules: R1 (numpy), R6 (deterministic pure function). No model fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import ab_validate as av  # noqa: E402


def test_oof_uses_other_folds_only() -> None:
    # 6 rows, 2 groups (0/1), 2 folds. Rows 0-2 in fold 0, rows 3-5 in fold 1.
    y = np.array([0.0, 10.0, 20.0, 100.0, 200.0, 300.0])
    groups = np.array([0, 0, 1, 0, 1, 1])
    folds = np.array([0, 0, 0, 1, 1, 1])
    oof = av._oof_group_mean(y, groups, folds)
    # fold 0 rows: means from fold 1 (rows 3,4,5):
    #   group 0 -> row3 only = 100; group 1 -> rows 4,5 = 250
    assert oof[0] == pytest.approx(100.0)   # row0 (g0, fold0) <- row3 mean
    assert oof[1] == pytest.approx(100.0)   # row1 (g0, fold0)
    assert oof[2] == pytest.approx(250.0)   # row2 (g1, fold0) <- rows4,5 mean
    # fold 1 rows: means from fold 0 (rows 0,1,2):
    #   group 0 -> rows 0,1 = 5; group 1 -> row2 = 20
    assert oof[3] == pytest.approx(5.0)     # row3 (g0, fold1)
    assert oof[4] == pytest.approx(20.0)    # row4 (g1, fold1)
    assert oof[5] == pytest.approx(20.0)    # row5 (g1, fold1)


def test_empty_group_falls_back_to_global() -> None:
    y = np.array([1.0, 2.0, 3.0, 100.0])
    groups = np.array([0, 0, 0, 1])
    folds = np.array([0, 0, 0, 1])
    oof = av._oof_group_mean(y, groups, folds, global_mean=7.0)
    # row3 (g1, fold1): no g1 rows in fold0 -> fallback to global_mean 7.0
    assert oof[3] == pytest.approx(7.0)
