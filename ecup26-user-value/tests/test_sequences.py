"""RED tests for T3.19/E0 sequence dataset (build_sequences).

Verifies: (1) tensor shape, (2) leak-free (future rows never enter the sequence),
(3) label = log1p(GMV over (ref, ref+30]), (4) determinism.

Per .clinerules: R1 open-source, R6 deterministic, R8 test-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_sequences import D, RAW_COLS, build_sequences


def _mk(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """rows = [(event_date, user_id, gmv), ...] with minimal activity columns."""
    base = {"search": 0, "cat": 0, "searches": 0, "to_cart": 0, "to_ord": 0,
            "search_to_cart": 0, "search_to_ord": 0, "cat_to_cart": 0,
            "cat_to_ord": 0, "gmv_search": 0.0, "gmv_cat": 0.0}
    out = []
    for d, u, gmv in rows:
        r = dict(base)
        r["user_id"] = u
        r["event_date"] = pd.Timestamp(d)
        r["gmv"] = gmv
        r["to_ord"] = 1 if gmv > 0 else 0
        out.append(r)
    return pd.DataFrame(out)


def test_shape() -> None:
    df = _mk([("2026-01-10", 1, 10.0), ("2026-01-11", 1, 5.0)])
    X, y, uids = build_sequences(df, pd.Timestamp("2026-01-14"), window=180)
    assert X.shape == (1, 180, D)
    assert len(y) == 1 and len(uids) == 1


def test_leak_free_future_row() -> None:
    """A row AFTER ref must not appear in the sequence (all-zero except structure)."""
    df = _mk([("2026-01-10", 1, 10.0), ("2026-01-20", 1, 999.0)])  # 2nd = future
    X, _, _ = build_sequences(df, pd.Timestamp("2026-01-14"), window=30)
    # future day is at position (2026-01-20 - 2025-12-16) = 35 > 29 -> outside window
    raw = X[0, :, : len(RAW_COLS)]
    assert np.all(raw <= 10.0)  # no 999 leaked


def test_label_exact_window() -> None:
    """y = log1p(sum over (ref, ref+30]); outside rows excluded."""
    df = _mk([
        ("2026-01-10", 1, 100.0),   # before ref: in sequence, NOT label
        ("2026-01-20", 1, 50.0),    # in label window
        ("2026-02-01", 1, 30.0),    # in label window
        ("2026-02-20", 1, 999.0),   # outside (ref+30 = 02-13): NOT label
    ])
    _, y, _ = build_sequences(df, pd.Timestamp("2026-01-14"), window=60)
    assert y[0] == pytest.approx(np.log1p(80.0), abs=1e-4)  # 50 + 30


def test_deterministic() -> None:
    df = _mk([("2026-01-10", 1, 10.0), ("2026-01-11", 2, 5.0)])
    X1, y1, u1 = build_sequences(df, pd.Timestamp("2026-01-14"), window=90)
    X2, y2, u2 = build_sequences(df, pd.Timestamp("2026-01-14"), window=90)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and np.array_equal(u1, u2)


def test_multi_user_alignment() -> None:
    """Two users -> tensor [2, L, D] with correct per-user placement."""
    df = pd.concat([
        _mk([("2026-01-10", 1, 10.0)]),
        _mk([("2026-01-11", 2, 20.0)]),
    ], ignore_index=True)
    X, _, uids = build_sequences(df, pd.Timestamp("2026-01-14"), window=30)
    assert X.shape[0] == 2 and len(uids) == 2
    # user 1 order on 2026-01-10 -> pos = (01-10 - (12-15)) = 26 days from start
    assert X[0, 25, 11] == pytest.approx(10.0)   # gmv col idx 11
    assert X[1, 26, 11] == pytest.approx(20.0)   # user 2 order on 01-11


def test_window_truncation() -> None:
    """window=30: a row 60 days before ref must not appear."""
    df = _mk([("2025-11-15", 1, 999.0), ("2026-01-10", 1, 10.0)])
    X, _, _ = build_sequences(df, pd.Timestamp("2026-01-14"), window=30)
    # 2025-11-15 is 60 days before ref -> outside window; 2026-01-10 -> pos 25
    assert X[0, 25, 11] == pytest.approx(10.0)
    assert np.all(X[0, :, 11] <= 10.0)  # no 999


def test_structural_features() -> None:
    """day_of_week (idx 12) and days_to_ref (idx 13) correct."""
    df = _mk([("2026-01-14", 1, 5.0)])  # ref day itself (2026-01-14 is Wednesday=2)
    X, _, _ = build_sequences(df, pd.Timestamp("2026-01-14"), window=7)
    assert X[0, -1, 12] == pytest.approx(2.0)   # Wednesday
    assert X[0, -1, 13] == pytest.approx(0.0)   # days_to_ref = 0 at ref
    assert X[0, -2, 13] == pytest.approx(1.0)   # day before = 1


def test_label_ref_excluded() -> None:
    """Label = (ref, ref+30]; the ref day itself is excluded."""
    df = _mk([
        ("2026-01-14", 1, 100.0),   # == ref: in sequence, NOT label
        ("2026-01-15", 1, 50.0),    # ref+1: label
        ("2026-02-13", 1, 30.0),    # ref+30: label
    ])
    _, y, _ = build_sequences(df, pd.Timestamp("2026-01-14"), window=60)
    assert y[0] == pytest.approx(np.log1p(80.0), abs=1e-4)
