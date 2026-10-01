"""RED tests for T3.16 reactivation / activity-decay features.

The test window (Feb-Mar 2026) is a post-Q4 decline; the users who still spend
are those ramping back up. These tests verify the reactivation flags,
overdue-recency ratio and the yoy AOV/frequency deltas, all leak-free
(computed only from rows <= ref_date).

Per .clinerules: R1 open-source, R6 seeded/deterministic, R8 test-only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecup26_user_value.features.build_features import build_features


@pytest.fixture(autouse=True)
def _env(monkeypatch) -> None:
    monkeypatch.setenv("ECUP26_REACTIVATION_FEATURES", "1")


def _rows(dates: list[str], uid: int = 1, orders: int = 1, price: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": uid, "event_date": pd.Timestamp(d), "to_ord": orders, "price": price}
            for d in dates
        ]
    )


def test_columns_present() -> None:
    df = _rows(["2025-01-01", "2025-02-20"])
    out = build_features(df, ref_date=pd.Timestamp("2025-03-05"))
    for col in (
        "reactivated_last_7d",
        "reactivated_last_14d",
        "reactivated_last_30d",
        "recency_ratio",
        "activity_acceleration",
    ):
        assert col in out.columns, f"missing {col}"


def test_reactivated_last_7d() -> None:
    """Order after >30d gap, last order 4d before ref -> reactivated_7d = 1."""
    df = _rows(["2025-01-01", "2025-03-01"])
    out = build_features(df, ref_date=pd.Timestamp("2025-03-05"))
    r = out.iloc[0]
    assert r["reactivated_last_7d"] == 1.0
    assert r["reactivated_last_30d"] == 1.0


def test_reactivated_in_30d_not_7d() -> None:
    """Last order 13d before ref (gap 50d) -> 7d=0, 30d=1."""
    df = _rows(["2025-01-01", "2025-02-20"])
    out = build_features(df, ref_date=pd.Timestamp("2025-03-05"))
    r = out.iloc[0]
    assert r["reactivated_last_7d"] == 0.0
    assert r["reactivated_last_14d"] == 1.0
    assert r["reactivated_last_30d"] == 1.0


def test_recency_ratio_overdue() -> None:
    """Weekly buyer dormant 24d -> days_since_last / mean_interval > 2."""
    df = _rows(["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"])
    out = build_features(df, ref_date=pd.Timestamp("2025-02-15"))
    assert out["recency_ratio"].iloc[0] > 2.0


def test_activity_acceleration_dormant_to_active() -> None:
    """No orders in (ref-14, ref-7], then 3 orders in (ref-7, ref] -> big jump."""
    df = _rows(["2025-01-01", "2025-01-05", "2025-01-10", "2025-03-01", "2025-03-02", "2025-03-03"])
    out = build_features(df, ref_date=pd.Timestamp("2025-03-05"))
    assert out["activity_acceleration"].iloc[0] > 0.0


def test_no_yoy_features_schema_uniform() -> None:
    """yoy_* removed (untrainable: needs 2024 window). Schema must match across refs."""
    df = _rows(["2025-01-20", "2025-02-01", "2026-01-20", "2026-02-01"])
    out_2026 = build_features(df, ref_date=pd.Timestamp("2026-02-13"))
    out_2025 = build_features(df, ref_date=pd.Timestamp("2025-06-18"))
    assert "yoy_freq_change" not in out_2026.columns
    assert "yoy_aov_change" not in out_2025.columns
    # same feature set across different ref dates (train/infer schema uniformity)
    assert set(out_2026.columns) == set(out_2025.columns)


def test_leak_free_reactivation() -> None:
    """A future order (after ref_date) must NOT trigger reactivation."""
    df = _rows(["2025-01-01", "2025-03-05"])  # second order IS ref_date -> excluded
    out = build_features(df, ref_date=pd.Timestamp("2025-02-20"))
    assert out["reactivated_last_30d"].iloc[0] == 0.0
