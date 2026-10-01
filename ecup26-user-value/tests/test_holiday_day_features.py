"""RED tests for T3.14/Г1.5 precise holiday-day features (ECUP26_HOLIDAY_DAY_FEATURES).

Two groups:
1. Per-user day-level GMV on the exact 2025 gift days (14.02/23.02/08.03) - leak-free.
2. Per-origin window-structure scalars (days_to_next_gift_day / n_ru_holidays_30d):
   they VARY across walk-forward origins and at infer (2026-02-13) the next gift
   day is 2026-02-14 -> the same "gift-regime" marker the 2025 gift origins carry.

Per .clinerules: R1 open-source, R6 seeded/deterministic, R8 test-only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecup26_user_value.features.build_features import (
    _GIFT_DAYS,
    _days_until_next,
    build_features,
)


@pytest.fixture(autouse=True)
def _holiday_env(monkeypatch) -> None:
    monkeypatch.setenv("ECUP26_HOLIDAY_DAY_FEATURES", "1")


def _tx() -> pd.DataFrame:
    """Small deterministic daily activity: user 1 active Jan-Mar 2025."""
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    return pd.DataFrame(
        [
            {"user_id": 1, "event_date": d, "to_ord": 1, "price": 10.0}
            for d in dates
        ]
    )


def test_holiday_day_columns_present() -> None:
    out = build_features(_tx(), ref_date=pd.Timestamp("2025-06-30"))
    for col in (
        "gmv_giftday_2025_02",
        "gmv_giftday_2025_03",
        "n_gift_holiday_days_2025",
        "days_to_next_gift_day",
        "days_to_next_ru_holiday",
        "n_ru_holidays_30d",
    ):
        assert col in out.columns, f"missing {col}"


def test_infer_gift_regime_marker() -> None:
    """At the test ref (2026-02-13) the next gift day is 2026-02-14 -> 1 day."""
    df = _tx()
    out = build_features(df, ref_date=pd.Timestamp("2026-02-13"))
    assert out["days_to_next_gift_day"].iloc[0] == 1.0


def test_leak_free_at_gift_origin() -> None:
    """At origin 2025-02-13 the 14.02/23.02/08.03 GMV is in the future -> 0."""
    df = _tx()
    out = build_features(df, ref_date=pd.Timestamp("2025-02-13"))
    assert out["gmv_giftday_2025_02"].iloc[0] == 0.0
    assert out["n_gift_holiday_days_2025"].iloc[0] == 0.0


def test_scalars_vary_across_origins() -> None:
    """Window-structure scalars differ between gift and non-gift origins."""
    df = _tx()
    gift = build_features(df, ref_date=pd.Timestamp("2025-02-13"))
    mid = build_features(df, ref_date=pd.Timestamp("2025-06-18"))
    assert gift["days_to_next_gift_day"].iloc[0] == 1.0
    assert mid["days_to_next_gift_day"].iloc[0] > 30.0
    assert gift["n_ru_holidays_30d"].iloc[0] >= 4.0


def test_gmv_giftday_positive_for_2025_active_user() -> None:
    """A user active on 14.02.2025 gets positive gmv_giftday_2025_02."""
    df = _tx()  # active every day Jan-Mar 2025, so 14.02 included
    out = build_features(df, ref_date=pd.Timestamp("2025-06-30"))
    assert out["gmv_giftday_2025_02"].iloc[0] > 0.0
    assert out["n_gift_holiday_days_2025"].iloc[0] == 3.0  # 14.02, 23.02, 08.03


def test_days_until_next_2026() -> None:
    assert _days_until_next(set(_GIFT_DAYS), pd.Timestamp("2026-02-13")) == 1.0
    assert _days_until_next(set(_GIFT_DAYS), pd.Timestamp("2026-02-15")) == 8.0  # 23.02
