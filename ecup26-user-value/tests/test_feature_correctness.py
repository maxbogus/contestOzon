"""RED tests for T3.17: feature-correctness (calendar / seasonal / rolling / label).

The user suspects bugs in the feature code. These tests verify the STRUCTURE of
features against ground-truth calendar dates and window semantics:

A. Calendar (_RU_HOLIDAYS / _is_holiday / _GIFT_DAYS): exact RU holidays +
   bridge/transfer days + commercial gift days (14.02 both years).
B. Seasonal (_seasonal_features): gift_buyer_2025 window [23.02, 08.03.2025],
   feb_mar_gmv_2025 boundary behaviour, dec_nov_ratio.
C. Rolling + label: right-open rolling windows ((ref-w, ref]) and the
   walk-forward label window ((ref, ref+30]) - no off-by-one / future leak.

Per .clinerules: R1 open-source, R6 seeded/deterministic, R8 test-only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ecup26_user_value.features.build_features import (
    _GIFT_DAYS,
    _is_holiday,
    build_features,
)
from ecup26_user_value.run_pipeline import _walk_forward_origins

# ---------------------------------------------------------------------------
# A. Calendar
# ---------------------------------------------------------------------------

class TestCalendarCorrectness:
    def test_known_ru_holidays(self) -> None:
        for d in ("2025-02-23", "2025-03-08", "2025-05-09", "2026-02-23",
                  "2026-03-08", "2026-05-09", "2026-06-12", "2026-11-04"):
            assert _is_holiday(pd.Timestamp(d)) == 1, d

    def test_bridge_days_are_holidays(self) -> None:
        for d in ("2025-02-22", "2025-03-09", "2025-11-03", "2026-02-21", "2026-03-07"):
            assert _is_holiday(pd.Timestamp(d)) == 1, f"bridge {d}"

    def test_jan_7_christmas(self) -> None:
        assert _is_holiday(pd.Timestamp("2025-01-07")) == 1
        assert _is_holiday(pd.Timestamp("2026-01-07")) == 1

    def test_ordinary_days_not_holidays(self) -> None:
        for d in ("2025-07-01", "2025-09-01", "2026-07-01", "2026-02-15"):
            assert _is_holiday(pd.Timestamp(d)) == 0, d

    def test_gift_days_include_both_years(self) -> None:
        assert (2025, 2, 14) in _GIFT_DAYS
        assert (2026, 2, 14) in _GIFT_DAYS
        assert (2026, 2, 23) in _GIFT_DAYS
        assert (2026, 3, 8) in _GIFT_DAYS

# ---------------------------------------------------------------------------
# B. Seasonal (gift window / feb_mar_gmv / dec_nov_ratio)
# ---------------------------------------------------------------------------

def _mk(orders: list[tuple[str, float]]) -> pd.DataFrame:
    """orders = [(event_date, price), ...] with to_ord=1 each."""
    return pd.DataFrame(
        [
            {"user_id": 1, "event_date": pd.Timestamp(d), "to_ord": 1, "price": p}
            for d, p in orders
        ]
    )


class TestSeasonalCorrectness:
    def test_gift_buyer_2025_inside_window(self) -> None:
        df = _mk([("2025-02-25", 100.0)])  # inside [23.02, 08.03.2025]
        out = build_features(df, ref_date=pd.Timestamp("2025-06-30"))
        assert out["gift_buyer_2025"].iloc[0] == 1.0

    def test_gift_buyer_2025_outside_window(self) -> None:
        df = _mk([("2025-02-20", 100.0)])  # before the window
        out = build_features(df, ref_date=pd.Timestamp("2025-06-30"))
        assert out["gift_buyer_2025"].iloc[0] == 0.0

    def test_feb_mar_gmv_2025_window_sum(self) -> None:
        # 02-20 (outside) + 03-01 (inside) + 03-09 (outside) -> sum of inside only
        df = _mk([("2025-02-20", 10.0), ("2025-03-01", 90.0), ("2025-03-09", 50.0)])
        out = build_features(df, ref_date=pd.Timestamp("2025-06-30"))
        assert out["feb_mar_gmv_2025"].iloc[0] == pytest.approx(float(np.log1p(90.0)))

    def test_dec_nov_ratio(self) -> None:
        df = _mk([("2025-11-15", 100.0), ("2025-12-15", 300.0)])
        out = build_features(df, ref_date=pd.Timestamp("2026-01-14"))
        assert out["dec_nov_ratio"].iloc[0] == pytest.approx(3.0)

    def test_gift_share_window_14_02(self) -> None:
        # gmv_share_gift covers 14.02-08.03 (the E-CUP gift window)
        df = _mk([("2025-02-15", 50.0)])
        out = build_features(df, ref_date=pd.Timestamp("2025-06-30"))
        assert out["gmv_share_gift"].iloc[0] == 1.0  # 100% of total in gift window


# ---------------------------------------------------------------------------
# C. Rolling windows + label window
# ---------------------------------------------------------------------------

class TestRollingAndLabel:
    def test_rolling_7d_right_open(self) -> None:
        """An order exactly at ref-7 must NOT enter gmv_avg_7d but must enter 14d."""
        df = _mk([("2025-06-01", 100.0)])
        ref = pd.Timestamp("2025-06-08")  # ref-7 == 2025-06-01
        out = build_features(df, ref_date=ref)
        assert out["gmv_avg_7d"].iloc[0] == 0.0
        assert out["gmv_avg_14d"].iloc[0] == pytest.approx(100.0 / 14)

    def test_rolling_includes_ref_day(self) -> None:
        """An order ON ref_date must enter the rolling window."""
        df = _mk([("2025-06-08", 50.0)])
        out = build_features(df, ref_date=pd.Timestamp("2025-06-08"))
        assert out["gmv_avg_7d"].iloc[0] == pytest.approx(50.0 / 7)

    def test_label_window_30d_not_overlapping_ref(self) -> None:
        """For ref=2026-02-13 the last origin's label ends AT ref, not past it."""
        origins = _walk_forward_origins(pd.Timestamp("2026-02-13"))
        latest = max(origins)
        assert latest == pd.Timestamp("2026-01-14")
        assert latest + pd.Timedelta(days=30) == pd.Timestamp("2026-02-13")

    def test_extra_origin_guard_keeps_label_inside(self) -> None:
        """ECUP26_EXTRA_ORIGINS adds only origins whose (t, t+30] fits <= ref."""
        os.environ["ECUP26_EXTRA_ORIGINS"] = "2025-02-13,2026-01-20"
        try:
            origins = _walk_forward_origins(pd.Timestamp("2026-02-13"))
            assert pd.Timestamp("2025-02-13") in origins
            assert pd.Timestamp("2026-01-20") not in origins  # 20.01+30 > 13.02
        finally:
            os.environ.pop("ECUP26_EXTRA_ORIGINS", None)
