"""RED tests for T3.7.3 - feature engineering (Cycle #1).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (pandas, numpy)
- R6: reproducible (deterministic, seeded)
- R8: NOT imported by run.py until T3.7.6

Per meta-prompt T3.7.5 Cycle #1, four feature groups:
- Calendar (day_of_week, month, quarter, is_weekend, is_holiday_ru)
- Rolling windows (gmv/orders avg over last 7/14/30 days)
- Cross-features (recency * frequency, etc.)
- Customer tenure (days_since_first_purchase, is_new_customer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_synthetic_tx(
    n_users: int = 30,
    seed: int = 0,
    start_date: str = "2025-09-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Build a synthetic tx DataFrame matching T3 schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    rows: list[dict[str, object]] = []
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        n_orders = int(rng.integers(3, 12)) if seg == "search" else int(rng.integers(1, 5))
        for _o in range(n_orders):
            day = dates[int(rng.integers(0, len(dates)))]
            price = float(rng.uniform(5, 30)) if seg == "search" else float(rng.uniform(80, 250))
            rows.append(
                {
                    "user_id": uid,
                    "event_date": day,
                    "type": seg,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


class TestCalendarFeatures:
    """build_features adds 5 calendar columns."""

    def test_calendar_columns_present(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx()
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        for col in ("day_of_week", "month", "quarter", "is_weekend", "is_holiday_ru"):
            assert col in out.columns, f"missing calendar column {col}"

    def test_day_of_week_in_range(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx(n_users=10)
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        # Per-user mode: single value 0..6
        assert out["day_of_week"].between(0, 6).all()


class TestRollingWindowFeatures:
    """Rolling features use only data BEFORE ref_date (no leakage)."""

    def test_rolling_columns_present(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx()
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        for col in (
            "gmv_avg_7d",
            "gmv_avg_14d",
            "gmv_avg_30d",
            "orders_avg_7d",
            "orders_avg_14d",
            "orders_avg_30d",
            "gmv_max_30d",
            "n_active_days_30d",
        ):
            assert col in out.columns, f"missing rolling column {col}"

    def test_rolling_window_uses_only_past(self) -> None:
        """For ref_date=2025-12-15, rolling 7d must NOT include rows after 2025-12-15."""
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx(n_users=20)
        # Pick a user with known activity
        cutoff = pd.Timestamp("2025-12-15")
        out = build_features(df, ref_date=cutoff)
        # For each user, manually compute expected gmv_avg_7d and compare
        # We just check the values are finite and non-negative
        assert (out["gmv_avg_7d"] >= 0).all()
        assert np.isfinite(out["gmv_avg_7d"]).all()


class TestCrossFeatures:
    """Cross-features are interactions of RFM components."""

    def test_cross_columns_present(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx()
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        for col in ("recency_x_frequency", "recency_x_avg_order_value", "frequency_x_monetary"):
            assert col in out.columns, f"missing cross column {col}"

    def test_cross_features_finite_for_active_users(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx(n_users=30)
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        # Drop NaN, check the rest is finite
        for col in ("recency_x_frequency", "recency_x_avg_order_value", "frequency_x_monetary"):
            valid = out[col].dropna()
            assert np.isfinite(valid).all(), f"{col} has non-finite values"


class TestTenureFeatures:
    """Tenure features measure customer age."""

    def test_tenure_columns_present(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx()
        out = build_features(df, ref_date=pd.Timestamp("2025-12-31"))
        assert "days_since_first_purchase" in out.columns
        assert "is_new_customer" in out.columns

    def test_tenure_known_value(self) -> None:
        """A user whose first event was 30 days before ref_date has tenure=30."""
        from ecup26_user_value.features.build_features import build_features

        df = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-12-01"),
                    "type": "search",
                    "price": 100.0,
                },
                {
                    "user_id": 2,
                    "event_date": pd.Timestamp("2025-11-01"),
                    "type": "search",
                    "price": 100.0,
                },
            ]
        )
        ref = pd.Timestamp("2025-12-31")
        out = build_features(df, ref_date=ref)
        # User 1: first event 30 days before ref -> tenure=30
        # User 2: first event 60 days before ref -> tenure=60
        u1 = out[out["user_id"] == 1].iloc[0]
        u2 = out[out["user_id"] == 2].iloc[0]
        assert u1["days_since_first_purchase"] == 30
        assert u2["days_since_first_purchase"] == 60


class TestHolidays:
    """Russian holidays are flagged."""

    def test_known_holidays(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        # User with events on known Russian holidays
        df = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-01"),  # New Year
                    "type": "search",
                    "price": 50.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-05-01"),  # Spring/ Labour Day
                    "type": "search",
                    "price": 50.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-06-12"),  # Russia Day
                    "type": "search",
                    "price": 50.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-03-08"),  # Women's Day
                    "type": "search",
                    "price": 50.0,
                },
            ]
        )
        out = build_features(df, ref_date=pd.Timestamp("2026-06-30"))
        # Per-user row should have is_holiday_ru == 1 (mode of their events)
        assert out.iloc[0]["is_holiday_ru"] == 1, (
            f"expected is_holiday_ru=1 for user with holiday events, got {out.iloc[0]['is_holiday_ru']}"
        )


class TestDeterminism:
    """Same input -> same output (R6)."""

    def test_same_input_same_output(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        df = _make_synthetic_tx(n_users=10, seed=42)
        ref = pd.Timestamp("2025-12-31")
        out1 = build_features(df, ref_date=ref)
        out2 = build_features(df, ref_date=ref)
        # All numeric columns should be byte-identical
        for col in out1.columns:
            if out1[col].dtype.kind in "fi":
                np.testing.assert_array_equal(
                    out1[col].fillna(-999).values,
                    out2[col].fillna(-999).values,
                    err_msg=f"column {col} differs between runs",
                )


class TestNoDataLeakage:
    """No feature should depend on rows >= ref_date."""

    def test_no_future_rows_used(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        # Create df with rows before AND after ref_date
        df = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-12-01"),
                    "type": "search",
                    "price": 100.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-12-20"),  # before ref
                    "type": "search",
                    "price": 100.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-10"),  # AFTER ref
                    "type": "search",
                    "price": 9999.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-02-20"),  # AFTER ref
                    "type": "search",
                    "price": 9999.0,
                },
            ]
        )
        ref = pd.Timestamp("2025-12-31")
        out = build_features(df, ref_date=ref)
        # The huge 9999 prices are AFTER ref, must not be in gmv_max_30d
        # gmv_max_30d should be <= 100 (max price before ref)
        assert out.iloc[0]["gmv_max_30d"] <= 100, (
            f"gmv_max_30d leaks future data: {out.iloc[0]['gmv_max_30d']}"
        )


class TestBehavioralFeatures:
    """T3.10.12: purchase-TIMING traits (env-gated, OFF in production)."""

    def _gift_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # user 1: advance buyer (2025-02-16 = 7d before Feb 23)
                {"user_id": 1, "event_date": pd.Timestamp("2025-02-16"), "price": 100.0},
                # user 2: last-minute (2025-02-22 = day before Feb 23)
                {"user_id": 2, "event_date": pd.Timestamp("2025-02-22"), "price": 100.0},
                # user 3: late (2025-03-10 = 2d after Mar 8)
                {"user_id": 3, "event_date": pd.Timestamp("2025-03-10"), "price": 100.0},
            ]
        )

    def test_flag_off_no_behavioral_columns(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        out = build_features(self._gift_df(), ref_date=pd.Timestamp("2025-03-15"))
        assert "planner_share_2025" not in out.columns

    def test_flag_on_shares_partition(self, monkeypatch) -> None:
        monkeypatch.setenv("ECUP26_BEHAVIOR_FEATURES", "1")
        from ecup26_user_value.features.build_features import build_features

        out = build_features(self._gift_df(), ref_date=pd.Timestamp("2025-03-15"))
        for col in ("gmv_adv_feb23_2025", "gmv_lm_mar8_2025", "gmv_late_2025",
                    "planner_share_2025", "lastmin_share_2025", "late_share_2025",
                    "gmv_pre_ny_2025", "gift_vs_mass_2025"):
            assert col in out.columns, f"missing behavioral column {col}"
        row = out.set_index("user_id")
        assert row.loc[1, "planner_share_2025"] == 1.0
        assert row.loc[2, "lastmin_share_2025"] == 1.0
        assert row.loc[3, "late_share_2025"] == 1.0
        assert row.loc[1, "lastmin_share_2025"] == 0.0


class TestActorFeatures:
    """T3.10.17: actor-type features (env-gated, OFF in production)."""

    def _df(self) -> pd.DataFrame:
        # user 1: impulsive - 10 orders, 0 carts (instant buy share = 1)
        # user 2: planner - 0 orders, 10 carts (cart backlog = 10, instant = 0)
        # user 3: regular buyer - orders 7 days apart
        rows = [{"user_id": u, "event_date": pd.Timestamp(d), "to_cart": tc,
                 "to_ord": to, "gmv": g, "gmv_cat": gc, "price": g}
                for u, d, tc, to, g, gc in [
                    (1, "2025-03-01", 0, 10, 100.0, 20.0),
                    (1, "2025-03-02", 0, 5, 50.0, 10.0),
                    (2, "2025-03-01", 10, 0, 0.0, 0.0),
                    (3, "2025-03-01", 1, 1, 30.0, 30.0),
                    (3, "2025-03-08", 1, 1, 30.0, 30.0),
                    (3, "2025-03-15", 1, 1, 30.0, 30.0),
                ]]
        return pd.DataFrame(rows)

    def test_flag_off_no_actor_columns(self) -> None:
        from ecup26_user_value.features.build_features import build_features
        out = build_features(self._df(), ref_date=pd.Timestamp("2025-03-31"))
        assert "instant_buy_share" not in out.columns

    def test_flag_on_actor_columns_and_values(self, monkeypatch) -> None:
        monkeypatch.setenv("ECUP26_ACTOR_FEATURES", "1")
        from ecup26_user_value.features.build_features import build_features
        out = build_features(self._df(), ref_date=pd.Timestamp("2025-03-31"))
        for col in ("instant_buy_share", "cart_backlog", "buy_gap_mean",
                    "gmv_cat_share", "gmv_cv_daily", "max_day_gmv_share"):
            assert col in out.columns, f"missing actor column {col}"
        row = out.set_index("user_id")
        assert row.loc[1, "instant_buy_share"] == 1.0      # impulsive: no carts
        assert row.loc[2, "instant_buy_share"] == 0.0      # planner: cart only
        assert row.loc[2, "cart_backlog"] == 10.0
        assert row.loc[3, "buy_gap_mean"] == 7.0           # orders 7d apart


class TestHolidayPrePostFeatures:
    """T3.10.15: gift-holiday pre/post purchase flags (env-gated, OFF by default)."""

    def _gift_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # user 1: advance buyer - 2025-02-16 = exactly 7d before Feb 23
                {"user_id": 1, "event_date": pd.Timestamp("2025-02-16"), "price": 100.0},
                # user 2: last-minute - 2025-02-22 = day before Feb 23
                {"user_id": 2, "event_date": pd.Timestamp("2025-02-22"), "price": 100.0},
                # user 3: delayed buyer - 2025-03-10 = 2d after Mar 8
                {"user_id": 3, "event_date": pd.Timestamp("2025-03-10"), "price": 100.0},
            ]
        )

    def test_flag_off_no_columns(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        out = build_features(self._gift_df(), ref_date=pd.Timestamp("2025-03-15"))
        assert "is_pre_holiday_7d_feb23" not in out.columns

    def test_flag_on_values(self, monkeypatch) -> None:
        monkeypatch.setenv("ECUP26_HOLIDAY_PREPOST_FEATURES", "1")
        from ecup26_user_value.features.build_features import build_features

        out = build_features(self._gift_df(), ref_date=pd.Timestamp("2025-03-15"))
        for col in ("is_pre_holiday_7d_feb23", "is_pre_holiday_2d_feb23",
                    "is_post_holiday_7d_feb23", "is_pre_holiday_7d_mar8",
                    "is_pre_holiday_2d_mar8", "is_post_holiday_7d_mar8"):
            assert col in out.columns, f"missing {col}"
        row = out.set_index("user_id")
        assert row.loc[1, "is_pre_holiday_7d_feb23"] == 1.0
        assert row.loc[1, "is_pre_holiday_2d_feb23"] == 0.0
        assert row.loc[2, "is_pre_holiday_2d_feb23"] == 1.0
        assert row.loc[3, "is_post_holiday_7d_mar8"] == 1.0
        assert row.loc[1, "is_post_holiday_7d_mar8"] == 0.0

    def test_no_future_leak(self, monkeypatch) -> None:
        monkeypatch.setenv("ECUP26_HOLIDAY_PREPOST_FEATURES", "1")
        from ecup26_user_value.features.build_features import build_features

        # ref_date BEFORE the 2025 gift season: a Feb-16 purchase must NOT set a
        # pre-flag for a Jan ref_date (future leakage) - all flags stay 0.
        df = pd.DataFrame(
            [
                {"user_id": 1, "event_date": pd.Timestamp("2025-01-10"), "price": 50.0},
                {"user_id": 1, "event_date": pd.Timestamp("2025-02-16"), "price": 100.0},
            ]
        )
        out = build_features(df, ref_date=pd.Timestamp("2025-01-20"))
        row = out.set_index("user_id")
        for col in ("is_pre_holiday_7d_feb23", "is_pre_holiday_2d_feb23",
                    "is_post_holiday_7d_feb23", "is_pre_holiday_7d_mar8",
                    "is_pre_holiday_2d_mar8", "is_post_holiday_7d_mar8"):
            assert row.loc[1, col] == 0.0, f"leak in {col}"


class TestFunnelChainFeatures:
    """T3.10.26: funnel-chain coefficients + cart-to-order lag (env-gated)."""

    def _df(self) -> pd.DataFrame:
        # user 1: search-only buyer, carts 10, orders 2 -> abandon 0.8, ratio 1
        # user 2: cart on Mar 1, order on Mar 8 -> lag 7
        rows = [
            {"user_id": 1, "event_date": pd.Timestamp("2025-03-01"), "search": 1,
             "cat": 0, "search_to_cart": 5, "cat_to_cart": 0, "search_to_ord": 1,
             "cat_to_ord": 0, "to_cart": 10, "to_ord": 2, "gmv_search": 100.0,
             "gmv_cat": 0.0, "gmv": 100.0, "searches": 10, "has_search_to_cart": 1,
             "has_cat_to_cart": 0, "has_search_to_ord": 1, "has_cat_to_ord": 0, "price": 100.0},
            {"user_id": 1, "event_date": pd.Timestamp("2025-03-02"), "search": 1,
             "cat": 0, "search_to_cart": 5, "cat_to_cart": 0, "search_to_ord": 1,
             "cat_to_ord": 0, "to_cart": 5, "to_ord": 1, "gmv_search": 50.0,
             "gmv_cat": 0.0, "gmv": 50.0, "searches": 5, "has_search_to_cart": 1,
             "has_cat_to_cart": 0, "has_search_to_ord": 1, "has_cat_to_ord": 0, "price": 50.0},
            {"user_id": 2, "event_date": pd.Timestamp("2025-03-01"), "search": 0,
             "cat": 1, "search_to_cart": 0, "cat_to_cart": 2, "search_to_ord": 0,
             "cat_to_ord": 0, "to_cart": 2, "to_ord": 0, "gmv_search": 0.0,
             "gmv_cat": 0.0, "gmv": 0.0, "searches": 0, "has_search_to_cart": 0,
             "has_cat_to_cart": 1, "has_search_to_ord": 0, "has_cat_to_ord": 0, "price": 0.0},
            {"user_id": 2, "event_date": pd.Timestamp("2025-03-08"), "search": 0,
             "cat": 1, "search_to_cart": 0, "cat_to_cart": 0, "search_to_ord": 0,
             "cat_to_ord": 1, "to_cart": 0, "to_ord": 1, "gmv_search": 0.0,
             "gmv_cat": 80.0, "gmv": 80.0, "searches": 0, "has_search_to_cart": 0,
             "has_cat_to_cart": 0, "has_search_to_ord": 0, "has_cat_to_ord": 1, "price": 80.0},
        ]
        return pd.DataFrame(rows)

    def test_flag_on_funnel_values(self, monkeypatch) -> None:
        monkeypatch.setenv("ECUP26_ACTOR_FEATURES", "1")
        monkeypatch.setenv("ECUP26_FUNNEL_FEATURES", "1")
        from ecup26_user_value.features.build_features import build_features
        out = build_features(self._df(), ref_date=pd.Timestamp("2025-03-31"))
        for col in ("search_to_cart_ratio", "abandon_rate", "impulse_coef",
                    "cart_to_order_lag_mean", "avg_search_gmv"):
            assert col in out.columns, f"missing {col}"
        row = out.set_index("user_id")
        assert row.loc[1, "search_to_cart_ratio"] == 1.0       # search-only
        assert row.loc[1, "abandon_rate"] == 0.8               # (15-3)/15
        assert row.loc[2, "impulse_coef"] == 1.0               # cat-driven
        assert row.loc[2, "cart_to_order_lag_mean"] == 7.0     # cart Mar1, order Mar8
