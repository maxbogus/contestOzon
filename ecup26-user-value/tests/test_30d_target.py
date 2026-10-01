"""Tests for the 30-day-target fix (T3 quality gate).

Regression test for the critical bug: the production pipeline used to train
LightGBM on *lifetime* cumulative GMV (groupby sum over the whole train),
while the E-CUP metric scores GMV over the *next 30 days*. This made the
model over-predict ~13x and score worse than the naive baseline.

Per .clinerules/02-ecup-rules.md:
- R6: reproducible (seeded)
- R8: dev-only test
- R10: user_id uniqueness unaffected
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecup26_user_value.run_pipeline import (
    TARGET_DAYS,
    _split_for_30d_target,
    predict_for_users,
)


def _make_synthetic_train(
    n_users: int = 60,
    seed: int = 0,
    n_days: int = 120,
) -> pd.DataFrame:
    """Synthetic tx over n_days with repeat + churned users."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-09-01", periods=n_days, freq="D")
    rows = []
    for uid in range(n_users):
        n_orders = int(rng.integers(1, 8))
        for _ in range(n_orders):
            # bias activity toward the recent tail so 30-day vs lifetime differ
            idx = int(rng.integers(int(n_days * 0.5), n_days))
            rows.append(
                {
                    "user_id": uid,
                    "event_date": dates[idx],
                    "type": "search",
                    "price": float(rng.uniform(10, 100)),
                }
            )
    return pd.DataFrame(rows)


class TestSplitFor30dTarget:
    """_split_for_30d_target picks the last 30 days as the label window."""

    def test_target_window_is_last_30_days(self) -> None:
        df = _make_synthetic_train(seed=1)
        feature_df, target_df, feature_cutoff, ref_date = _split_for_30d_target(df)

        assert ref_date == pd.Timestamp(df["event_date"].max())
        assert feature_cutoff == ref_date - pd.Timedelta(days=TARGET_DAYS)
        assert feature_df["event_date"].max() <= feature_cutoff
        assert (target_df["event_date"] > feature_cutoff).all()
        assert (target_df["event_date"] <= ref_date).all()

    def test_target_is_not_lifetime_sum(self) -> None:
        """A user's label equals their last-30d GMV, NOT the full-period sum."""
        df = pd.DataFrame(
            [
                # old activity (100 days before ref) -> big lifetime, zero 30d
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-10-01"),
                    "type": "search",
                    "price": 1000.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2025-10-02"),
                    "type": "search",
                    "price": 2000.0,
                },
                # recent activity (within last 30 days) -> 30d label = 50
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-15"),
                    "type": "search",
                    "price": 20.0,
                },
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-20"),
                    "type": "search",
                    "price": 30.0,
                },
            ]
        )
        feature_df, target_df, _fc, _ref = _split_for_30d_target(df)
        y = target_df.groupby("user_id")["price"].sum()
        # lifetime would be 3050; the 30-day label must be 50
        assert y.loc[1] == 50.0
        # the huge old GMV is in the FEATURE window, not the label window
        assert feature_df.loc[feature_df["price"] == 1000.0].empty is False


class TestPredictionScale:
    """Predictions should be on the 30-day scale and emit zeros for churn."""

    def test_predictions_not_lifetime_scale(self) -> None:
        df = _make_synthetic_train(seed=3, n_days=120)
        # lifetime mean is much larger than the 30-day window mean
        lifetime_mean = float(df.groupby("user_id")["price"].sum().mean())
        pred = predict_for_users(df, df["user_id"].unique())
        assert pred.mean() < lifetime_mean, (
            f"pred mean {pred.mean():.1f} should be below lifetime mean {lifetime_mean:.1f}"
        )
        assert np.isfinite(pred).all() and (pred >= 0).all()

    def test_churned_users_get_small_predictions(self) -> None:
        """Users with recent activity must predict much higher than churned."""
        rng = np.random.default_rng(11)
        ref = pd.Timestamp("2026-02-13")
        rows = []
        # active users: activity spread over [ref-240, ref] so walk-forward
        # origins see features AND positive 30-day targets at many snapshots
        for uid in range(40):
            for _ in range(int(rng.integers(3, 6))):
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": ref - pd.Timedelta(days=int(rng.integers(0, 240))),
                        "type": "search",
                        "price": float(rng.uniform(20, 100)),
                    }
                )
        # churned users: activity only in [ref-400, ref-300] — strictly before
        # the earliest walk-forward origin, so their 30-day labels are all 0
        for uid in range(40, 80):
            for _ in range(int(rng.integers(3, 6))):
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": ref - pd.Timedelta(days=int(rng.integers(300, 400))),
                        "type": "search",
                        "price": float(rng.uniform(20, 100)),
                    }
                )
        df = pd.DataFrame(rows)
        pred = predict_for_users(df, np.arange(80))
        active = pred[:40].mean()
        churned = pred[40:].mean()
        # active users (recent GMV) must be predicted much higher than churned
        assert active > churned * 3, (
            f"active mean {active:.2f} vs churned mean {churned:.2f} — "
            "model is not distinguishing recent activity"
        )


class TestActivityFeatures:
    """Full daily-schema activity features are produced when columns present."""

    def _frame_with_activity(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # user 1: strong search->order funnel in the last 30 days
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-15"),
                    "search": 1,
                    "cat": 0,
                    "has_search_to_cart": 1,
                    "has_search_to_ord": 1,
                    "search_to_cart": 2,
                    "search_to_ord": 1,
                    "cat_to_cart": 0,
                    "cat_to_ord": 0,
                    "gmv_search": 100.0,
                    "gmv_cat": 0.0,
                    "to_cart": 2,
                    "to_ord": 1,
                    "gmv": 100.0,
                    "searches": 5,
                    "type": "search",
                    "price": 100.0,
                },
                # user 2: catalog-only activity, no orders
                {
                    "user_id": 2,
                    "event_date": pd.Timestamp("2026-01-20"),
                    "search": 0,
                    "cat": 1,
                    "has_search_to_cart": 0,
                    "has_search_to_ord": 0,
                    "search_to_cart": 0,
                    "search_to_ord": 0,
                    "cat_to_cart": 1,
                    "cat_to_ord": 0,
                    "gmv_search": 0.0,
                    "gmv_cat": 50.0,
                    "to_cart": 1,
                    "to_ord": 0,
                    "gmv": 50.0,
                    "searches": 0,
                    "type": "cat",
                    "price": 50.0,
                },
            ]
        )

    def test_activity_features_present(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        out = build_features(self._frame_with_activity(), ref_date=pd.Timestamp("2026-02-13"))
        for col in (
            "to_ord_total",
            "searches_total",
            "gmv_search_total",
            "conv_to_ord",
            "gmv_search_share",
            "search_day_share",
            "gmv_search_30d",
            "to_ord_30d",
            "recency_to_ord",
            "recency_gmv",
        ):
            assert col in out.columns, f"missing activity feature {col}"

        u1 = out[out["user_id"] == 1].iloc[0]
        u2 = out[out["user_id"] == 2].iloc[0]
        # user 1 converted (to_ord=1), user 2 did not
        assert u1["to_ord_total"] == 1.0 and u2["to_ord_total"] == 0.0
        assert u1["gmv_search_share"] == 1.0 and u2["gmv_search_share"] == 0.0
        # recency: user 2 never ordered -> never-active bucket (999)
        assert u2["recency_to_ord"] == 999.0

    def test_activity_features_absent_for_minimal_schema(self) -> None:
        from ecup26_user_value.features.build_features import build_features

        minimal = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "event_date": pd.Timestamp("2026-01-15"),
                    "type": "search",
                    "price": 100.0,
                },
            ]
        )
        out = build_features(minimal, ref_date=pd.Timestamp("2026-02-13"))
        assert "recency_to_ord" not in out.columns
        assert "to_ord_total" not in out.columns
