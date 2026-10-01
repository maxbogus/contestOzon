"""RED tests for T3.14/WS-1 history-length-invariant (normalized) features.

The gift-season training folds (ECUP26_EXTRA_ORIGINS=2025-02-13, -02-23, -03-08)
have SHORT history (~6-9 weeks) while production infer users have ~13 months.
Absolute totals (frequency, monetary, T, gmv) are not comparable across such
origins, so the gift-season label does not transfer. Per-day rates must be
EQUAL for two users whose per-day behavior is identical but whose history
length differs — that is what makes the gift-origin folds usable.

Per .clinerules: R1 open-source, R6 seeded/deterministic, R8 test-only.
"""

from __future__ import annotations

import pandas as pd

from ecup26_user_value.features.build_features import build_features

NORM_COLS = (
    "orders_per_active_day",
    "gmv_per_active_day",
    "activity_rate",
    "gmv_per_day_alive",
    "orders_per_day_alive",
    "recency_rate",
)


def _daily_rows(
    start: str,
    n_days: int,
    orders_per_day: float,
    gmv_per_day: float,
    uid: int = 1,
) -> pd.DataFrame:
    """Deterministic daily activity: active every day, fixed orders/GMV."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        [
            {
                "user_id": uid,
                "event_date": d,
                "to_ord": orders_per_day,
                "price": gmv_per_day,
            }
            for d in dates
        ]
    )


def test_normalization_columns_present() -> None:
    """All 6 normalized feature columns are always emitted."""
    df = _daily_rows("2025-01-01", 30, 1.0, 10.0)
    out = build_features(df, ref_date=pd.Timestamp("2025-02-01"))
    for col in NORM_COLS:
        assert col in out.columns, f"missing normalized column {col}"


def test_history_invariance_per_day_rates() -> None:
    """Equal per-day behavior -> equal per-active-day rates regardless of history length."""
    short = _daily_rows("2025-01-01", 14, 1.0, 10.0, uid=1)   # 2-week history
    long = _daily_rows("2025-01-01", 60, 1.0, 10.0, uid=2)     # 2-month history
    out = build_features(
        pd.concat([short, long], ignore_index=True),
        ref_date=pd.Timestamp("2025-03-02"),
    )
    s = out[out["user_id"] == 1].iloc[0]
    lg = out[out["user_id"] == 2].iloc[0]
    # 14 orders / 14 active days == 60 orders / 60 active days
    assert abs(s["orders_per_active_day"] - lg["orders_per_active_day"]) < 1e-9
    assert abs(s["gmv_per_active_day"] - lg["gmv_per_active_day"]) < 1e-9


def test_higher_rate_monotonic() -> None:
    """A user with a higher per-day order rate gets a higher feature value."""
    slow = _daily_rows("2025-01-01", 14, 0.5, 5.0, uid=1)
    fast = _daily_rows("2025-01-01", 14, 2.0, 20.0, uid=2)
    out = build_features(
        pd.concat([slow, fast], ignore_index=True),
        ref_date=pd.Timestamp("2025-02-01"),
    )
    s = out[out["user_id"] == 1].iloc[0]
    f = out[out["user_id"] == 2].iloc[0]
    assert f["orders_per_active_day"] > s["orders_per_active_day"]
    assert f["gmv_per_active_day"] > s["gmv_per_active_day"]


def test_rates_bounded_and_finite() -> None:
    """Rate features are finite and activity/recency rates live in [0, ~1.01]."""
    df = _daily_rows("2025-01-01", 30, 1.0, 10.0)
    out = build_features(df, ref_date=pd.Timestamp("2025-02-01"))
    for col in NORM_COLS:
        assert out[col].notna().all(), f"{col} has NaN"
        assert (out[col] >= 0).all(), f"{col} has negative values"
    assert out["activity_rate"].max() <= 1.01
    assert out["recency_rate"].max() <= 1.01


def test_never_active_fallback() -> None:
    """A user with zero activity rows (no rows <= ref) -> empty frame, no crash."""
    df = _daily_rows("2025-01-01", 14, 1.0, 10.0)
    out = build_features(df, ref_date=pd.Timestamp("2024-12-31"))
    assert out.empty
