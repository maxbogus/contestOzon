"""RED tests for T3.17 per-user Markov state-transition features.

States: 0=browsing, 1=carting, 2=buying (per active day); gap >= 7d = dormant
period ending in a wake-up. All features leak-free (rows <= ref_date).

Per .clinerules: R1 open-source, R6 deterministic, R8 test-only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecup26_user_value.features.build_features import build_features


@pytest.fixture(autouse=True)
def _env(monkeypatch) -> None:
    monkeypatch.setenv("ECUP26_MARKOV_FEATURES", "1")


MARKOV_COLS = (
    "current_state", "days_in_state", "p_buy_active",
    "p_cart_to_buy", "p_browse_to_buy", "p_buy_repeat",
    "n_wakeups_total", "p_wakeup",
)


def _rows(days: list[str], to_cart: int = 0, to_ord: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "event_date": pd.Timestamp(d),
             "to_cart": to_cart, "to_ord": to_ord, "price": 10.0, "search": 1}
            for d in days
        ]
    )


def test_columns_present() -> None:
    out = build_features(_rows(["2025-01-05"]), ref_date=pd.Timestamp("2025-02-01"))
    for c in MARKOV_COLS:
        assert c in out.columns, f"missing {c}"


def test_current_state_buying() -> None:
    """Last active day had an order -> current_state == 2."""
    out = build_features(_rows(["2025-01-05", "2025-01-20"], to_ord=1),
                         ref_date=pd.Timestamp("2025-01-25"))
    assert out["current_state"].iloc[0] == 2.0


def test_current_state_dormant() -> None:
    """No active day in the last 7d -> current_state == 3 (dormant)."""
    out = build_features(_rows(["2025-01-05"]), ref_date=pd.Timestamp("2025-01-20"))
    assert out["current_state"].iloc[0] == 3.0


def test_p_cart_to_buy() -> None:
    """cart(day1) -> buy(day2) is a Cart->Buy transition; ratio should be 1.0."""
    df = pd.DataFrame([
        {"user_id": 1, "event_date": pd.Timestamp("2025-01-10"),
         "to_cart": 1, "to_ord": 0, "price": 0.0, "search": 1},
        {"user_id": 1, "event_date": pd.Timestamp("2025-01-11"),
         "to_cart": 0, "to_ord": 1, "price": 10.0, "search": 0},
    ])
    out = build_features(df, ref_date=pd.Timestamp("2025-02-01"))
    assert out["p_cart_to_buy"].iloc[0] == pytest.approx(1.0)


def test_wakeup_count() -> None:
    """A 10-day gap between active days = 1 wake-up; continuous days = 0."""
    df = _rows(["2025-01-01", "2025-01-11"], to_ord=1)  # gap = 10 days
    out = build_features(df, ref_date=pd.Timestamp("2025-02-01"))
    assert out["n_wakeups_total"].iloc[0] == 1.0

    df2 = _rows(["2025-01-01", "2025-01-02"], to_ord=1)  # gap = 1 day
    out2 = build_features(df2, ref_date=pd.Timestamp("2025-02-01"))
    assert out2["n_wakeups_total"].iloc[0] == 0.0


def test_leak_free() -> None:
    """Future activity (after ref) must not create transitions/wake-ups."""
    df = pd.DataFrame([
        {"user_id": 1, "event_date": pd.Timestamp("2025-01-01"),
         "to_cart": 0, "to_ord": 0, "price": 0.0, "search": 1},
        # order on ref+5 (outside) - must not be counted
        {"user_id": 1, "event_date": pd.Timestamp("2025-02-05"),
         "to_cart": 0, "to_ord": 1, "price": 10.0, "search": 0},
    ])
    out = build_features(df, ref_date=pd.Timestamp("2025-01-15"))
    assert out["n_wakeups_total"].iloc[0] == 0.0
    assert out["current_state"].iloc[0] == 3.0  # dormant (no activity in last 7d)
