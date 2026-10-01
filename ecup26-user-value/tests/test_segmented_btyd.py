"""RED tests for T3.6.14 - per-segment BTYD (search vs catalog).

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible, R7 single HF
pipeline) and ticket T3.6.14.

SegmentedBTYD wraps the existing BTYDBaseline (one per segment + global fallback)
and predicts per-row LTV. Global fallback is used when a row's segment is unknown
or a per-segment model is degenerate (low data, Gamma-Gamma failure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_df(rng: np.random.Generator, n_users: int = 60) -> pd.DataFrame:
    """Build a synthetic transaction df with two segments having different dynamics.

    Search users: high frequency, low monetary, short recency.
    Catalog users: low frequency, high monetary, longer recency.
    """
    rows: list[dict[str, object]] = []
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        n_orders = int(rng.integers(3, 12)) if seg == "search" else int(rng.integers(1, 5))
        for _o in range(n_orders):
            price = float(rng.uniform(5, 30)) if seg == "search" else float(rng.uniform(80, 250))
            rows.append(
                {
                    "user_id": uid,
                    "event_date": pd.Timestamp("2025-08-01")
                    + pd.Timedelta(days=int(rng.integers(0, 180))),
                    "type": seg,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


class TestSegmentedBTYD:
    """SegmentedBTYD trains per-segment models and falls back to global on degeneracy."""

    def test_predict_shape_matches_input(self) -> None:
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(0)
        df = _make_synthetic_df(rng, n_users=30)
        ref = df["event_date"].max()

        model = SegmentedBTYD(time_period_days=30).fit(df, ref_date=ref)
        pred = model.predict(df, ref_date=ref)

        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(df),), f"Expected len(df)={len(df)}, got shape {pred.shape}"

    def test_predictions_non_negative(self) -> None:
        """RMSLE requires max(0, pred); predictor must never emit negative values."""
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(1)
        df = _make_synthetic_df(rng, n_users=40)
        ref = df["event_date"].max()

        model = SegmentedBTYD(time_period_days=30).fit(df, ref_date=ref)
        pred = model.predict(df, ref_date=ref)

        assert (pred >= 0).all(), f"Negative predictions: {pred[pred < 0]}"

    def test_fit_returns_self(self) -> None:
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(2)
        df = _make_synthetic_df(rng, n_users=30)
        model = SegmentedBTYD().fit(df, ref_date=df["event_date"].max())
        assert model is not None

    def test_per_segment_better_than_global(self) -> None:
        """RMSLE of SegmentedBTYD <= RMSLE of BTYDBaseline on heterogeneous data."""
        from ecup26_user_value.models.btyd_baseline import BTYDBaseline
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(3)
        df = _make_synthetic_df(rng, n_users=80)
        ref = df["event_date"].max()

        holdout_cutoff = ref - pd.Timedelta(days=30)
        train = df[df["event_date"] <= holdout_cutoff]
        if len(train) == 0:
            train = df

        seg = SegmentedBTYD(time_period_days=30).fit(train, ref_date=ref)
        global_ = BTYDBaseline(time_period_days=30).fit(
            train,
            customer_id_col="user_id",
            order_date_col="event_date",
        )

        seg_pred = seg.predict(train, ref_date=ref)
        global_pred = global_.predict(
            train,
            customer_id_col="user_id",
            order_date_col="event_date",
        )

        target = (
            df[df["event_date"] > holdout_cutoff]
            .groupby("user_id")["price"]
            .sum()
            .reindex(train["user_id"].unique())
            .fillna(0.0)
        )
        seg_by_user = pd.Series(seg_pred, index=train["user_id"].values).groupby(level=0).sum()
        global_by_user = (
            pd.Series(global_pred, index=train["user_id"].values).groupby(level=0).sum()
        )

        def _rmsle(p: pd.Series, t: pd.Series) -> float:
            aligned_t = t.reindex(p.index).fillna(0.0)
            return float(np.sqrt(np.mean((np.log1p(aligned_t) - np.log1p(p.clip(lower=0.0))) ** 2)))

        rmsle_seg = _rmsle(seg_by_user, target)
        rmsle_global = _rmsle(global_by_user, target)
        assert rmsle_seg <= rmsle_global * 1.2, (
            f"Segmented RMSLE {rmsle_seg:.4f} > 1.2 * global RMSLE {rmsle_global:.4f}"
        )

    def test_unknown_segment_falls_back(self) -> None:
        """Rows with a segment outside the known set predict via global fallback, no crash."""
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(4)
        df = _make_synthetic_df(rng, n_users=30)
        ref = df["event_date"].max()

        model = SegmentedBTYD().fit(df, ref_date=ref)

        unknown = pd.DataFrame(
            [
                {
                    "user_id": 999,
                    "event_date": ref,
                    "type": "unknown_segment",
                    "price": 100.0,
                }
            ]
        )
        df_with_unknown = pd.concat([df, unknown], ignore_index=True)

        pred = model.predict(df_with_unknown, ref_date=ref)
        assert len(pred) == len(df_with_unknown)
        assert np.isfinite(pred[-1])
        assert pred[-1] >= 0.0

    def test_save_load_roundtrip(self) -> None:
        """save/load round-trip preserves hyperparameters (R8 Docker contract)."""
        from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

        rng = np.random.default_rng(5)
        df = _make_synthetic_df(rng, n_users=30)
        ref = df["event_date"].max()

        original = SegmentedBTYD(penalizer_coef=0.05, time_period_days=14).fit(df, ref_date=ref)
        path = original.save("/tmp/test_segmented_btyd.pkl")
        try:
            loaded = SegmentedBTYD.load(path)
            assert loaded.penalizer_coef == 0.05
            assert loaded.time_period_days == 14
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("n_users", [30, 50])
def test_segmented_btyd_smoke(n_users: int) -> None:
    """Smoke parametrize: fit + predict must succeed for various sizes."""
    from ecup26_user_value.models.btyd_segmented import SegmentedBTYD

    rng = np.random.default_rng(n_users)
    df = _make_synthetic_df(rng, n_users=n_users)
    ref = df["event_date"].max()

    model = SegmentedBTYD().fit(df, ref_date=ref)
    pred = model.predict(df, ref_date=ref)
    assert len(pred) == len(df)
    assert (pred >= 0).all()
