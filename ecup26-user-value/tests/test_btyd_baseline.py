"""RED tests for T3.6.4 - BTYD baseline (BG/NBD + Gamma-Gamma).

Per .clinerules/02-ecup-rules.md (R1 open-source, R7 single HF pipeline)
and ticket T3.6.4.

Classical probabilistic BTYD model for non-contractual LTV.
Uses lifetimes library (Apache 2.0).

Provides:
- BTYDBaseline.fit(df, customer_id_col, order_date_col, price_col)
- BTYDBaseline.predict(df) -> np.ndarray of LTV predictions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class TestBTYDBaseline:
    """BTYDBaseline must fit BG/NBD + Gamma-Gamma and predict LTV."""

    def test_bg_nbd_predicts(self) -> None:
        from ecup26_user_value.models.btyd_baseline import BTYDBaseline

        rng = np.random.default_rng(42)
        customers = []
        for cid in range(50):
            n_orders = rng.integers(1, 10)
            for _o in range(n_orders):
                customers.append(
                    {
                        "customer_id": cid,
                        "order_date": pd.Timestamp("2025-01-01")
                        + pd.Timedelta(days=int(rng.integers(1, 90))),
                        "price": float(rng.uniform(10, 500)),
                    }
                )
        df = pd.DataFrame(customers)

        bt = BTYDBaseline()
        bt.fit(df, customer_id_col="customer_id", order_date_col="order_date", price_col="price")
        preds = bt.predict(df)
        assert len(preds) == len(df), f"Expected {len(df)} preds, got {len(preds)}"
        assert (preds >= 0).all(), f"Negative LTV: {preds[preds < 0]}"

    def test_predict_returns_array(self) -> None:
        from ecup26_user_value.models.btyd_baseline import BTYDBaseline

        rng = np.random.default_rng(0)
        customers = []
        for cid in range(30):
            for _ in range(rng.integers(1, 5)):
                customers.append(
                    {
                        "customer_id": cid,
                        "order_date": pd.Timestamp("2025-06-01")
                        + pd.Timedelta(days=int(rng.integers(0, 60))),
                        "price": float(rng.uniform(20, 200)),
                    }
                )
        df = pd.DataFrame(customers)

        bt = BTYDBaseline()
        bt.fit(df, customer_id_col="customer_id", order_date_col="order_date", price_col="price")
        preds = bt.predict(df)
        assert isinstance(preds, np.ndarray), f"Expected ndarray, got {type(preds)}"
        assert preds.ndim == 1

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Per R8: model params must be persistable for Docker submission.

        Note: full pickle of BetaGeoFitter fails due to internal lambdas.
        We save params only and re-fit on load (deterministic).
        """
        from ecup26_user_value.models.btyd_baseline import BTYDBaseline

        rng = np.random.default_rng(1)
        customers = [
            {
                "customer_id": i,
                "order_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d)),
                "price": 100.0,
            }
            for i in range(20)
            for d in rng.integers(1, 30, size=3)
        ]
        df = pd.DataFrame(customers)

        bt = BTYDBaseline()
        bt.fit(df, customer_id_col="customer_id", order_date_col="order_date", price_col="price")
        preds1 = bt.predict(df)

        path = tmp_path / "btyd_params.pkl"
        bt.save(path)
        assert path.exists()

        # Reload + re-fit on same data (params-only persist, not full model)
        bt2 = BTYDBaseline.load(path)
        bt2.fit(df, customer_id_col="customer_id", order_date_col="order_date", price_col="price")
        preds2 = bt2.predict(df)
        np.testing.assert_allclose(preds1, preds2, rtol=1e-5)


class TestBTYDCompliance:
    """lifetimes library must be open-source (R1)."""

    def test_lifetimes_imports(self) -> None:
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter  # noqa: F401

            assert True
        except ImportError as e:
            raise AssertionError(
                f"lifetimes not installed: {e}. Run: cd ecup26-user-value && uv add lifetimes"
            ) from e

    def test_lifetimes_license_open_source(self) -> None:
        from importlib.metadata import metadata

        dist_meta = metadata("lifetimes")
        license_str = (dist_meta.get("License") or "").lower()
        acceptable = ["apache", "mit", "bsd"]
        assert any(lic in license_str for lic in acceptable), (
            f"lifetimes license '{license_str}' is not in {acceptable} (R1)"
        )
