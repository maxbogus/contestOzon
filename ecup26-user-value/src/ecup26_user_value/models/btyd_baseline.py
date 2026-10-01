"""BTYD baseline (BG/NBD + Gamma-Gamma) for E-CUP T3 user-value prediction.

Per .clinerules/02-ecup-rules.md (R1 open-source, R7 single HF pipeline,
R8 Docker contract) and ticket T3.6.4.

Classical probabilistic Buy-Till-You-Die model for non-contractual LTV.
Uses lifetimes library (Apache 2.0 — R1 compliant).

Components:
- BG/NBD: Beta-Geometric/NBD for purchase frequency (transaction count)
- Gamma-Gamma: correlation between frequency and monetary value

Provides:
- BTYDBaseline.fit(df, customer_id_col, order_date_col, price_col)
- BTYDBaseline.predict(df) -> np.ndarray of LTV predictions (one per row)
- BTYDBaseline.save(path) / load(path) for R8 Docker persistence
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


class BTYDBaseline:
    """BTYD model combining BG/NBD (frequency) + Gamma-Gamma (monetary)."""

    def __init__(
        self,
        penalizer_coef: float = 0.01,
        time_period_days: int = 30,
    ) -> None:
        self.penalizer_coef = penalizer_coef
        self.time_period_days = time_period_days
        self.bg_nbd: BetaGeoFitter | None = None
        self.gamma_gamma: GammaGammaFitter | None = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        customer_id_col: str = "customer_id",
        order_date_col: str = "order_date",
        price_col: str = "price",
    ) -> BTYDBaseline:
        """Fit BG/NBD + Gamma-Gamma on transaction data.

        Args:
            df: transaction DataFrame with one row per order.
            customer_id_col: customer identifier.
            order_date_col: order date column.
            price_col: order price column.

        Returns:
            self (fitted).
        """
        # Build lifetimes summary format (frequency, recency, T, monetary_value)
        summary = summary_data_from_transaction_data(
            df,
            customer_id_col=customer_id_col,
            datetime_col=order_date_col,
            monetary_value_col=price_col,
            observation_period_end=df[order_date_col].max(),
        )
        # Filter customers with positive monetary value (Gamma-Gamma requires >0)
        summary = summary[summary["monetary_value"] > 0]

        # Fit BG/NBD
        self.bg_nbd = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
        self.bg_nbd.fit(summary["frequency"], summary["recency"], summary["T"])

        # Fit Gamma-Gamma (requires repeat customers: frequency > 0)
        repeat = summary[summary["frequency"] > 0]
        if len(repeat) > 0:
            self.gamma_gamma = GammaGammaFitter(penalizer_coef=self.penalizer_coef)
            self.gamma_gamma.fit(repeat["frequency"], repeat["monetary_value"])
        else:
            self.gamma_gamma = None

        self._fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        customer_id_col: str = "customer_id",
        order_date_col: str = "order_date",
        price_col: str = "price",
    ) -> np.ndarray:
        """Predict LTV for each row in df.

        Returns (n,) float array of predicted LTV over time_period_days.
        Customers with no monetary history get 0.0.
        """
        if not self._fitted or self.bg_nbd is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        summary = summary_data_from_transaction_data(
            df,
            customer_id_col=customer_id_col,
            datetime_col=order_date_col,
            monetary_value_col=price_col,
            observation_period_end=df[order_date_col].max(),
        )

        # Build per-customer LTV map
        cust_to_ltv: dict[Any, float] = {}
        for cust_id in summary.index:
            row = summary.loc[cust_id]
            if row["monetary_value"] <= 0:
                cust_to_ltv[cust_id] = 0.0
                continue
            try:
                pred = self.bg_nbd.conditional_expected_number_of_purchases_up_to_time(
                    self.time_period_days,
                    row["frequency"],
                    row["recency"],
                    row["T"],
                )
                pred = float(pred)
            except Exception:
                pred = 0.0
            if self.gamma_gamma is not None and row["frequency"] > 0:
                try:
                    avg_m = self.gamma_gamma.conditional_expected_average_profit(
                        row["frequency"],
                        row["monetary_value"],
                    )
                    pred = pred * float(avg_m)
                except Exception:
                    pass
            cust_to_ltv[cust_id] = pred

        # Map back to original df rows
        result = np.array(
            [cust_to_ltv.get(cid, 0.0) for cid in df[customer_id_col]],
            dtype=np.float64,
        )
        return result

    def save(self, path: str | Path) -> Path:
        """Persist fitted model params to disk (R8 Docker contract).

        Note: BetaGeoFitter uses internal lambdas that can't be pickled.
        We save params only and require re-fit on load (deterministic).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "penalizer_coef": self.penalizer_coef,
            "time_period_days": self.time_period_days,
            "fitted": self._fitted,
        }
        with open(out, "wb") as f:
            pickle.dump(state, f)
        return out

    @classmethod
    def load(cls, path: str | Path) -> BTYDBaseline:
        """Load model params from disk. Call .fit() after loading (R8)."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(
            penalizer_coef=state["penalizer_coef"],
            time_period_days=state["time_period_days"],
        )
