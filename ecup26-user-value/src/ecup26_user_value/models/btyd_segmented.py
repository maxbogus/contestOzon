"""Per-segment BTYD for E-CUP T3 user-value prediction.

Per .clinerules/02-ecup-rules.md (R1 open-source, R7 single HF pipeline, R8 Docker
contract) and ticket T3.6.14.

Wraps the existing BTYDBaseline (T3.6.4) with one model per known segment plus a
global fallback. Theoretical backing:
- GrowthGR / ItemLTV value decomposition (T3.6.1, docs/research/growth-gr.md)
- Per-segment BTYD comes from T3.5.2 (features/btyd_segments.py)

Provides:
- SegmentedBTYD.fit(df, segment_col, customer_id_col, order_date_col, price_col,
  ref_date) -> self
- SegmentedBTYD.predict(df, segment_col, ref_date) -> np.ndarray of len(df)
- SegmentedBTYD.save(path) / load(path) - pickle of hyperparameters only (R8)

Per-segment model is dropped to the global fallback if:
- the segment has fewer than MIN_TRANSACTIONS_PER_SEGMENT rows, OR
- the per-segment Gamma-Gamma fit raises (insufficient monetary variance).

The fallback is logged, not swallowed silently (00-coding-ethics).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ecup26_user_value.models.btyd_baseline import BTYDBaseline

logger = logging.getLogger(__name__)

#: Segments recognized by the model. Match features.btyd_segments.SEGMENT_TYPES.
KNOWN_SEGMENTS: tuple[str, ...] = ("search", "cat")

#: Below this many rows, a per-segment model is unreliable -> use global fallback.
MIN_TRANSACTIONS_PER_SEGMENT: int = 20


class SegmentedBTYD:
    """Per-segment BTYD with global fallback."""

    def __init__(
        self,
        penalizer_coef: float = 0.01,
        time_period_days: int = 30,
        known_segments: tuple[str, ...] = KNOWN_SEGMENTS,
        min_transactions_per_segment: int = MIN_TRANSACTIONS_PER_SEGMENT,
    ) -> None:
        self.penalizer_coef = penalizer_coef
        self.time_period_days = time_period_days
        self.known_segments = known_segments
        self.min_transactions_per_segment = min_transactions_per_segment
        self._by_segment: dict[str, BTYDBaseline] = {}
        self._global: BTYDBaseline | None = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        segment_col: str = "type",
        customer_id_col: str = "user_id",
        order_date_col: str = "event_date",
        price_col: str = "price",
        ref_date: pd.Timestamp | str | None = None,
    ) -> SegmentedBTYD:
        """Fit per-segment models + global fallback."""
        if ref_date is None:
            ref_date = df[order_date_col].max()
        if isinstance(ref_date, str):
            ref_date = pd.Timestamp(ref_date)

        # Global model sees all rows
        self._global = BTYDBaseline(
            penalizer_coef=self.penalizer_coef,
            time_period_days=self.time_period_days,
        ).fit(
            df,
            customer_id_col=customer_id_col,
            order_date_col=order_date_col,
            price_col=price_col,
        )

        # Per-segment models
        for seg in self.known_segments:
            seg_df = df[df[segment_col] == seg]
            model = self._fit_one_segment(seg_df, seg, customer_id_col, order_date_col, price_col)
            if model is not None:
                self._by_segment[seg] = model

        self._fitted = True
        return self

    def _fit_one_segment(
        self,
        seg_df: pd.DataFrame,
        seg: str,
        customer_id_col: str,
        order_date_col: str,
        price_col: str,
    ) -> BTYDBaseline | None:
        """Fit a single per-segment BTYDBaseline, or return None if degenerate."""
        if len(seg_df) < self.min_transactions_per_segment:
            logger.warning(
                "SegmentedBTYD: segment %r has %d rows (<%d), using global fallback",
                seg,
                len(seg_df),
                self.min_transactions_per_segment,
            )
            return None
        try:
            return BTYDBaseline(
                penalizer_coef=self.penalizer_coef,
                time_period_days=self.time_period_days,
            ).fit(
                seg_df,
                customer_id_col=customer_id_col,
                order_date_col=order_date_col,
                price_col=price_col,
            )
        except Exception as exc:
            logger.warning(
                "SegmentedBTYD: per-segment fit failed for %r (%s), using global fallback",
                seg,
                exc,
            )
            return None

    def predict(
        self,
        df: pd.DataFrame,
        segment_col: str = "type",
        customer_id_col: str = "user_id",
        order_date_col: str = "event_date",
        price_col: str = "price",
        ref_date: pd.Timestamp | str | None = None,
    ) -> np.ndarray:
        """Predict per-row LTV. Rows with unknown segment or degenerate segment
        model fall back to the global model.

        Returns np.ndarray of shape (len(df),) with non-negative floats.
        """
        if not self._fitted or self._global is None:
            raise RuntimeError("SegmentedBTYD not fitted. Call fit() first.")
        if ref_date is None:
            ref_date = df[order_date_col].max()
        if isinstance(ref_date, str):
            ref_date = pd.Timestamp(ref_date)

        if len(df) == 0:
            return np.array([], dtype=np.float64)

        # Pre-compute predictions per known segment in a single batch
        per_seg_pred: dict[str, np.ndarray] = {}
        for seg, model in self._by_segment.items():
            seg_df = df[df[segment_col] == seg]
            if len(seg_df) == 0:
                continue
            per_seg_pred[seg] = model.predict(
                seg_df,
                customer_id_col=customer_id_col,
                order_date_col=order_date_col,
                price_col=price_col,
            )

        # Global fallback predictions: compute once over the whole df
        global_pred = self._global.predict(
            df,
            customer_id_col=customer_id_col,
            order_date_col=order_date_col,
            price_col=price_col,
        )

        # Assemble per-row: per-segment model if available, else global
        result = np.array(global_pred, dtype=np.float64, copy=True)
        for seg, seg_pred in per_seg_pred.items():
            mask = df[segment_col].values == seg
            result[mask] = seg_pred

        # Clip at zero (RMSLE requires max(0, pred))
        np.maximum(result, 0.0, out=result)
        return result

    def save(self, path: str | Path) -> Path:
        """Persist hyperparameters to disk (R8 Docker contract).

        Per-segment and global BTYDBaseline internal state uses lambdas that
        can't be pickled; we persist hyperparameters only and require re-fit
        on load. This is deterministic given the same data and seed (R6).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "penalizer_coef": self.penalizer_coef,
            "time_period_days": self.time_period_days,
            "known_segments": list(self.known_segments),
            "min_transactions_per_segment": self.min_transactions_per_segment,
            "fitted": self._fitted,
        }
        with open(out, "wb") as f:
            pickle.dump(state, f)
        return out

    @classmethod
    def load(cls, path: str | Path) -> SegmentedBTYD:
        """Load hyperparameters from disk. Call .fit() after loading (R8)."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(
            penalizer_coef=state["penalizer_coef"],
            time_period_days=state["time_period_days"],
            known_segments=tuple(state["known_segments"]),
            min_transactions_per_segment=state["min_transactions_per_segment"],
        )
