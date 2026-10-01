"""RED tests for T3.5.2 - BTYD per-segment (Search vs Catalog).

Per .clinerules/02-ecup-rules.md (R1 open-source, R7 single HF pipeline)
and ticket T3.5.2.

Compute BTYD parameters separately for Search users vs Catalog users.
Returns per-user features including BG/NBD parameters + predicted LTV per
segment.

Provides:
- compute_btyd_segments(df, ref_date) -> DataFrame with
  [user_id, segment, frequency, recency, monetary, p_alive, predicted_ltv]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestBTYDSegments:
    """compute_btyd_segments returns per-user BG/NBD parameters per segment."""

    def test_btyd_per_segment(self) -> None:
        from ecup26_user_value.features.btyd_segments import compute_btyd_segments

        # Synthetic data: 30 users, mixed search/cat activity
        rng = np.random.default_rng(0)
        rows = []
        for uid in range(30):
            n_orders = rng.integers(1, 8)
            for o in range(n_orders):
                # 50/50 search vs cat (per the ticket's "search vs cat" distinction)
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": pd.Timestamp("2025-06-01")
                        + pd.Timedelta(days=int(rng.integers(0, 90))),
                        "type": "search" if o % 2 == 0 else "cat",
                        "price": float(rng.uniform(10, 200)),
                    }
                )
        df = pd.DataFrame(rows)

        result = compute_btyd_segments(df, ref_date=df["event_date"].max())
        assert isinstance(result, pd.DataFrame)
        # Required columns
        for col in [
            "user_id",
            "segment",
            "frequency",
            "recency",
            "monetary",
            "p_alive",
            "predicted_ltv",
        ]:
            assert col in result.columns, f"Missing column {col}"
        # Per-user row count: each user produces a row PER segment they engaged in
        assert len(result) >= 30, f"Expected at least 30 user-segments, got {len(result)}"

    def test_segments_are_search_or_cat(self) -> None:
        from ecup26_user_value.features.btyd_segments import compute_btyd_segments

        rows = [
            {
                "user_id": 1,
                "event_date": pd.Timestamp("2025-06-01"),
                "type": "search",
                "price": 100.0,
            },
            {"user_id": 2, "event_date": pd.Timestamp("2025-06-01"), "type": "cat", "price": 50.0},
            {"user_id": 1, "event_date": pd.Timestamp("2025-06-15"), "type": "cat", "price": 75.0},
        ]
        df = pd.DataFrame(rows)

        result = compute_btyd_segments(df, ref_date=df["event_date"].max())
        unique_segs = result["segment"].unique()
        valid_segs = {"search", "cat"}
        assert set(unique_segs).issubset(valid_segs), f"Invalid segments: {unique_segs}"

    def test_user_can_have_multiple_segments(self) -> None:
        """A user active in both search and cat produces 2 rows."""
        from ecup26_user_value.features.btyd_segments import compute_btyd_segments

        rows = [
            {
                "user_id": 1,
                "event_date": pd.Timestamp("2025-06-01"),
                "type": "search",
                "price": 100.0,
            },
            {"user_id": 1, "event_date": pd.Timestamp("2025-06-15"), "type": "cat", "price": 75.0},
        ]
        df = pd.DataFrame(rows)

        result = compute_btyd_segments(df, ref_date=df["event_date"].max())
        # User 1 should produce 2 rows (search + cat)
        user1_rows = result[result["user_id"] == 1]
        assert len(user1_rows) == 2, f"Expected 2 rows for user 1, got {len(user1_rows)}"

    def test_predicted_ltv_non_negative(self) -> None:
        """predicted_ltv must be >= 0 for all rows."""
        from ecup26_user_value.features.btyd_segments import compute_btyd_segments

        rng = np.random.default_rng(5)
        rows = []
        for uid in range(15):
            for o in range(rng.integers(1, 5)):
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": pd.Timestamp("2025-06-01")
                        + pd.Timedelta(days=int(rng.integers(0, 60))),
                        "type": "search" if o % 2 == 0 else "cat",
                        "price": float(rng.uniform(20, 100)),
                    }
                )
        df = pd.DataFrame(rows)

        result = compute_btyd_segments(df, ref_date=df["event_date"].max())
        assert (result["predicted_ltv"] >= 0).all(), (
            f"Negative LTV: {result[result['predicted_ltv'] < 0]['predicted_ltv']}"
        )
