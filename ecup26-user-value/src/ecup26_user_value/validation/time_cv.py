"""Time-based cross-validation for E-CUP T3 user-value prediction.

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible) and
ticket T3.6.11.

Time-based CV prevents random leakage when modeling time-series data.
- 4 chronological folds
- Each fold = ~14 days validation window
- gap_days between train and val to prevent label leakage

Provides:
- TimeBasedCV(n_splits=4, gap_days=7, val_window_days=14)
- .split(df) -> Iterator of (train_df, val_df) tuples
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import polars as pl


class TimeBasedCV:
    """Chronological time-based CV splitter.

    Usage:
        cv = TimeBasedCV(n_splits=4, gap_days=7, val_window_days=14)
        for train_df, val_df in cv.split(df):
            model.fit(train_df)
            preds = model.predict(val_df)
    """

    def __init__(
        self,
        n_splits: int = 4,
        gap_days: int = 7,
        val_window_days: int = 14,
        date_col: str = "event_date",
    ) -> None:
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.val_window_days = val_window_days
        self.date_col = date_col

    def split(self, df: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """Yield (train, val) tuples, each val is a chronological window.

        Strategy: walk forward in time. Each fold = train on all data
        before val_start - gap_days, then val on val_window_days.
        """
        if self.date_col not in df.columns:
            raise ValueError(f"Column '{self.date_col}' not in DataFrame. Available: {df.columns}")

        dates = df[self.date_col]
        max_date = dates.max()
        min_date = dates.min()

        # Walk forward: val windows stack back-to-back from max_date
        for i in range(self.n_splits):
            # i=0 is the latest fold, i=1 the previous, etc.
            val_end = max_date - timedelta(days=i * self.val_window_days)
            val_start = val_end - timedelta(days=self.val_window_days - 1)
            train_end = val_start - timedelta(days=self.gap_days + 1)

            if train_end < min_date:
                continue

            val_mask = (dates >= val_start) & (dates <= val_end)
            train_mask = dates <= train_end

            val_df = df.filter(val_mask)
            train_df = df.filter(train_mask)

            if train_df.is_empty() or val_df.is_empty():
                continue

            yield train_df, val_df
