"""RED tests for T3.6.11 - time-based CV (4 folds by 14 days).

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible) and
ticket T3.6.11.

Time-based CV prevents random leakage when modeling time-series data.
4 folds, each ~14 days, with gap_days to prevent label leakage.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


class TestTimeBasedCV:
    """TimeBasedCV must produce chronological (train, val) splits."""

    def test_yields_n_splits(self) -> None:
        from ecup26_user_value.validation.time_cv import TimeBasedCV

        cv = TimeBasedCV(n_splits=4, gap_days=7)
        # 80 days of data
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(80)]
        df = pl.DataFrame({"event_date": dates, "value": range(80)})

        splits = list(cv.split(df))
        assert len(splits) == 4, f"Expected 4 splits, got {len(splits)}"

    def test_splits_are_chronological(self) -> None:
        """Each split's train period must precede val period."""
        from ecup26_user_value.validation.time_cv import TimeBasedCV

        cv = TimeBasedCV(n_splits=4, gap_days=7)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(80)]
        df = pl.DataFrame({"event_date": dates, "value": range(80)})

        for train, val in cv.split(df):
            train_max = train["event_date"].max()
            val_min = val["event_date"].min()
            assert train_max < val_min, (
                f"Train ends {train_max}, val starts {val_min} - chronological order violated!"
            )

    def test_gap_days_enforced(self) -> None:
        """gap_days ensures no overlap between train and val."""
        from ecup26_user_value.validation.time_cv import TimeBasedCV

        gap = 7
        cv = TimeBasedCV(n_splits=4, gap_days=gap)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(120)]
        df = pl.DataFrame({"event_date": dates, "value": range(120)})

        for train, val in cv.split(df):
            train_max = train["event_date"].max()
            val_min = val["event_date"].min()
            gap_actual = (val_min - train_max).days
            assert gap_actual >= gap, f"Gap {gap_actual} days < required {gap}"

    def test_no_row_overlap(self) -> None:
        """No user/row should appear in both train and val of same split."""
        from ecup26_user_value.validation.time_cv import TimeBasedCV

        cv = TimeBasedCV(n_splits=4, gap_days=7)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(80)]
        df = pl.DataFrame({"event_date": dates, "user_id": range(80)})

        for train, val in cv.split(df):
            train_ids = set(train["user_id"].to_list())
            val_ids = set(val["user_id"].to_list())
            overlap = train_ids & val_ids
            assert not overlap, f"Overlap found: {len(overlap)} rows"

    def test_val_window_size(self) -> None:
        """Each val window should be ~14 days (per ticket)."""
        from ecup26_user_value.validation.time_cv import TimeBasedCV

        cv = TimeBasedCV(n_splits=4, gap_days=7, val_window_days=14)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(120)]
        df = pl.DataFrame({"event_date": dates, "value": range(120)})

        for _train, val in cv.split(df):
            val_min = val["event_date"].min()
            val_max = val["event_date"].max()
            window = (val_max - val_min).days + 1
            # Window = 14 days exactly when data covers full range
            assert 13 <= window <= 15, f"Val window {window} days, expected ~14"
