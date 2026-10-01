"""Smoke tests for ecup26-user-value."""

from __future__ import annotations


def test_imports() -> None:
    """Package must be importable."""
    import ecup26_user_value  # noqa: F401


def test_polars_available() -> None:
    """Polars available."""
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.shape == (3, 1)
