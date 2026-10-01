"""Tests for the dev-resume cache (features + LightGBM model).

Per .clinerules/02-ecup-rules.md:
- R4: cache is local-disk only (no network)
- R6: deterministic; the cache key flips when data or model config changes
- R8: a fresh cache dir in eval -> cache miss -> full pipeline (still correct)
- R10: submit.csv uniqueness is unaffected by caching
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value import (  # noqa: E402
    caching,
    run_pipeline,
)
from ecup26_user_value.run_pipeline import predict_for_users  # noqa: E402


def _make_synthetic_train(n_users: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic train DataFrame with the T3 schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-09-01", periods=60, freq="D")
    rows = []
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        for _ in range(int(rng.integers(3, 10))):
            rows.append(
                {
                    "user_id": uid,
                    "event_date": dates[int(rng.integers(0, len(dates)))],
                    "type": seg,
                    "price": float(rng.uniform(5, 100)),
                }
            )
    return pd.DataFrame(rows)


def _user_ids() -> list[int]:
    return list(range(0, 25, 3))


class TestCacheKey:
    """feature_cache_key() is stable and sensitive to data changes."""

    def test_key_stable_for_same_data(self) -> None:
        df = _make_synthetic_train(seed=1)
        assert caching.feature_cache_key(df) == caching.feature_cache_key(df)

    def test_key_changes_when_price_changes(self) -> None:
        df = _make_synthetic_train(seed=1)
        df2 = df.copy()
        df2.loc[df2.index[0], "price"] += 1.0
        assert caching.feature_cache_key(df) != caching.feature_cache_key(df2)

    def test_key_changes_when_rows_change(self) -> None:
        df = _make_synthetic_train(seed=1)
        df2 = df.drop(index=df.index[0]).reset_index(drop=True)
        assert caching.feature_cache_key(df) != caching.feature_cache_key(df2)

    def test_key_changes_with_model_config(self) -> None:
        df = _make_synthetic_train(seed=1)
        cfg = {"params": {"num_leaves": 15}, "num_boost_round": 50}
        other = {"params": {"num_leaves": 31}, "num_boost_round": 50}
        assert caching.feature_cache_key(df, model_config=cfg) != caching.feature_cache_key(
            df, model_config=other
        )


class TestPredictForUsersCache:
    """predict_for_users() creates, reuses and invalidates the cache."""

    def test_first_run_creates_cache_files(self, tmp_path: Path) -> None:
        df = _make_synthetic_train(seed=7)
        preds = predict_for_users(df, _user_ids(), cache_dir=tmp_path)
        assert np.isfinite(preds).all() and (preds >= 0).all()
        files = {p.name for p in tmp_path.iterdir()}
        assert any(f.startswith("features_") and f.endswith(".parquet") for f in files)
        assert any(f.startswith("model_") and f.endswith(".txt") for f in files)

    def test_cache_hit_skips_feature_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _make_synthetic_train(seed=7)
        pred1 = predict_for_users(df, _user_ids(), cache_dir=tmp_path)

        def _boom(train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001
            raise AssertionError("feature build must be skipped on cache hit")

        monkeypatch.setattr(run_pipeline, "_build_features_for_users", _boom)
        pred2 = predict_for_users(df, _user_ids(), cache_dir=tmp_path)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_cache_hit_skips_model_fit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _make_synthetic_train(seed=7)
        pred1 = predict_for_users(df, _user_ids(), cache_dir=tmp_path)

        def _boom(features: pd.DataFrame, y_train: pd.Series) -> object:  # noqa: ARG001
            raise AssertionError("model fit must be skipped on cache hit")

        monkeypatch.setattr(run_pipeline, "_fit_lightgbm", _boom)
        pred2 = predict_for_users(df, _user_ids(), cache_dir=tmp_path)
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_no_cache_writes_nothing(self, tmp_path: Path) -> None:
        df = _make_synthetic_train(seed=3)
        predict_for_users(df, _user_ids(), cache_dir=tmp_path, use_cache=False)
        assert list(tmp_path.iterdir()) == []

    def test_data_change_invalidates_cache(self, tmp_path: Path) -> None:
        df = _make_synthetic_train(seed=5)
        predict_for_users(df, _user_ids(), cache_dir=tmp_path)
        files_before = {p.name for p in tmp_path.iterdir()}

        df2 = df.copy()
        df2.loc[df2.index[0], "price"] += 100.0
        predict_for_users(df2, _user_ids(), cache_dir=tmp_path)
        files_after = {p.name for p in tmp_path.iterdir()}

        # A new key must produce new cache files (old ones may linger).
        assert files_after != files_before

    def test_cached_run_matches_fresh_run(self, tmp_path: Path) -> None:
        df = _make_synthetic_train(seed=9)
        user_ids = _user_ids()
        fresh = predict_for_users(df, user_ids, cache_dir=tmp_path, use_cache=False)
        cached = predict_for_users(df, user_ids, cache_dir=tmp_path)
        np.testing.assert_array_almost_equal(fresh, cached)


class TestRunPyCliCache:
    """run.py CLI: --no-cache / --cache-dir flags and end-to-end resume."""

    def test_help_shows_cache_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, "-u", str(REPO_ROOT / "run.py"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert "--no-cache" in result.stdout
        assert "--cache-dir" in result.stdout

    def test_run_py_resume_end_to_end(self, tmp_path: Path) -> None:
        # Data spans 60 days so the 30-day feature window is non-empty and the
        # full train+cache path is exercised (not the tiny-data fallback).
        rng = np.random.default_rng(42)
        rows = []
        for uid in range(1, 21):
            for _ in range(int(rng.integers(2, 5))):
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": pd.Timestamp("2025-09-01")
                        + pd.Timedelta(days=int(rng.integers(0, 60))),
                        "type": "search" if uid % 2 == 0 else "cat",
                        "price": float(rng.uniform(10, 100)),
                    }
                )
        train_df = pd.DataFrame(rows)
        train_path = tmp_path / "train.parquet"
        train_df.to_parquet(train_path)
        test_csv = tmp_path / "test_data.csv"
        pd.DataFrame({"user_id": list(range(1, 21))}).to_csv(test_csv, index=False)
        cache_dir = tmp_path / "cache"
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

        def _run(out_path: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "run.py"),
                    "--test_data_path",
                    str(test_csv),
                    "--output_path",
                    str(out_path),
                    "--cache-dir",
                    str(cache_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )

        out1 = tmp_path / "submit1.csv"
        r1 = _run(out1)
        assert r1.returncode == 0, f"first run failed:\n{r1.stderr}"
        assert any(cache_dir.iterdir()), "expected cache files after first run"
        assert "cache miss" in r1.stderr.lower()

        out2 = tmp_path / "submit2.csv"
        r2 = _run(out2)
        assert r2.returncode == 0, f"second run failed:\n{r2.stderr}"
        assert "cache hit" in r2.stderr.lower(), (
            f"expected cache hit on second run, got:\n{r2.stderr}"
        )

        pd.testing.assert_frame_equal(pd.read_csv(out1), pd.read_csv(out2))
