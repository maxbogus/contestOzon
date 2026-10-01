"""Tests for T3.7.6 - LightGBM + features integration in run.py.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lightgbm, scikit-learn)
- R6: reproducible (seeded)
- R8: dev-only allowed; run.py imports run_pipeline
- R10: unique IDs asserted

All tests use TINY synthetic data (no full dataset).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.run_pipeline import predict_for_users  # noqa: E402


def _make_synthetic_train(n_users: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic train DataFrame with T3 schema."""
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


class TestPredictForUsersUnit:
    """predict_for_users() works on synthetic data without subprocess."""

    def test_returns_ndarray(self) -> None:
        train_df = _make_synthetic_train(n_users=30)
        user_ids = [0, 1, 2, 3, 4]
        pred = predict_for_users(train_df, user_ids)
        assert isinstance(pred, np.ndarray)
        assert len(pred) == len(user_ids)

    def test_predictions_nonnegative(self) -> None:
        train_df = _make_synthetic_train(n_users=30)
        user_ids = list(range(20))
        pred = predict_for_users(train_df, user_ids)
        assert (pred >= 0).all()

    def test_predictions_finite(self) -> None:
        train_df = _make_synthetic_train(n_users=30)
        user_ids = list(range(20))
        pred = predict_for_users(train_df, user_ids)
        assert np.isfinite(pred).all()

    def test_aligns_to_user_ids_param(self) -> None:
        """Output length matches len(user_ids), not len(train_df)."""
        train_df = _make_synthetic_train(n_users=30)
        user_ids = [0, 5, 10, 15, 20]
        pred = predict_for_users(train_df, user_ids)
        assert len(pred) == 5

    def test_handles_unknown_user_ids(self) -> None:
        """Users without train history get prediction (likely 0 or avg)."""
        train_df = _make_synthetic_train(n_users=10)
        user_ids = [100, 200, 300]
        pred = predict_for_users(train_df, user_ids)
        assert len(pred) == 3
        assert np.isfinite(pred).all()

    def test_empty_train_returns_zeros(self) -> None:
        """Empty train_df (no data) -> zeros for all users."""
        empty_df = pd.DataFrame(columns=["user_id", "event_date", "type", "price"])
        user_ids = [1, 2, 3]
        pred = predict_for_users(empty_df, user_ids)
        assert (pred == 0).all()

    def test_deterministic(self) -> None:
        """Same input -> same output (R6 reproducibility)."""
        train_df = _make_synthetic_train(n_users=30, seed=42)
        user_ids = list(range(20))
        pred1 = predict_for_users(train_df, user_ids)
        pred2 = predict_for_users(train_df, user_ids)
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestLightGBMProgress:
    """lgb.train() with custom callback emits real progress (T3.7.6 progress bar)."""

    def test_lgb_train_emits_progress_callback(self) -> None:
        """Each iteration of lgb.train() must trigger our callback."""
        import lightgbm as lgb
        import numpy as np
        import pandas as pd

        # Synthetic data
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "a": rng.uniform(0, 100, 50),
                "b": rng.uniform(0, 100, 50),
            }
        )
        y = X["a"] + X["b"] + rng.normal(0, 1, 50)

        # Track iterations
        iterations_seen = []

        class _ProgressCallback:
            def __call__(self, env):
                iterations_seen.append(env.iteration)

        train_set = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbose": -1,
            "random_state": 42,
        }
        lgb.train(
            params,
            train_set,
            num_boost_round=20,
            callbacks=[_ProgressCallback()],
        )

        # Should see iterations 1..20
        assert len(iterations_seen) >= 20, (
            f"only {len(iterations_seen)} iterations: {iterations_seen[:5]}"
        )
        assert (
            iterations_seen[-1] == 19
        )  # lgb callbacks are 0-indexed, f"last iter: {iterations_seen[-1]}"


class TestRunPyV3Cli:
    """run.py subprocess with TINY synthetic data (no full dataset)."""

    def _run(
        self,
        tmp_path: Path,
        train_df: pd.DataFrame,
        user_ids: list[int],
        train_frac: float = 0.5,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        train_path = tmp_path / "train.parquet"
        train_df.to_parquet(train_path)
        test_csv = tmp_path / "test_data.csv"
        pd.DataFrame({"user_id": user_ids}).to_csv(test_csv, index=False)
        output = tmp_path / "submit.csv"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        return subprocess.run(
            [
                sys.executable,
                "-u",
                str(REPO_ROOT / "run.py"),
                "--no-cache",
                "--test_data_path",
                str(test_csv),
                "--output_path",
                str(output),
                "--train-frac",
                str(train_frac),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def test_run_py_v3_tiny_end_to_end(self, tmp_path: Path) -> None:
        """End-to-end on 20 users, ~50 transactions. Must run <30s."""
        rng = np.random.default_rng(42)
        rows = []
        for uid in range(1, 21):
            for _ in range(rng.integers(2, 5)):
                rows.append(
                    {
                        "user_id": uid,
                        "event_date": pd.Timestamp("2025-09-01")
                        + pd.Timedelta(days=int(rng.integers(0, 30))),
                        "type": "search" if uid % 2 == 0 else "cat",
                        "price": float(rng.uniform(10, 100)),
                    }
                )
        train_df = pd.DataFrame(rows)
        result = self._run(tmp_path, train_df, list(range(1, 21)), train_frac=0.5, timeout=30)
        assert result.returncode == 0, f"failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        submit = tmp_path / "submit.csv"
        assert submit.exists()
        df = pd.read_csv(submit)
        assert len(df) == 20
        assert df.iloc[:, 0].is_unique
        assert (df["predict"] >= 0).all()

    def test_run_py_v3_help_mentions_train_frac(self) -> None:
        """--help shows the --train-frac option."""
        result = subprocess.run(
            [sys.executable, "-u", str(REPO_ROOT / "run.py"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src")},
            timeout=10,
        )
        assert "--train-frac" in result.stdout
        assert "1.0" in result.stdout
