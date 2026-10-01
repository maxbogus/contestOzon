"""Tests for the submission entry point run.py (T3.6.16).

Per .clinerules/02-ecup-rules.md R8 (Docker contract):
- entry point: `python -u run.py --test_data_path <path> --output_path <path>`
- T3 specific args per ECUP_RULES.md

Per R10: submit.csv must have unique IDs (no chunk_id collisions, see note 002).

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


def _make_tiny_train_and_test(
    tmp_path: Path,
    n_users: int = 20,
    seed: int = 42,
) -> tuple[Path, list[int]]:
    """Create tiny train.parquet + test_data.csv with n_users."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_users + 1):
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
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path)
    user_ids = list(range(1, n_users + 1))
    test_csv = tmp_path / "test_data.csv"
    pd.DataFrame({"user_id": user_ids}).to_csv(test_csv, index=False)
    return test_csv, user_ids


def _run(
    args: list[str], cwd: Path | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / "run.py"), "--no-cache", *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


class TestRunPyCliContract:
    """R8: CLI must accept --test_data_path and --output_path."""

    def test_help_shows_required_args(self) -> None:
        result = _run(["--help"])
        assert "--test_data_path" in result.stdout, (
            f"--test_data_path missing from --help output:\n{result.stdout}"
        )
        assert "--output_path" in result.stdout, (
            f"--output_path missing from --help output:\n{result.stdout}"
        )

    def test_no_legacy_validate_or_submit_flags(self) -> None:
        """Old stub used --validate/--submit. R8 expects only --test_data_path."""
        result = _run(["--help"])
        for legacy in ("--validate", "--submit"):
            assert legacy not in result.stdout, (
                f"Legacy flag {legacy} still present in --help output (violates R8)"
            )


class TestRunPyEndToEnd:
    """R10 + R8: pipeline produces valid submit.csv (TINY synthetic data)."""

    def test_run_py_produces_unique_ids(self, tmp_path: Path) -> None:
        """20 users, ~50 tx, --train-frac 0.5. Must run <30s."""
        test_csv, user_ids = _make_tiny_train_and_test(tmp_path, n_users=20)
        out = tmp_path / "submit.csv"
        result = _run(
            [
                "--test_data_path",
                str(test_csv),
                "--output_path",
                str(out),
                "--train-frac",
                "0.5",
            ],
            timeout=30,
        )
        assert result.returncode == 0, (
            f"run.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert out.exists(), f"submit.csv not created at {out}"

        df = pd.read_csv(out)
        assert "user_id" in df.columns, f"user_id column missing in {df.columns.tolist()}"
        assert "predict" in df.columns, f"predict column missing in {df.columns.tolist()}"
        assert df.iloc[:, 0].is_unique, (
            f"Duplicate IDs in submit.csv: {df[df.iloc[:, 0].duplicated()].head()}"
        )
        assert (df["predict"] >= 0).all(), f"Negative predictions: {df[df['predict'] < 0]}"

    def test_run_py_output_matches_input_users(self, tmp_path: Path) -> None:
        """Every user_id from test_data must appear exactly once in output."""
        test_csv, user_ids = _make_tiny_train_and_test(tmp_path, n_users=20)
        out = tmp_path / "submit.csv"
        result = _run(
            [
                "--test_data_path",
                str(test_csv),
                "--output_path",
                str(out),
                "--train-frac",
                "0.5",
            ],
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

        df = pd.read_csv(out)
        assert set(df["user_id"]) == set(user_ids), (
            f"user_id set mismatch: missing={set(user_ids) - set(df['user_id'])[:5]}, "
            f"extra={set(df['user_id']) - set(user_ids)[:5]}"
        )


class TestDockerArtifacts:
    """R8: Dockerfile and metadata.json must exist."""

    def test_dockerfile_exists(self) -> None:
        dockerfile = REPO_ROOT / "Dockerfile"
        assert dockerfile.exists(), f"{dockerfile} missing - R8 requires Docker image"

    def test_metadata_json_exists_and_schema(self) -> None:
        metadata = REPO_ROOT / "metadata.json"
        assert metadata.exists(), f"{metadata} missing - R8 requires metadata.json"
        import json

        with open(metadata) as f:
            data = json.load(f)
        assert "image" in data, "metadata.json missing 'image' field"
        assert "entry_point" in data, "metadata.json missing 'entry_point' field"
        assert data["entry_point"] == "python -u run.py", (
            f"entry_point must be 'python -u run.py', got {data['entry_point']!r}"
        )
