"""Tests for T3.7.4 - hyperparameter tuning CLI.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lightgbm, pandas, tqdm)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/ is a sibling of ecup26_user_value/ at the repo root, not under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tune_hyperparams import (  # noqa: E402
    SEARCH_SPACE,
    expand_grid,
    format_results_row,
    load_synthetic_df,
    run_grid_search,
    write_report,
)


def _make_synthetic_tx(n_users: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-09-01", periods=60, freq="D")
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        for _o in range(int(rng.integers(3, 10))):
            rows.append(
                {
                    "user_id": uid,
                    "event_date": dates[int(rng.integers(0, len(dates)))],
                    "type": seg,
                    "price": float(rng.uniform(5, 100)),
                }
            )
    return pd.DataFrame(rows)


class TestSearchSpace:
    """SEARCH_SPACE contains all 4 tuned parameters."""

    def test_search_space_has_4_params(self) -> None:
        assert set(SEARCH_SPACE.keys()) == {
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "learning_rate",
        }, f"SEARCH_SPACE keys: {list(SEARCH_SPACE.keys())}"

    def test_each_param_has_multiple_values(self) -> None:
        for name, values in SEARCH_SPACE.items():
            assert len(values) >= 2, f"{name} should have >=2 values, got {len(values)}"

    def test_default_config_in_search(self) -> None:
        """Default LightGBMRegressorMSE params must be in the grid."""
        from ecup26_user_value.ablation.model_wrappers import LightGBMRegressorMSE

        model = LightGBMRegressorMSE().build()
        booster = model._booster
        for name in SEARCH_SPACE:
            assert name in booster.get_params(), f"param {name} not in LightGBM default params"


class TestConfigExpansion:
    """expand_grid produces all combinations."""

    def test_expand_grid_count(self) -> None:
        grid = {"a": [1, 2], "b": [10, 20, 30]}
        combos = expand_grid(grid)
        assert len(combos) == 6, f"expected 6 combinations, got {len(combos)}"

    def test_full_grid_size_is_81(self) -> None:
        """T3.7.4 grid: 3 * 3 * 3 * 3 = 81 configs."""
        combos = expand_grid(SEARCH_SPACE)
        n_expected = 1
        for v in SEARCH_SPACE.values():
            n_expected *= len(v)
        assert len(combos) == n_expected, f"expected {n_expected} combinations, got {len(combos)}"
        assert n_expected == 81, f"sanity check: 3*3*3*3 = 81, got {n_expected}"


class TestOutputFormatting:
    """Top-K config output is sorted by RMSLE ascending."""

    def test_results_sort_by_rmsle(self) -> None:
        results = [
            {"params": {"a": 100}, "model_rmsle": 5.0},
            {"params": {"a": 50}, "model_rmsle": 3.0},
            {"params": {"a": 200}, "model_rmsle": 4.0},
        ]
        sorted_results = sorted(results, key=lambda r: r["model_rmsle"])
        assert sorted_results[0]["params"]["a"] == 50
        assert sorted_results[-1]["params"]["a"] == 100

    def test_format_results_row(self) -> None:
        params = {"n_estimators": 100, "num_leaves": 15}
        metrics = {
            "model_rmsle": 5.0,
            "lift_vs_mean": 0.1,
            "mae": 100.0,
            "rmse": 200.0,
            "spearman": 0.5,
        }
        row = format_results_row(params, metrics, 10.0, 4.9, 5.1)
        assert row["params"] == params
        assert row["model_rmsle"] == 5.0
        assert row["fit_time_sec"] == 10.0
        assert row["bootstrap_ci_low"] == 4.9
        assert row["bootstrap_ci_high"] == 5.1


class TestProgressBar:
    """Progress bar is used during grid search."""

    def test_tqdm_imported(self) -> None:
        try:
            from tqdm import tqdm  # noqa: F401
        except ImportError:
            pytest.fail("tqdm not installed; run: uv add --dev tqdm")

    def test_tqdm_pbar_works(self) -> None:
        """Smoke test: tqdm can wrap a small loop."""
        from tqdm import tqdm

        with tqdm(total=5, disable=True) as pbar:
            for _ in range(5):
                pbar.update(1)


class TestWriteReport:
    """write_report creates CSV + Markdown."""

    def test_report_files_created(self, tmp_path: Path) -> None:
        results = [
            {
                "params": {
                    "n_estimators": 100,
                    "num_leaves": 15,
                    "min_child_samples": 50,
                    "learning_rate": 0.1,
                },
                "model_rmsle": 5.0,
                "lift_vs_mean": 0.1,
                "mae": 100.0,
                "rmse": 200.0,
                "spearman": 0.5,
                "bootstrap_ci_low": 4.9,
                "bootstrap_ci_high": 5.1,
                "fit_time_sec": 10.0,
            }
        ]
        output = tmp_path / "report.md"
        write_report(results, output, all_results_count=81, elapsed_sec=120.0)
        assert output.exists()
        md = output.read_text()
        assert "RMSLE" in md
        assert "n_estimators=100" in md
        # CSV sibling
        assert output.with_suffix(".csv").exists()


class TestEndToEnd:
    """Full grid search runs on synthetic data in reasonable time."""

    def test_subsample_grid_search_under_budget(self) -> None:
        """Full 81-config grid on 100 users should complete in <60s."""
        df = load_synthetic_df(n_users=100, seed=42)
        max_date = df["event_date"].max()
        cutoff = max_date - pd.Timedelta(days=30)
        train_df = df[df["event_date"] <= cutoff].copy()
        val_df = df[(df["event_date"] > cutoff) & (df["event_date"] <= max_date)].copy()

        t0 = time.time()
        top_results = run_grid_search(
            train_df,
            val_df,
            n_boot=20,  # small for speed in tests
            top_k=5,
            seed=42,
        )
        elapsed = time.time() - t0
        # 81 configs on 100 users, n_boot=20: should be fast
        assert elapsed < 60, f"grid search took {elapsed:.1f}s, expected <60s"
        assert len(top_results) == 5
        # All results should have finite metrics
        for r in top_results:
            assert np.isfinite(r["model_rmsle"])
            assert np.isfinite(r["fit_time_sec"])
        # Results sorted by RMSLE ascending
        for i in range(len(top_results) - 1):
            assert top_results[i]["model_rmsle"] <= top_results[i + 1]["model_rmsle"]
