"""RED tests for T3.8.1 benchmarker (scripts/benchmark_t3.py).

Per .clinerules/02-ecup-rules.md:
- R6: reproducible (seeded)
- R8: dev-only tool
- R1: lightgbm Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ecup26_user_value import hybrid  # noqa: E402
from scripts import benchmark_t3 as bench  # noqa: E402


def _make_synthetic(n_users: int = 40, seed: int = 1, n_days: int = 120) -> pd.DataFrame:
    """Synthetic tx with the minimal pipeline schema (user_id, event_date, price)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-09-01", periods=n_days, freq="D")
    rows = []
    for uid in range(n_users):
        for _ in range(int(rng.integers(3, 8))):
            rows.append(
                {
                    "user_id": uid,
                    "event_date": dates[int(rng.integers(0, n_days))],
                    "type": "search",
                    "price": float(rng.uniform(10, 100)),
                }
            )
    return pd.DataFrame(rows)


class TestSearchSpace:
    """Search space completeness + determinism."""

    def test_search_space_has_required_params(self) -> None:
        for key in (
            "n_estimators",
            "num_leaves",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "min_child_samples",
        ):
            assert key in bench.SEARCH_SPACE, f"missing {key}"

    def test_curated_grid_includes_base(self) -> None:
        assert bench.BASE_CONFIG in bench.curated_grid()

    def test_curated_grid_has_coordinate_variations(self) -> None:
        grid = bench.curated_grid()
        assert len(grid) >= 10
        base = bench.BASE_CONFIG
        single = [c for c in grid if sum(1 for k in base if c[k] != base[k]) == 1]
        assert len(single) >= 8

    def test_random_configs_seeded_deterministic(self) -> None:
        assert bench.random_configs(5, seed=42) == bench.random_configs(5, seed=42)
        assert bench.random_configs(5, seed=42) != bench.random_configs(5, seed=7)


class TestUtils:
    """RMSLE + params helpers."""

    def test_rmsle_perfect_is_zero(self) -> None:
        y = np.array([0.0, 100.0, 5.0])
        assert bench.rmsle(y, y) == 0.0

    def test_rmsle_positive_error(self) -> None:
        assert bench.rmsle(np.array([0.0]), np.array([10.0])) > 0.0

    def test_make_params(self) -> None:
        p = bench.make_params(bench.BASE_CONFIG, seed=7)
        assert p["objective"] == "regression"
        assert p["num_leaves"] == 31
        assert p["random_state"] == 7

    def test_format_results_table(self) -> None:
        r = bench.ConfigResult(
            config=bench.BASE_CONFIG,
            rmsle_mean=1.5,
            rmsle_std=0.1,
            rmsle_folds=[1.4, 1.6],
            pred_mean=50.0,
            pred_median=10.0,
            fit_time_sec=5.0,
        )
        table = bench.format_results_table([r])
        assert "RMSLE" in table
        assert "1.5000" in table


class TestEndToEnd:
    """Synthetic fold + evaluation: deterministic, finite, sorted."""

    def test_eval_two_configs_deterministic_and_finite(self, tmp_path: Path) -> None:
        df = _make_synthetic(seed=3)
        max_date = df["event_date"].max()
        cutoff = max_date - pd.Timedelta(days=bench.TARGET_DAYS)
        bench._build_fold(df, cutoff, "A", tmp_path)

        cfg1 = {
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "min_child_samples": 5,
        }
        cfg2 = {
            "n_estimators": 30,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "min_child_samples": 5,
        }
        r1a, pm1a, _, _ = bench._eval_fold(cfg1, "A", bench.SEED, tmp_path)
        r1b, pm1b, _, _ = bench._eval_fold(cfg1, "A", bench.SEED, tmp_path)
        r2, pm2, _, _ = bench._eval_fold(cfg2, "A", bench.SEED, tmp_path)
        # deterministic: same config twice -> same RMSLE
        assert r1a == r1b
        # finite and non-negative
        for v in (r1a, r2, pm1a, pm1b, pm2):
            assert np.isfinite(v)
            assert v >= 0


class TestHybrid:
    """T3.8.10: OOF stacking is deterministic and leave-origin-out (no leakage)."""

    def _fold(self, tmp_path: Path) -> None:
        df = _make_synthetic(seed=5, n_users=60, n_days=200)
        max_date = df["event_date"].max()
        cutoff = max_date - pd.Timedelta(days=bench.TARGET_DAYS)
        bench._build_fold(df, cutoff, "A", tmp_path)

    def test_run_hybrid_deterministic_and_finite(self, tmp_path: Path) -> None:
        self._fold(tmp_path)
        cfg = {
            "n_estimators": 30,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "min_child_samples": 5,
        }
        r1, folds1 = bench.run_hybrid(cfg, ["A"], tmp_path, bench.SEED)
        r2, folds2 = bench.run_hybrid(cfg, ["A"], tmp_path, bench.SEED)
        assert r1 == r2
        assert folds1 == folds2
        assert np.isfinite(r1)
        assert r1 >= 0

    def test_clf_fast_rows_is_mask(self) -> None:
        """Regression: _clf_fast_rows must return a len-N boolean MASK.

        The first hybrid run crashed with a broadcast error because it returned
        row INDICES (len < N) while _oof_purchase_prob combined it with a
        boolean mask of length N.
        """
        y = np.array([0] * 1000 + [1] * 20)
        old_cap = hybrid.HYBRID_NEG_CAP
        hybrid.HYBRID_NEG_CAP = 10
        try:
            mask = hybrid.fast_rows_mask(y, seed=1)
        finally:
            hybrid.HYBRID_NEG_CAP = old_cap
        assert mask.dtype == bool
        assert mask.shape == (len(y),)
        assert mask[y == 1].all()  # all positives kept
        assert mask.sum() == 20 + 10  # positives + capped negatives
        assert mask.sum() < len(y)  # strictly a subset (mask contract)

    def test_oof_works_with_capped_negatives(self, tmp_path: Path) -> None:
        """Capped negatives + leave-origin-out must not crash (broadcast bug)."""
        self._fold(tmp_path)
        X, y, origin, _, _, _ = bench._load_fold_with_origin("A", tmp_path)
        Xf = X.fillna(0.0)
        oids = origin.astype("datetime64[ns]").values
        y_bin = (np.asarray(y) > 0).astype(np.int32)
        old_cap = hybrid.HYBRID_NEG_CAP
        hybrid.HYBRID_NEG_CAP = 20
        try:
            oof = hybrid.oof_purchase_prob(Xf.values, y_bin, oids, seed=1)
        finally:
            hybrid.HYBRID_NEG_CAP = old_cap
        assert oof.shape == (len(y_bin),)
        assert np.isfinite(oof).all()

    def test_oof_is_leave_origin_out(self, tmp_path: Path) -> None:
        """Inject all-positive labels into one origin: its OOF must stay far from 1.

        If the classifier saw that origin's (injected) labels, the OOF would
        memorize ~1.0. Leave-origin-out cannot -> mean must be well below 1.
        """
        self._fold(tmp_path)
        X, y, origin, _, _, _ = bench._load_fold_with_origin("A", tmp_path)
        Xf = X.fillna(0.0)
        oids = origin.astype("datetime64[ns]").values
        y_bin = (np.asarray(y) > 0).astype(np.int32)
        target_o = np.unique(oids)[0]
        y_injected = y_bin.copy()
        y_injected[oids == target_o] = 1
        oof = hybrid.oof_purchase_prob(Xf.values, y_injected, oids, seed=1)
        oof_injected = oof[oids == target_o]
        assert len(oof_injected) > 0
        # memorization would push mean ~1.0; leak-free -> base rate, far below 1
        assert float(oof_injected.mean()) < 0.9
