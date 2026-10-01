"""RED tests for T3.6.13 - GMV distribution EDA.

Per .clinerules/02-ecup-rules.md (R1 open-source, R6 reproducible) and
ticket T3.6.13.

EDA: histogram of GMV values, percentiles, zero count, outlier analysis.
Result should be ~45.9% zeros (per mlaw-rag/T3.6.7 ticket reference).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class TestGMVEDAReport:
    """explore_gmv_distribution must produce structured EDA report."""

    def test_gmv_eda_report_returns_dict(self) -> None:
        from ecup26_user_value.eda.explore_gmv_distribution import (
            compute_gmv_distribution,
        )

        # Synthetic GMV with ~46% zeros
        rng = np.random.default_rng(0)
        gmv = np.concatenate(
            [
                np.zeros(460),
                rng.lognormal(mean=4.0, sigma=2.0, size=540),
            ]
        )
        report = compute_gmv_distribution(pd.Series(gmv))
        assert isinstance(report, dict)
        assert "n_total" in report
        assert "n_zero" in report
        assert "percent_zero" in report
        assert "percentiles" in report
        assert "outlier_count" in report

    def test_zero_count_matches_data(self) -> None:
        """Per ticket: ~45.9% zeros expected."""
        from ecup26_user_value.eda.explore_gmv_distribution import (
            compute_gmv_distribution,
        )

        rng = np.random.default_rng(42)
        gmv = np.concatenate(
            [
                np.zeros(459),
                rng.lognormal(mean=4.0, sigma=2.0, size=541),
            ]
        )
        report = compute_gmv_distribution(pd.Series(gmv))
        # Should detect ~45.9% zeros
        assert 0.40 < report["percent_zero"] < 0.50, (
            f"Expected ~45.9% zeros, got {report['percent_zero']:.2%}"
        )

    def test_percentiles_include_required_points(self) -> None:
        """Per ticket: 50%, 90%, 95%, 99%, 99.9% percentiles."""
        from ecup26_user_value.eda.explore_gmv_distribution import (
            compute_gmv_distribution,
        )

        rng = np.random.default_rng(1)
        gmv = rng.lognormal(mean=4.0, sigma=2.0, size=1000)
        report = compute_gmv_distribution(pd.Series(gmv))
        pcts = report["percentiles"]
        for p in ["50", "90", "95", "99", "99.9"]:
            # Accept either "50" or "50.0" key format
            assert str(p) in pcts or f"{float(p)}" in pcts, (
                f"Missing percentile {p}, got keys {list(pcts.keys())}"
            )

    def test_outlier_analysis(self) -> None:
        """Outliers: values > 99.9th percentile."""
        from ecup26_user_value.eda.explore_gmv_distribution import (
            compute_gmv_distribution,
        )

        rng = np.random.default_rng(2)
        gmv = rng.lognormal(mean=4.0, sigma=2.0, size=1000)
        report = compute_gmv_distribution(pd.Series(gmv))
        # 0.1% outliers = 1 in 1000
        assert report["outlier_count"] >= 1


class TestEDAReporting:
    """EDA script must save report to disk."""

    def test_save_report(self, tmp_path: Path) -> None:
        from ecup26_user_value.eda.explore_gmv_distribution import (
            save_report,
        )

        report = {
            "n_total": 1000,
            "n_zero": 459,
            "percent_zero": 0.459,
            "percentiles": {"50": 1.0, "90": 10.0, "95": 20.0, "99": 50.0, "99.9": 100.0},
            "outlier_count": 1,
        }
        out_path = tmp_path / "eda_report.json"
        save_report(report, out_path)
        assert out_path.exists()
        import json

        loaded = json.loads(out_path.read_text())
        assert loaded["n_total"] == 1000
