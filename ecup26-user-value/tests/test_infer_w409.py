"""RED tests for T3.19 infer_w409 (auto-discovery / ensemble / hybrid math).

Per .clinerules: R1 (PyTorch BSD), R6 (determinism), R10 (unique IDs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from infer_w409 import ensemble_logs, find_best_model, hybrid_log  # noqa: E402


def _write_json(tmp_path: Path, name: str, cfg: dict) -> Path:
    jp = tmp_path / f"{name}.json"
    jp.write_text(json.dumps(cfg))
    return jp


class TestFindBestModel:
    def test_picks_min_rmsle_for_kind_and_window(self, tmp_path) -> None:
        _write_json(
            tmp_path, "gru_a", {"model": "gru", "window": 409, "lr": 0.0005, "best_rmsle": 1.6800}
        )
        _write_json(
            tmp_path, "gru_b", {"model": "gru", "window": 409, "lr": 0.0005, "best_rmsle": 1.6702}
        )
        _write_json(
            tmp_path, "gru_c", {"model": "gru", "window": 409, "lr": 0.0003, "best_rmsle": 1.6745}
        )
        _write_json(
            tmp_path, "lstm", {"model": "lstm", "window": 409, "lr": 0.0005, "best_rmsle": 1.6761}
        )
        name, r = find_best_model("gru", 409, tmp_path)  # type: ignore[arg-type]
        assert name == "gru_b" and r == 1.6702

    def test_window_filter(self, tmp_path) -> None:
        _write_json(
            tmp_path, "gru_360", {"model": "gru", "window": 360, "lr": 0.0005, "best_rmsle": 1.6700}
        )
        assert find_best_model("gru", 409, tmp_path) is None  # type: ignore[arg-type]

    def test_lr_filter(self, tmp_path) -> None:
        _write_json(
            tmp_path, "gru_05", {"model": "gru", "window": 409, "lr": 0.0005, "best_rmsle": 1.6800}
        )
        _write_json(
            tmp_path, "gru_08", {"model": "gru", "window": 409, "lr": 0.0008, "best_rmsle": 1.6700}
        )
        name, _ = find_best_model("gru", 409, tmp_path, lr=0.0005)  # type: ignore[arg-type]
        assert name == "gru_05"

    def test_skips_static_models(self, tmp_path) -> None:
        _write_json(
            tmp_path,
            "gru_st86",
            {"model": "gru", "window": 409, "lr": 0.0008, "best_rmsle": 1.6500, "static_dim": 86},
        )
        assert find_best_model("gru", 409, tmp_path) is None  # type: ignore[arg-type]

    def test_skips_missing_rmsle(self, tmp_path) -> None:
        _write_json(tmp_path, "gru_nores", {"model": "gru", "window": 409})
        assert find_best_model("gru", 409, tmp_path) is None  # type: ignore[arg-type]


class TestBlendMath:
    def test_ensemble_logs_mean(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 4.0, 5.0])
        out = ensemble_logs([a, b])
        np.testing.assert_allclose(out, [2.0, 3.0, 4.0])

    def test_hybrid_log_weight(self) -> None:
        n = np.array([2.0])
        t = np.array([0.0])
        out = hybrid_log(n, t, 0.25)
        np.testing.assert_allclose(out, 0.5)

    def test_hybrid_log_pure_tabular_at_w0(self) -> None:
        n = np.array([9.0])
        t = np.array([4.0])
        np.testing.assert_allclose(hybrid_log(n, t, 0.0), 4.0)

    def test_hybrid_log_pure_neural_at_w1(self) -> None:
        n = np.array([9.0])
        t = np.array([4.0])
        np.testing.assert_allclose(hybrid_log(n, t, 1.0), 9.0)
