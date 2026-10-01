"""M-007 — Sweep infrastructure (runner + aggregator)."""
from __future__ import annotations

import pytest


class TestSweepRunner:
    """Combinatorial parameter sweep over a grid."""

    def test_imports(self):
        from ecup26_shared.sweep.runner import SweepRunner
        assert SweepRunner is not None

    def test_combinations_cartesian_product(self):
        from ecup26_shared.sweep.runner import SweepRunner
        combos = SweepRunner.combinations({
            "lr": [0.01, 0.1],
            "depth": [3, 5],
        })
        assert len(combos) == 4  # 2 * 2
        # Each combo has all keys
        for c in combos:
            assert set(c.keys()) == {"lr", "depth"}

    def test_single_param_grid(self):
        from ecup26_shared.sweep.runner import SweepRunner
        combos = SweepRunner.combinations({"alpha": [0.1, 0.5, 1.0]})
        assert len(combos) == 3

    def test_empty_grid(self):
        from ecup26_shared.sweep.runner import SweepRunner
        combos = SweepRunner.combinations({})
        # Single empty combo {} is OK for sweep; at least one result
        assert combos == [{}]

    def test_run_executes_for_each_combo(self):
        from ecup26_shared.sweep.runner import SweepRunner
        calls = []

        def fake_eval(params):
            calls.append(params)
            return {"score": params["x"] ** 2}

        results = SweepRunner.run(
            {"x": [1, 2, 3]},
            fake_eval,
        )
        assert len(results) == 3
        assert len(calls) == 3
        # All x values covered
        assert {r.params["x"] for r in results} == {1, 2, 3}
        # Best score = 9 (x=3)
        best = max(results, key=lambda r: r.metrics["score"])
        assert best.metrics["score"] == 9

    def test_run_parallel(self):
        from ecup26_shared.sweep.runner import SweepRunner

        def fake_eval(params):
            return {"score": params["k"] * 2}

        results = SweepRunner.run_parallel(
            {"k": [1, 2, 3]},
            fake_eval,
            max_workers=2,
        )
        assert len(results) == 3
        scores = sorted([r.metrics["score"] for r in results])
        assert scores == [2, 4, 6]


class TestParetoAggregator:
    """Pareto frontier: best tradeoffs between objectives."""

    def test_imports(self):
        from ecup26_shared.sweep.aggregator import ParetoAggregator
        assert ParetoAggregator is not None

    def test_pareto_frontier_simple(self):
        from ecup26_shared.sweep.aggregator import ParetoAggregator
        from ecup26_shared.sweep.runner import SweepResult

        # Two metrics: maximize "quality", minimize "latency"
        results = [
            SweepResult(params={"x": 1}, metrics={"quality": 0.5, "latency": 100}),
            SweepResult(params={"x": 2}, metrics={"quality": 0.7, "latency": 50}),  # dominates
            SweepResult(params={"x": 3}, metrics={"quality": 0.8, "latency": 200}),
        ]
        frontier = ParetoAggregator.pareto_frontier(
            results, maximize={"quality"}, minimize={"latency"},
        )
        # Pareto-optimal: x=2 (quality=0.7, latency=50) dominates x=1
        # x=3 has better quality but worse latency — on frontier
        ids = {r.params["x"] for r in frontier}
        assert 2 in ids
        assert 1 not in ids  # dominated by x=2

    def test_pareto_with_min_max_objectives(self):
        from ecup26_shared.sweep.aggregator import ParetoAggregator
        from ecup26_shared.sweep.runner import SweepResult

        results = [
            SweepResult(params={"i": 1}, metrics={"a": 1.0, "b": 1.0}),
            SweepResult(params={"i": 2}, metrics={"a": 2.0, "b": 1.0}),  # dominates i=1
            SweepResult(params={"i": 3}, metrics={"a": 1.5, "b": 0.5}),  # dominates i=1
        ]
        frontier = ParetoAggregator.pareto_frontier(
            results, maximize={"a", "b"},
        )
        # i=2 dominates i=1 (a is bigger, b same)
        # i=3 dominates i=1 (a bigger, b bigger)
        # i=2 vs i=3: a bigger in 2, b bigger in 3 — both on frontier
        ids = {r.params["i"] for r in frontier}
        assert 1 not in ids

    def test_csv_export(self):
        from ecup26_shared.sweep.aggregator import ParetoAggregator
        from ecup26_shared.sweep.runner import SweepResult
        import io

        results = [
            SweepResult(params={"k": 1}, metrics={"score": 0.5}),
            SweepResult(params={"k": 2}, metrics={"score": 0.7}),
        ]
        csv_str = ParetoAggregator.to_csv(results)
        assert "k,score" in csv_str  # header
        assert "1,0.5" in csv_str
        assert "2,0.7" in csv_str
