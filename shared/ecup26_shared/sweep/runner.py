"""Combinatorial parameter sweep runner.

Adapted from mlaw-rag backend/src/application/sweep_runner.py.

Usage:
    results = SweepRunner.run(
        {"lr": [0.01, 0.05, 0.1], "depth": [3, 5, 7]},
        lambda params: {"accuracy": train_and_eval(**params)},
    )
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


RunFn = Callable[[dict[str, Any]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One grid combination + its metric scores."""
    params: dict[str, Any]
    metrics: dict[str, float]


class SweepRunner:
    """Combinatorial parameter sweep with optional parallel execution."""

    @staticmethod
    def combinations(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
        """Cartesian product of the grid -> one param dict per combination."""
        keys = list(param_grid)
        values = [param_grid[k] for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    @staticmethod
    def run(param_grid: Mapping[str, Sequence[Any]], run_fn: RunFn) -> list[SweepResult]:
        """Sequential run over all combinations."""
        return [SweepResult(p, run_fn(p)) for p in SweepRunner.combinations(param_grid)]

    @staticmethod
    def run_parallel(
        param_grid: Mapping[str, Sequence[Any]],
        run_fn: RunFn,
        max_workers: int = 4,
    ) -> list[SweepResult]:
        """Parallel run via ThreadPoolExecutor."""
        combos = SweepRunner.combinations(param_grid)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            metrics_list = list(ex.map(run_fn, combos))
        return [SweepResult(p, m) for p, m in zip(combos, metrics_list, strict=True)]
