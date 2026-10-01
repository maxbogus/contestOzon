"""Pareto frontier aggregation for sweep results.

Adapted from mlaw-rag backend/src/application/sweep_aggregator.py.

Pareto dominance:
    a dominates b if:
        - a >= b on every objective (maximize set)
        - a <= b on every objective (minimize set)
        - strictly better on at least one objective

Non-dominated points form the Pareto frontier.
"""
from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import inf
from typing import Any

from ecup26_shared.sweep.runner import SweepResult


def _dominates(
    a: SweepResult,
    b: SweepResult,
    maximize: set[str],
    minimize: set[str],
) -> bool:
    """True if a is >= b on every objective and strictly better on at least one."""
    strictly_better = False
    for m in maximize:
        av = a.metrics.get(m, -inf)
        bv = b.metrics.get(m, -inf)
        if av < bv:
            return False
        if av > bv:
            strictly_better = True
    for m in minimize:
        av = a.metrics.get(m, inf)
        bv = b.metrics.get(m, inf)
        if av > bv:
            return False
        if av < bv:
            strictly_better = True
    return strictly_better


@dataclass(frozen=True, slots=True)
class ParetoAggregator:
    """Static methods for Pareto frontier + CSV export."""

    @staticmethod
    def pareto_frontier(
        results: Sequence[SweepResult],
        maximize: set[str],
        minimize: set[str] | None = None,
    ) -> list[SweepResult]:
        """Non-dominated points (best tradeoffs between objectives)."""
        minimize = set(minimize or set())
        frontier: list[SweepResult] = []
        for r in results:
            dominated = False
            for other in results:
                if _dominates(other, r, maximize, minimize):
                    dominated = True
                    break
            if not dominated:
                frontier.append(r)
        return frontier

    @staticmethod
    def to_csv(results: Iterable[SweepResult]) -> str:
        """Serialize sweep results to CSV string (params + metrics columns)."""
        rows = list(results)
        if not rows:
            return ""
        # Collect all keys
        param_keys: list[str] = []
        metric_keys: list[str] = []
        seen_p = set()
        seen_m = set()
        for r in rows:
            for k in r.params:
                if k not in seen_p:
                    param_keys.append(k)
                    seen_p.add(k)
            for k in r.metrics:
                if k not in seen_m:
                    metric_keys.append(k)
                    seen_m.add(k)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(param_keys + metric_keys)
        for r in rows:
            row = [r.params.get(k, "") for k in param_keys]
            row += [r.metrics.get(k, "") for k in metric_keys]
            writer.writerow(row)
        return buf.getvalue()
