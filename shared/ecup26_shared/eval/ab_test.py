"""A/B test framework with statistical significance (paired t-test).

Adapted from mlaw-rag backend/src/application/ab_test.py.
Pure Python t-test via incomplete beta function (no scipy dependency).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

LIFT_THRESHOLD = 0.05
LATENCY_BUDGET_MS = 200.0


@dataclass(frozen=True, slots=True)
class ABLiftReport:
    """One A/B test result with statistical significance."""
    mean_lift: float
    p_value: float
    is_significant: bool
    reranker_latency_overhead_ms: float
    n_queries: int
    baseline_ndcg: float
    with_rerank_ndcg: float

    @property
    def is_worthwhile(self) -> bool:
        return (
            self.is_significant
            and self.mean_lift > LIFT_THRESHOLD
            and self.reranker_latency_overhead_ms < LATENCY_BUDGET_MS
        )

    def to_markdown(self) -> str:
        verdict = "WORTHWHILE" if self.is_worthwhile else "not worthwhile"
        return (
            f"## A/B test\n\n"
            f"| metric | value |\n|---|---|\n"
            f"| baseline | {self.baseline_ndcg:.4f} |\n"
            f"| treatment | {self.with_rerank_ndcg:.4f} |\n"
            f"| mean lift | {self.mean_lift:+.4f} |\n"
            f"| p-value | {self.p_value:.4f} |\n"
            f"| significant (p<0.05) | {self.is_significant} |\n"
            f"| latency overhead (ms) | {self.reranker_latency_overhead_ms:.1f} |\n\n"
            f"Verdict: **{verdict}**"
        )


def _gammaln(x: float) -> float:
    """Log gamma via Lanczos approximation (Numerical Recipes style)."""
    coeffs = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    if x < 0.5:
        return math.log(math.pi) - math.log(math.sin(math.pi * x)) - _gammaln(1.0 - x)
    z = x - 1.0
    a = coeffs[0]
    t = z + 7.5
    for i in range(1, 9):
        a += coeffs[i] / (z + i)
    return 0.5 * math.log(2.0 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(a)


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnbt = _gammaln(a + b) - _gammaln(a) - _gammaln(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(lnbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest_pvalue(baseline: Sequence[float], treatment: Sequence[float]) -> float:
    """Two-tailed paired Student's t-test p-value. No scipy dep."""
    n = len(baseline)
    if n < 2 or len(treatment) != n:
        return 1.0
    diffs = [t - b for t, b in zip(treatment, baseline, strict=True)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    t_stat = mean / math.sqrt(var / n)
    df = n - 1
    x = df / (df + t_stat * t_stat)
    p = _betai(df / 2.0, 0.5, x)
    return 1.0 - p  # two-tailed


def compute_lift(
    baseline_ndcgs: Sequence[float],
    treatment_ndcgs: Sequence[float],
    reranker_latency_overhead_ms: float = 0.0,
) -> ABLiftReport:
    """Compute nDCG lift with paired t-test (no scipy)."""
    n = len(baseline_ndcgs)
    if n == 0 or len(treatment_ndcgs) != n:
        raise ValueError("baseline and treatment must be non-empty and equal length")
    mean_lift = sum(t - b for b, t in zip(baseline_ndcgs, treatment_ndcgs, strict=True)) / n
    p_value = paired_ttest_pvalue(baseline_ndcgs, treatment_ndcgs)
    is_significant = p_value < 0.05
    return ABLiftReport(
        mean_lift=mean_lift,
        p_value=p_value,
        is_significant=is_significant,
        reranker_latency_overhead_ms=reranker_latency_overhead_ms,
        n_queries=n,
        baseline_ndcg=sum(baseline_ndcgs) / n,
        with_rerank_ndcg=sum(treatment_ndcgs) / n,
    )
