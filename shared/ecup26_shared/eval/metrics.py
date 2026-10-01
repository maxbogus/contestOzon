"""Retrieval metrics + E-CUP PR-AUC wrapper.

Adapted from mlaw-rag backend/src/application/metrics.py.
Pure functions over sets/lists, no I/O, no ML deps.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _top(ids: Sequence[str], k: int) -> list[str]:
    return list(ids[:k])


def ndcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 10) -> float:
    """nDCG@k: DCG of retrieved / DCG of ideal ranking.

    Adapted from mlaw-rag metrics.ndcg_at_k.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(_top(retrieved, k)) if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(j + 2) for j in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 10) -> float:
    """Recall@k: fraction of relevant docs found in top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in _top(retrieved, k) if doc_id in relevant)
    return hits / len(relevant)


def macro_pr_auc(
    y_true: Sequence[Sequence[int]],
    y_score: Sequence[Sequence[float]],
) -> float:
    """Macro-averaged PR-AUC for E-CUP submission metric.

    E-CUP computes PR-AUC per category then averages: see task1_matching.md.
    Uses sklearn's average_precision_score if available; pure-python fallback otherwise.

    Args:
        y_true: list of per-category binary labels [0, 1, 0, 1, ...].
        y_score: same shape as y_true, predicted probabilities.

    Returns:
        Macro-averaged PR-AUC in [0, 1].
    """
    if not y_true or len(y_true) != len(y_score):
        return 0.0

    # Try sklearn first (better numerical stability)
    try:
        from sklearn.metrics import average_precision_score  # type: ignore
        scores = []
        for yt, ys in zip(y_true, y_score, strict=True):
            if any(yt):
                scores.append(average_precision_score(yt, ys))
            # skip empty-positive categories
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    except ImportError:
        # Pure-python fallback: trapezoidal PR curve integration
        scores = []
        for yt, ys in zip(y_true, y_score, strict=True):
            if not any(yt):
                continue
            scores.append(_pr_auc_pure(list(yt), list(ys)))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


def _pr_auc_pure(y_true: list[int], y_score: list[float]) -> float:
    """Pure-python PR-AUC: integrate P(R) curve using trapezoidal rule."""
    # Sort by descending score
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    y_sorted = [y_true[i] for i in order]

    tp = 0
    fp = 0
    precisions = []
    recalls = []
    total_pos = sum(y_sorted)

    for yt in y_sorted:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
    return auc
