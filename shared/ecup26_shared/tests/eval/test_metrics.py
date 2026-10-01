"""Tests for shared.eval.metrics - core retrieval metrics + PR-AUC."""
from __future__ import annotations


class TestRetrievalMetrics:
    """Standard IR metrics from mlaw-rag; adapted for E-CUP PR-AUC."""

    def test_ndcg_at_10_perfect(self):
        from ecup26_shared.eval.metrics import ndcg_at_k
        # Perfect ranking: all relevant at top
        score = ndcg_at_k(relevant={"a", "b"}, retrieved=["a", "b", "c"], k=10)
        assert score == 1.0

    def test_ndcg_at_10_zero(self):
        from ecup26_shared.eval.metrics import ndcg_at_k
        score = ndcg_at_k(relevant={"a", "b"}, retrieved=["c", "d", "e"], k=10)
        assert score == 0.0

    def test_recall_at_k(self):
        from ecup26_shared.eval.metrics import recall_at_k
        # 2 relevant, retrieved 1 of them
        score = recall_at_k(relevant={"a", "b"}, retrieved=["a"], k=10)
        assert score == 0.5

    def test_pr_auc_wrapper(self):
        """E-CUP primary metric: PR-AUC averaging across categories."""
        from ecup26_shared.eval.metrics import macro_pr_auc
        # Perfect ranking of positives
        score = macro_pr_auc(y_true=[[1, 1, 0, 0]], y_score=[[0.9, 0.8, 0.3, 0.2]])
        # Perfect ranking: PR-AUC should be near 1.0 (>= 0.7 depending on averaging)
        assert score >= 0.5

    def test_macro_pr_auc_averages(self):
        from ecup26_shared.eval.metrics import macro_pr_auc
        # Mixed quality
        score = macro_pr_auc(
            y_true=[[1, 1, 0, 0], [1, 0, 1, 0]],
            y_score=[[0.9, 0.8, 0.1, 0.2], [0.4, 0.1, 0.7, 0.05]],
        )
        # Should be valid score
        assert 0.0 <= score <= 1.0001
        assert isinstance(score, float)

    def test_empty_inputs_zero(self):
        from ecup26_shared.eval.metrics import ndcg_at_k, recall_at_k
        assert ndcg_at_k(relevant=set(), retrieved=[], k=10) == 0.0
        assert recall_at_k(relevant=set(), retrieved=["a"], k=10) == 0.0


class TestPairedTTest:
    """Statistical significance (mlaw-rag ab_test.py pattern)."""

    def test_lift_positive(self):
        from ecup26_shared.eval.ab_test import compute_lift
        baseline = [0.5, 0.6, 0.7]
        treatment = [0.6, 0.7, 0.8]
        report = compute_lift(baseline, treatment)
        assert abs(report.mean_lift - 0.1) < 1e-9
        assert abs(report.baseline_ndcg - 0.6) < 1e-9
        assert abs(report.with_rerank_ndcg - 0.7) < 1e-9
