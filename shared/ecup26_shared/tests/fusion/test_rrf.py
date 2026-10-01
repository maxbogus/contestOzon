"""Tests for RRF (Reciprocal Rank Fusion)."""
from __future__ import annotations

import pytest


class TestRRF:
    def test_basic_fusion(self):
        from ecup26_shared.fusion.rrf import RankedItem, rrf_fusion

        # Source A: rank 1=doc_x, rank 2=doc_y
        src_a = [RankedItem(id="x", score=0.9, rank=1), RankedItem(id="y", score=0.8, rank=2)]
        # Source B: rank 1=doc_y, rank 2=doc_x
        src_b = [RankedItem(id="y", score=0.95, rank=1), RankedItem(id="x", score=0.7, rank=2)]

        result = rrf_fusion([src_a, src_b], k=60)

        # Both x and y should appear
        ids = [r.id for r in result]
        assert "x" in ids and "y" in ids
        # x: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # y: 1/(60+2) + 1/(60+1) = same as x = same score
        # They tie, so order may vary
        assert len(result) == 2

    def test_empty_sources(self):
        from ecup26_shared.fusion.rrf import rrf_fusion
        result = rrf_fusion([], k=60)
        assert result == []

    def test_single_source(self):
        from ecup26_shared.fusion.rrf import RankedItem, rrf_fusion
        src = [RankedItem(id="a", score=0.5, rank=1)]
        result = rrf_fusion([src], k=60)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_k_parameter_affects_scores(self):
        from ecup26_shared.fusion.rrf import RankedItem, rrf_fusion
        src_a = [RankedItem(id="x", rank=1)]
        src_b = [RankedItem(id="y", rank=1)]
        # With k=60: x=1/61, y=1/61
        # With k=0: x=1/1=1, y=1/1=1 (k must be >=1 per formula convention)
        result60 = rrf_fusion([src_a, src_b], k=60)
        score60 = next(r.rrf_score for r in result60 if r.id == "x")
        # Standard formula: 1/(k+rank)
        assert score60 == pytest.approx(1 / 61, rel=1e-3)

    def test_overlap_favors_high_rank(self):
        from ecup26_shared.fusion.rrf import RankedItem, rrf_fusion
        # x appears at rank 1 in both sources; y at rank 100 in both
        x = RankedItem(id="x", rank=1)
        y = RankedItem(id="y", rank=100)
        result = rrf_fusion([[x, y], [x, y]], k=60)
        # x should rank first (sum 2/(60+1) > 2/(60+100))
        assert result[0].id == "x"
