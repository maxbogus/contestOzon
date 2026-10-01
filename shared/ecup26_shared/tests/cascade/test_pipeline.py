"""Tests for cascade pipeline pattern."""
from __future__ import annotations


class TestCascade:
    def test_two_stage_pipeline(self):
        """Cascade: source_a (k1) -> intermediate (k2) -> final (k3)."""
        from ecup26_shared.cascade.pipeline import Cascade, CascadeStage

        stage1 = CascadeStage(name="dense_search", k_out=100)
        stage2 = CascadeStage(name="rerank", k_out=20)
        c = Cascade(stages=[stage1, stage2], final_k=5)

        assert len(c.stages) == 2
        assert c.final_k == 5

    def test_cascade_stage_params(self):
        from ecup26_shared.cascade.pipeline import CascadeStage
        s = CascadeStage(name="encode", k_out=10, params={"batch_size": 16})
        assert s.name == "encode"
        assert s.k_out == 10
        assert s.params["batch_size"] == 16
