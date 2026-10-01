"""T1.4.8 / M-006 — Submission pipeline orchestrator."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestSubmissionPipeline:
    """Pipeline orchestrator with injectable stages (mlaw-rag index_builder pattern)."""

    def test_imports(self):
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline
        assert SubmissionPipeline is not None

    def test_default_state(self):
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline
        p = SubmissionPipeline()
        assert p.stages == []
        assert p.stats is None

    def test_add_stage(self):
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline, Stage
        p = SubmissionPipeline()
        p.add_stage(Stage(name="read", callable=lambda: ["a", "b"]))
        assert len(p.stages) == 1
        assert p.stages[0].name == "read"

    def test_run_sequential(self):
        """Pipeline: read -> transform -> write."""
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline, Stage

        results = {}

        def read_fn():
            return [1, 2, 3]

        def transform_fn(items):
            return [x * 10 for x in items]

        def write_fn(items):
            results["written"] = items
            return len(items)

        p = SubmissionPipeline()
        p.add_stage(Stage(name="read", callable=read_fn))
        p.add_stage(Stage(name="transform", callable=transform_fn))
        p.add_stage(Stage(name="write", callable=write_fn))

        stats = p.run()
        assert results["written"] == [10, 20, 30]
        assert stats.n_records_read == 3
        assert stats.n_records_written == 3
        assert stats.duration_ms > 0

    def test_stage_failure_continues_with_error(self):
        """A failing stage should mark stats.success=False but not crash."""
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline, Stage

        p = SubmissionPipeline()
        p.add_stage(Stage(name="bad", callable=lambda: 1 / 0))
        stats = p.run()
        assert stats.success is False
        assert stats.error is not None
        assert "ZeroDivisionError" in stats.error

    def test_required_callable(self):
        from ecup26_shared.pipeline.orchestrator import Stage
        with pytest.raises(TypeError):
            Stage(name="x", callable=None)  # None not allowed

    def test_callable_output_required(self):
        """Each stage callable should return something (not None)."""
        from ecup26_shared.pipeline.orchestrator import SubmissionPipeline, Stage

        p = SubmissionPipeline()
        p.add_stage(Stage(name="noop", callable=lambda: None))
        with pytest.raises(ValueError):
            p.validate_or_raise()

    def test_stats_summary(self):
        from ecup26_shared.pipeline.orchestrator import (
            SubmissionPipeline, Stage, PipelineStats,
        )
        p = SubmissionPipeline()
        p.add_stage(Stage(name="ok", callable=lambda: [1, 2]))
        stats = p.run()
        summary = stats.summary()
        assert "n_records_read=2" in summary
        assert "success=True" in summary
