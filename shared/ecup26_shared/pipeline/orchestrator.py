"""Submission pipeline orchestrator.

Pattern from mlaw-rag backend/src/application/index_builder.py (T-057):
- Injectable stages for testability
- Stats dataclass with summary()
- Failures captured, not crashed (run returns stats; validate_or_raise raises)

Usage:
    pipe = SubmissionPipeline()
    pipe.add_stage(Stage("read", read_parquet))
    pipe.add_stage(Stage("encode", encode_items))
    pipe.add_stage(Stage("write", write_csv))
    stats = pipe.run()
    pipe.validate_or_raise()  # raises ValueError on failure
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Stage:
    """One stage in the submission pipeline."""
    name: str
    callable: Callable[..., Any]

    def __post_init__(self) -> None:
        if self.callable is None or not callable(self.callable):
            raise TypeError(
                f"Stage '{self.name}': callable must be a callable, "
                f"got {self.callable!r}"
            )


@dataclass(slots=True)
class PipelineStats:
    """Measured counts of one pipeline run."""
    n_records_read: int = 0
    n_records_written: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    stage_durations_ms: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"n_records_read={self.n_records_read}",
            f"n_records_written={self.n_records_written}",
            f"duration_ms={self.duration_ms:.1f}",
            f"success={self.success}",
        ]
        if self.error:
            parts.append(f"error={self.error}")
        return " ".join(parts)


class SubmissionPipeline:
    """Composed submission pipeline: stages run sequentially.

    Each stage receives the previous stage's output (or initial None).
    First stage typically returns the dataset; later stages transform it.
    """
    def __init__(self) -> None:
        self.stages: list[Stage] = []
        self.stats: PipelineStats | None = None

    def add_stage(self, stage: Stage) -> "SubmissionPipeline":
        self.stages.append(stage)
        return self

    def run(self) -> PipelineStats:
        """Execute all stages sequentially. Captures errors in stats."""
        stats = PipelineStats()
        self.stats = stats
        t_total = time.perf_counter()
        data: Any = None

        for i, stage in enumerate(self.stages):
            t0 = time.perf_counter()
            try:
                if i == 0:
                    data = stage.callable()
                else:
                    data = stage.callable(data)
            except Exception as e:
                stats.success = False
                stats.error = f"{type(e).__name__}: {e}"
                stats.duration_ms = (time.perf_counter() - t_total) * 1000
                return stats

            stage_ms = (time.perf_counter() - t0) * 1000
            stats.stage_durations_ms[stage.name] = stage_ms

            if data is None:
                stats.success = False
                stats.error = (
                    f"Stage '{stage.name}' returned None; "
                    "expected non-None output"
                )
                stats.duration_ms = (time.perf_counter() - t_total) * 1000
                return stats

        if isinstance(data, list):
            stats.n_records_read = len(data)
            stats.n_records_written = len(data)
        elif isinstance(data, int):
            stats.n_records_read = data
            stats.n_records_written = data

        stats.duration_ms = (time.perf_counter() - t_total) * 1000
        return stats

    def validate_or_raise(self) -> PipelineStats:
        """Run pipeline and raise ValueError on failure.

        Returns stats on success. Useful for tests and CLI scripts that
        should fail loudly on errors.
        """
        stats = self.run()
        if not stats.success:
            raise ValueError(f"Pipeline failed: {stats.error}")
        return stats
