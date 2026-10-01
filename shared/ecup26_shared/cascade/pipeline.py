"""Cascade pipeline pattern: stage1 (k1) -> stage2 (k2) -> ... -> final (k).

Adapted from mlaw-rag backend/src/application/index_builder.py.
The idea: each stage reduces candidate count, trading recall for speed/quality.

E-CUP uses this for:
- T1.4.7: bi-encoder (100) -> LightGBM classifier (20) -> top predictions
- T2: VLM (100) -> filter (20) -> final verdict
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CascadeStage:
    """One stage in the cascade.

    Attributes:
        name: human-readable stage name.
        k_out: number of candidates to pass to the next stage.
        params: arbitrary stage parameters (e.g. batch_size).
    """
    name: str
    k_out: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Cascade:
    """Composed cascade: list of stages + final_k.

    Note: This is the SCHEMA, not the orchestrator. See shared/pipeline/orchestrator.py
    for the runner. We keep schema here so it can be serialized (YAML/JSON).
    """
    stages: Sequence[CascadeStage]
    final_k: int = 5

    def __post_init__(self) -> None:
        if self.final_k < 1:
            raise ValueError(f"final_k must be >= 1, got {self.final_k}")
        for s in self.stages:
            if s.k_out < 1:
                raise ValueError(f"stage {s.name}: k_out must be >= 1, got {s.k_out}")
