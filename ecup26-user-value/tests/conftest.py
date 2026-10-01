"""Test-wide fixtures.

Disables the walk-forward RAM guard so training is deterministic in tests
regardless of ambient memory pressure (the guard would otherwise skip origins
non-deterministically and change predictions).
"""

from __future__ import annotations

import pytest

from ecup26_user_value import run_pipeline


@pytest.fixture(autouse=True)
def _no_memory_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_pipeline, "_available_mem_gb", lambda: float("inf"))
