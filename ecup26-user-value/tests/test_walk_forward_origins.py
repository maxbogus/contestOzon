"""Leak-guard tests for ECUP26_EXTRA_ORIGINS (T3.13 / Г3#1).

Adding a historical gift-season origin (2025-02-13, whose label window is the
fold-G gift window) to walk-forward training must be guarded so that fold-G's
own training (ref = cutoff = 2025-02-13) cannot leak its truth window.

Per .clinerules/02-ecup-rules.md W4: tests must verify rules apply.
"""

from __future__ import annotations

import pandas as pd

from ecup26_user_value.run_pipeline import _walk_forward_origins


def test_extra_gift_origin_injected_for_production(monkeypatch) -> None:
    monkeypatch.setenv("ECUP26_EXTRA_ORIGINS", "2025-02-13")
    origins = _walk_forward_origins(pd.Timestamp("2026-02-13"))
    assert pd.Timestamp("2025-02-13") in origins


def test_fold_g_training_cannot_see_own_truth_window(monkeypatch) -> None:
    """Fold-G training ref == 2025-02-13; its label window would BE the
    fold-G truth, so the gift origin must be skipped (no leakage)."""
    monkeypatch.setenv("ECUP26_EXTRA_ORIGINS", "2025-02-13")
    origins = _walk_forward_origins(pd.Timestamp("2025-02-13"))
    assert pd.Timestamp("2025-02-13") not in origins
    # the last origin label window must end at/before the fold-G cutoff
    assert origins[0] == pd.Timestamp("2025-01-14")


def test_extra_origin_full_label_window_required(monkeypatch) -> None:
    """Origin whose label window extends past ref_date is dropped (partial
    label would mix training signal with the validation horizon)."""
    monkeypatch.setenv("ECUP26_EXTRA_ORIGINS", "2026-01-20")
    # 2026-01-20 + 30d = 2026-02-19 > ref 2026-02-13 -> dropped
    origins = _walk_forward_origins(pd.Timestamp("2026-02-13"))
    assert pd.Timestamp("2026-01-20") not in origins


def test_default_behavior_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("ECUP26_EXTRA_ORIGINS", raising=False)
    origins = _walk_forward_origins(pd.Timestamp("2026-02-13"))
    assert len(origins) == 8  # default WALK_FORWARD_ORIGINS, no extras
