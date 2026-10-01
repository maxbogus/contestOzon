#!/usr/bin/env python3
"""Tests for cross-repo stage gating in check_plan.py (T1->T2->T3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from check_plan import unfinished_in_prefix


class TestUnfinishedInPrefix:
    """unfinished_in_prefix() returns backlog/in_progress tickets in a prefix."""

    def _fake_ticket(self, tid, status):
        return {"id": tid, "status": status, "priority": "P0",
                "repo": "test", "rice": {"score": "5.0"}, "path": Path("x")}

    def _patch_get_all(self, tickets):
        import check_plan
        check_plan.get_all = lambda: tickets

    def test_returns_t1_backlog(self, tmp_path, monkeypatch):
        """T1 backlog → returned."""
        t = self._fake_ticket("T1.0.9", "backlog")
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 1

    def test_excludes_done(self, tmp_path, monkeypatch):
        """T1 done → NOT returned."""
        t = self._fake_ticket("T1.0.9", "done")
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 0

    def test_excludes_in_progress_from_other_prefix(self, tmp_path, monkeypatch):
        """T2 in_progress not returned for T1 prefix."""
        t = self._fake_ticket("T2.6.6", "in_progress")
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 0

    def test_t1_done_t2_backlog(self, tmp_path, monkeypatch):
        """T1 done, T2 backlog → T2 returned."""
        t1 = self._fake_ticket("T1.0.9", "done")
        t2 = self._fake_ticket("T2.6.6", "backlog")
        self._patch_get_all([t1, t2])
        assert len(unfinished_in_prefix("T1")) == 0
        assert len(unfinished_in_prefix("T2")) == 1

    def test_multiple_stages(self, tmp_path, monkeypatch):
        """Many tickets: only backlog/in_progress count."""
        tickets = [
            self._fake_ticket("T1.0.9", "backlog"),
            self._fake_ticket("T1.1.1", "done"),
            self._fake_ticket("T2.6.6", "in_progress"),
            self._fake_ticket("T3.6.13", "backlog"),
        ]
        self._patch_get_all(tickets)
        assert len(unfinished_in_prefix("T1")) == 1
        assert len(unfinished_in_prefix("T2")) == 1
        assert len(unfinished_in_prefix("T3")) == 1


class TestTerminalStatuses:
    """TERMINAL_STATUSES ({done, duplicate, blocked, deferred, wont_fix})
    are filtered out of unfinished_in_prefix and treated as not-startable.
    """

    def _fake_ticket(self, tid, status):
        return {"id": tid, "status": status, "priority": "P0",
                "repo": "test", "rice": {"score": "5.0"}, "path": Path("x")}

    def _patch_get_all(self, tickets):
        import check_plan
        check_plan.get_all = lambda: tickets

    @pytest.mark.parametrize("status", ["duplicate", "blocked", "deferred", "wont_fix"])
    def test_duplicate_blocked_deferred_excluded_from_unfinished(self, status, tmp_path, monkeypatch):
        """Terminal-but-not-done statuses must be excluded from unfinished_in_prefix."""
        t = self._fake_ticket("T1.0.9", status)
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 0

    @pytest.mark.parametrize("status", ["duplicate", "blocked", "deferred", "wont_fix"])
    def test_can_start_returns_false_for_terminal(self, status, tmp_path, monkeypatch):
        from check_plan import can_start
        t = self._fake_ticket("T1.0.9", status)
        self._patch_get_all([t])
        ok, reasons = can_start("T1.0.9")
        assert not ok
        assert any("terminal" in r for r in reasons), reasons

    def test_backlog_still_in_unfinished(self, tmp_path, monkeypatch):
        t = self._fake_ticket("T1.0.9", "backlog")
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 1

    def test_in_progress_still_in_unfinished(self, tmp_path, monkeypatch):
        t = self._fake_ticket("T1.0.9", "in_progress")
        self._patch_get_all([t])
        assert len(unfinished_in_prefix("T1")) == 1
