#!/usr/bin/env python3
"""Tests for rice.py — RICE prioritization scorer.

TDD: RED first. These tests define the expected API.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

import rice


class TestImpact(unittest.TestCase):
    def test_impact_known_p0(self):
        assert rice.impact("P0") == 3.0

    def test_impact_known_p3(self):
        assert rice.impact("P3") == 0.5

    def test_impact_unknown_default(self):
        assert rice.impact("unknown") == 1.0

    def test_impact_p1(self):
        assert rice.impact("P1") == 2.0

    def test_impact_p2(self):
        assert rice.impact("P2") == 1.0


class TestReach(unittest.TestCase):
    def test_reach_no_dependents(self):
        """Standalone ticket: reach = 1."""
        t = _make(id_="T1.1.1", status="backlog")
        assert rice.reach(t, [t]) == 1

    def test_reach_blocks_dependents(self):
        """Ticket blocks 2 others (one done, one backlog): reach = 1 + 1 = 2."""
        t = _make(id_="T1.1.1", status="backlog", blocks={"T1.2.2", "T1.4.7"})
        others = [
            t,
            _make(id_="T1.2.2", status="backlog", depends_on={"T1.1.1"}),
            _make(id_="T1.4.7", status="done"),
        ]
        assert rice.reach(t, others) == 2  # only T1.2.2 is not-done

    def test_reach_via_depends_on(self):
        """Ticket is depended on by 2 not-done tickets."""
        t = _make(id_="T1.1.1", status="backlog")
        others = [
            t,
            _make(id_="T1.2.2", status="backlog", depends_on={"T1.1.1"}),
            _make(id_="T1.4.7", status="in_progress", depends_on={"T1.1.1"}),
        ]
        assert rice.reach(t, others) == 3  # 1 + 2 not-done

    def test_reach_counts_done_dependents_not(self):
        """Done tickets do NOT count toward reach."""
        t = _make(id_="T1.0.1", status="done")
        others = [
            t,
            _make(id_="T1.1.1", status="backlog", depends_on={"T1.0.1"}),
            _make(id_="T1.1.3", status="done", depends_on={"T1.0.1"}),
        ]
        # T1.1.1 (backlog) counts; T1.1.3 (done) does not.
        # reach = 1 (base) + 1 (T1.1.1) = 2
        assert rice.reach(t, others) == 2


class TestScore(unittest.TestCase):
    def test_score_basic(self):
        """P0, reach=1 (no dependents), confidence=0.8, effort_h=0.5 → 1*3*0.8/0.5 = 4.8"""
        t = _make(id_="T1.1.1", priority="P0", confidence=0.8, effort_h=0.5)
        others = [t]
        assert abs(rice.score(t, others) - 4.8) < 0.01

    def test_score_zero_effort_defaults_to_8(self):
        """effort_h=0 → default 8h."""
        t = _make(id_="T1.1.1", priority="P0", confidence=0.8, effort_h=0)
        others = [t]
        assert abs(rice.score(t, others) - (1 * 3.0 * 0.8 / 8.0)) < 0.01

    def test_score_missing_confidence_defaults(self):
        """confidence=None → default 0.8."""
        t = _make(id_="T1.1.1", priority="P0", confidence=None, effort_h=1.0)
        others = [t]
        assert abs(rice.score(t, others) - (1 * 3.0 * 0.8 / 1.0)) < 0.01

    def test_score_reach_counts(self):
        """Same ticket, different reach: should differ by reach factor."""
        base = _make(id_="T1.1.1", priority="P0", confidence=0.8, effort_h=1.0)
        other = _make(id_="T1.2.2", status="backlog", depends_on={"T1.1.1"})
        others = [base, other]
        s = rice.score(base, others)
        assert abs(s - (2 * 3.0 * 0.8 / 1.0)) < 0.01  # reach=2


class TestRank(unittest.TestCase):
    def test_rank_sorts_by_score_desc(self):
        a = _make(id_="A", priority="P3", confidence=0.8, effort_h=1.0)
        b = _make(id_="B", priority="P0", confidence=0.8, effort_h=1.0)
        c = _make(id_="C", priority="P1", confidence=0.8, effort_h=1.0)
        ranked = rice.rank([a, b, c], [a, b, c])
        ids = [r[0]["id"] for r in ranked]
        assert ids == ["B", "C", "A"]

    def test_rank_returns_tuples(self):
        t = _make(id_="T1.1.1", priority="P0", confidence=0.8, effort_h=1.0)
        ranked = rice.rank([t], [t])
        assert isinstance(ranked, list)
        assert len(ranked) == 1
        assert isinstance(ranked[0], tuple)
        assert ranked[0][0]["id"] == "T1.1.1"
        assert isinstance(ranked[0][1], float)

    def test_rank_tie_break_priority(self):
        """Same RICE score: P1 should beat P3."""
        a = _make(id_="A", priority="P3", confidence=1.0, effort_h=1.0)
        b = _make(id_="B", priority="P1", confidence=1.0, effort_h=1.0)
        ranked = rice.rank([a, b], [a, b])
        ids = [r[0]["id"] for r in ranked]
        assert ids == ["B", "A"]


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make(
    id_: str,
    status: str = "backlog",
    priority: str = "P1",
    confidence: float | None = 0.8,
    effort_h: float = 8.0,
    depends_on: set = None,
    blocks: set = None,
) -> dict:
    return {
        "id": id_,
        "status": status,
        "priority": priority,
        "confidence": confidence,
        "effort_h": effort_h,
        "depends_on": depends_on or set(),
        "blocks": blocks or set(),
    }


if __name__ == "__main__":
    unittest.main()
