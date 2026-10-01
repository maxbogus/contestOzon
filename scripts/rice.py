#!/usr/bin/env python3
"""RICE prioritization for contest backlog tickets.

RICE = (Reach × Impact × Confidence) / Effort

Adapted from mlaw-rag/backend/src/application/rice.py
"""
from __future__ import annotations

from typing import Any

# Impact multipliers per priority level
IMPACT: dict[str, float] = {
    "P0": 3.0,
    "P1": 2.0,
    "P2": 1.0,
    "P3": 0.5,
}

DEFAULT_CONFIDENCE = 0.8
DEFAULT_EFFORT_H = 8.0


def impact(priority: str) -> float:
    """Impact multiplier for a priority level."""
    return IMPACT.get(priority, 1.0)


def reach(ticket: dict[str, Any], all_tickets: list[dict[str, Any]]) -> int:
    """1 + number of not-done tickets that depend on this one.

    Counts forward via:
      - other.depends_on includes this ticket's id
      - other.id is in this ticket's blocks list
    """
    tid = ticket["id"]
    blocks = ticket.get("blocks", set())
    dependents = 0
    for other in all_tickets:
        if other["status"] == "done":
            continue
        if tid in other.get("depends_on", set()) or other["id"] in blocks:
            dependents += 1
    return 1 + dependents


def score(ticket: dict[str, Any], all_tickets: list[dict[str, Any]]) -> float:
    """RICE score: (Reach × Impact × Confidence) / Effort."""
    r = reach(ticket, all_tickets)
    i = impact(ticket.get("priority", "P1"))
    c = ticket.get("confidence") or DEFAULT_CONFIDENCE
    e = ticket.get("effort_h") or DEFAULT_EFFORT_H
    if e <= 0:
        e = DEFAULT_EFFORT_H
    return (r * i * c) / e


def rank(
    rankable: list[dict[str, Any]],
    all_tickets: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    """Sort ready tickets by RICE descending.

    Tie-breaking: higher priority (higher impact) first, then by id.
    """
    scored = [(t, score(t, all_tickets)) for t in rankable]
    return sorted(
        scored,
        key=lambda x: (
            -x[1],                              # RICE descending
            -impact(x[0].get("priority", "P1")),  # priority descending
            x[0]["id"],                          # id ascending
        ),
    )
