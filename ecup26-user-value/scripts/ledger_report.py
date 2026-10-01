#!/usr/bin/env python3
"""Decision-ledger v2 reader for E-CUP T3 (user-value).

Reads ``docs/research/decision_ledger_v2.jsonl`` (one atomic check per line)
and prints:

1. summary by status,
2. TODO checks grouped by parent ticket,
3. REOPEN candidates: closed checks (CONFIRMED/FALSIFIED/TIE) whose
   ``verdict_conditions`` mention a capability that now exists in the repo
   (fold G, actor features, segmental correction, ...).  This is the ledger's
   false-path detector: a verdict is conditional, and when the conditions
   change, the ledger flags it for review instead of keeping it as a hard ban.

Compliance: R1 (stdlib only), R6 (deterministic read), R8 (dev-only).
Usage::

    uv run python scripts/ledger_report.py
    uv run python scripts/ledger_report.py --jsonl /tmp/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = REPO_ROOT / "docs/research/decision_ledger_v2.jsonl"

#: Capabilities that exist NOW and may reopen a previously closed verdict.
NOW_AVAILABLE = (
    "fold g",
    "actor",
    "сегментн",
    "segment",
    "whale",
    "gift",
)


def load_rows(path: Path) -> list[dict]:
    """Load JSONL rows, skipping blank lines."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def reopen_candidates(rows: list[dict]) -> list[dict]:
    """Closed checks whose verdict conditions mention a now-available capability."""
    closed = {"CONFIRMED", "FALSIFIED", "TIE"}
    hits: list[dict] = []
    for row in rows:
        if row.get("status") not in closed:
            continue
        conditions = str(row.get("verdict_conditions", "")).lower()
        if any(kw in conditions for kw in NOW_AVAILABLE):
            hits.append(row)
    return hits


def _fmt_command(r: dict) -> str:
    cmd = str(r.get("command", "?"))
    return cmd if len(cmd) <= 140 else cmd[:137] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT)
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    if not rows:
        print("ledger is empty")
        return 1

    by_status = Counter(str(r.get("status", "?")) for r in rows)
    print(f"atomic checks in ledger: {len(rows)}")
    print("by status:", dict(by_status))

    todo = [r for r in rows if r.get("status") == "TODO"]
    if todo:
        print("\nTODO checks (grouped by parent ticket):")
        parents: dict[str, list[dict]] = {}
        for r in todo:
            for p in r.get("parents", ["?"]):
                parents.setdefault(str(p), []).append(r)
        for parent in sorted(parents):
            print(f"  {parent}:")
            for r in parents[parent]:
                print(f"    - {r['id']}: {r['hypothesis']}")
                print(f"      cmd: {_fmt_command(r)}")
                print(f"      gate: {r.get('gate_pre')}")

    reopen = reopen_candidates(rows)
    if reopen:
        print("\nREOPEN candidates (verdict conditions now available):")
        for r in reopen:
            print(f"  - {r['id']} [{r.get('status')}] {r['hypothesis']}")
            print(f"      revisit_trigger: {r.get('revisit_trigger')}")
    else:
        print("\nno reopen candidates")

    print(f"\nTODO: {sum(1 for r in rows if r.get('status') == 'TODO')} open, "
          f"{len(rows) - sum(1 for r in rows if r.get('status') == 'TODO')} closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
