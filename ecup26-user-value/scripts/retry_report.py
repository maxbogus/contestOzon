#!/usr/bin/env python3
"""Read docs/research/retry_candidates.jsonl and print the retry-queue summary.

Each session appends rows to the JSONL; this one-pass reader builds the
variant x new_capability view and lists retry candidates (YES first, then
MAYBE) grouped by category, so future sessions can pick the next experiment.

Compliance: R1 (stdlib only), R6 (deterministic read), R8 (dev-only).
Usage: uv run python scripts/retry_report.py [--jsonl docs/research/retry_candidates.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = REPO_ROOT / "docs/research/retry_candidates.jsonl"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT)
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    print(f"variants in JSONL: {len(rows)}")
    retries = [r for r in rows if r["retry"] in ("YES", "MAYBE")]
    order = {"calibration": 0, "segmentation": 1, "zero-inflation": 2,
            "features": 3, "NN": 4, "other": 5, "ensemble": 6, "target-transform": 7}
    retries.sort(key=lambda r: (r["retry"] != "YES", order.get(r["category"], 9), r["date"]))

    print("\nretry candidates (sorted):")
    for r in retries:
        flag = "YES " if r["retry"] == "YES" else "MAY"
        print(f"  [{flag}] {r['id']:<32} ({r['category']}, {r['date']})")
        print(f"       new_since: {r['new_since']}")
        print(f"       retry_with: {r['retry_with']}")
    print(f"\n{len(retries)} retry candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
