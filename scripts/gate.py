#!/usr/bin/env python3
"""Gate checker: python scripts/gate.py <TICKET_ID>
Exits 0 if ready, 2 if blocked."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from check_plan import can_start, get_ticket

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/gate.py <TICKET_ID>")
        sys.exit(1)
    tid = sys.argv[1]
    ok, reasons = can_start(tid)
    if ok:
        print(f"OK GATE PASSED: {tid} is ready")
        sys.exit(0)
    else:
        print(f"X GATE BLOCKED: {tid}")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(2)
