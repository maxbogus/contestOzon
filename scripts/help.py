#!/usr/bin/env python3
"""Pretty help output — parses ## descriptions from Makefile."""
import re
import sys
from pathlib import Path

MAKEFILE = Path(__file__).parents[1] / "Makefile"


def main() -> None:
    print("\033[1mcontest - Backlog Management\033[0m")
    print("Usage: make <target>\n")

    text = MAKEFILE.read_text()
    pattern = re.compile(r"^([a-zA-Z_-][a-zA-Z0-9_-]*)\s*:\s*.*?##\s*(.*)$", re.MULTILINE)

    targets = []
    for m in pattern.finditer(text):
        name, desc = m.group(1), m.group(2).strip()
        if name not in ("help", "DEFAULT_GOAL", "PHONY"):
            targets.append((name, desc))

    for name, desc in targets:
        print(f"  \033[36m{name:<16}\033[0m {desc}")

    print()


if __name__ == "__main__":
    main()
