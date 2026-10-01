#!/usr/bin/env python3
"""Scaffold a new backlog ticket from template."""

import argparse
from pathlib import Path

TEMPLATE = """---
id: {id}
phase: {phase}
title: {title}
priority: {priority}
effort: {effort}
unit: hours
rice:
  R: {r}
  I: {i}
  C: {c}
  score: {score}
depends_on: {depends_on}
blocks: {blocks}
tags: [{tags}]
repo: {repo}
status: backlog
created: {created}
---

## Context

TODO: describe the problem and why this task matters.

## RED

```bash
pytest tests/test_{slug}.py::test_{slug}_basic  # must fail
```

## GREEN

`src/{module}/{slug}.py`:
```python
# TODO: implement
```

## Verification

```bash
pytest tests/test_{slug}.py -v
```
"""

def main():
    p = argparse.ArgumentParser(description="Scaffold a new backlog ticket")
    p.add_argument("--id", required=True, help="Ticket ID, e.g. T1.0.1")
    p.add_argument("--title", required=True, help="Ticket title")
    p.add_argument("--repo", required=True, help="Repo: matching|quality|user-value|shared")
    p.add_argument("--phase", default="S1", help="Phase: S0-S5")
    p.add_argument("--priority", default="P1", help="Priority: P0|P1|P2")
    p.add_argument("--effort", type=float, default=1.0, help="Effort in hours")
    p.add_argument("--r", type=float, default=1.0, help="RICE Reach")
    p.add_argument("--i", type=float, default=2.0, help="RICE Impact")
    p.add_argument("--c", type=float, default=0.9, help="RICE Confidence")
    p.add_argument("--depends-on", default="[]", help="depends_on as JSON list")
    p.add_argument("--blocks", default="[]", help="blocks as JSON list")
    p.add_argument("--tags", default="task", help="Comma-separated tags")
    p.add_argument("--module", default="models", help="Src module: models|data|features")
    args = p.parse_args()

    from datetime import date
    import json
    import re

    # slug from title
    slug = re.sub(r"[^a-z0-9]+", "-", args.title.lower()).strip("-")[:50]

    # score
    score = round(args.r * args.i * args.c / args.effort, 1)

    # depends_on / blocks from JSON strings
    deps = json.loads(args.depends_on)
    blocks = json.loads(args.blocks)

    # Repo dir
    repo = args.repo.replace("user-value", "user-value")
    ticket_dir = Path(__file__).parent.parent / "docs" / "backlog" / "tickets"
    ticket_dir.mkdir(parents=True, exist_ok=True)

    filename = ticket_dir / f"{args.id}-{slug}.md"

    content = TEMPLATE.format(
        id=args.id,
        phase=args.phase,
        title=args.title,
        priority=args.priority,
        effort=args.effort,
        r=args.r,
        i=args.i,
        c=args.c,
        score=score,
        depends_on=json.dumps(deps),
        blocks=json.dumps(blocks),
        tags=args.tags,
        repo=args.repo,
        created=date.today().isoformat(),
        slug=slug,
        module=args.module,
    )

    filename.write_text(content)
    print(f"Created: {filename}")
    print(f"  ID: {args.id}")
    print(f"  RICE: {score} (R={args.r} × I={args.i} × C={args.c} / E={args.effort})")


if __name__ == "__main__":
    main()
