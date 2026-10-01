---
id: M-007
phase: S0
title: contest — migrate sweep infrastructure (runner + aggregator)
priority: P0
effort: 1.0
unit: hours
rice:
  R: 2
  I: 2
  C: 0.95
  score: 3.8
depends_on: [M-001]
blocks: [T1.5.3]
tags: [migration, sweep]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate sweep infrastructure from mlaw-rag (2 modules, ~192 lines).
Foundation for T1.5.3 parameter sweeps.

Source modules:
- application/sweep_runner.py (70 lines) — generic grid sweep
- application/sweep_aggregator.py (122 lines) — Pareto frontier

## GREEN

`shared/sweep/`:
- `runner.py`: SweepRunner.combinations() + run() + run_parallel()
- `aggregator.py`: pareto_frontier() + CSV export

## RED

```bash
pytest shared/tests/sweep/ -v  # 10+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/sweep/ -v
```

## Compliance

- **R1** (open-source code): Pure Python sweep runner
- **R5** (no plagiarism): Re-implements mlaw-rag sweep (same author)
- **R7** (single HF pipeline): CPU-only eval

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

