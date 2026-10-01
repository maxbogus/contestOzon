---
id: M-006
phase: S0
title: contest — migrate pipeline orchestrator (submission pattern)
priority: P0
effort: 1.5
unit: hours
rice:
  R: 2
  I: 3
  C: 0.9
  score: 3.6
depends_on: [M-001]
blocks: [T1.4.8]
tags: [migration, pipeline]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate index_builder.py pattern from mlaw-rag (125 lines).
Used as template for T1.4.8 submission pipeline orchestrator.

Source: backend/src/application/index_builder.py

## GREEN

`shared/pipeline/orchestrator.py`:
- `SubmissionPipeline(reader, encoder, classifier, writer)` with injectable deps
- Stages: read → encode → predict → write
- Stats dataclass with summary()

## RED

```bash
pytest shared/tests/pipeline/ -v  # 8+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/pipeline/ -v
```

## Compliance

- **R1** (open-source code): Pure Python orchestrator
- **R5** (no plagiarism): Re-implements mlaw-rag pipeline (same author)
- **R7** (single HF pipeline): Wraps HF transformers

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

