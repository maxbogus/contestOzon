---
id: M-005
phase: S0
title: contest — migrate RRF fusion + retrieval cascade
priority: P0
effort: 1.0
unit: hours
rice:
  R: 3
  I: 3
  C: 0.95
  score: 8.55
depends_on: [M-001]
blocks: [T1.4.7, T1.4.8]
tags: [migration, fusion, cascade]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate RRF fusion and retrieval cascade from mlaw-rag (2 modules, ~116 lines).
Foundation for T1.4.7 two-stage inference.

Source modules:
- domain/rrf_fusion.py (37 lines) — pure function
- application/retrieval_cascade.py (79 lines) — cascade pattern

## GREEN

`shared/fusion/rrf.py`: rrf_fusion() pure function
`shared/cascade/pipeline.py`: RetrievalCascade pattern

## RED

```bash
pytest shared/tests/fusion/ shared/tests/cascade/ -v  # 10+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/fusion/ shared/tests/cascade/ -v
```

## Compliance

- **R1** (open-source code): Pure NumPy RRF
- **R5** (no plagiarism): Re-implements mlaw-rag cascade (same author)
- **R7** (single HF pipeline): CPU fusion

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

