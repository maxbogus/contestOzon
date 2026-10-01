---
id: M-001
phase: S0
title: contest — create shared/ package with reusable modules from mlaw-rag
priority: P0
effort: 3.0
unit: hours
rice:
  R: 5
  I: 3
  C: 0.9
  score: 4.5
depends_on: []
blocks: [T1.0.9, T1.2.2, T1.4.7, T1.4.8, T1.5.1, T1.5.2, T1.5.6, T1.5.7, T2.0.5, T3.0.1]
tags: [infra, migration, shared]
repo: contest
status: done
created: 2026-08-15
---

## Context

mlaw-rag donor project has 92 Python modules (~11K lines), 58 done tickets.
Reuse strategy documented in ADR-003. Create shared/ package at contest/ root
to avoid code duplication across ecup26-matching, ecup26-quality, ecup26-user-value.

## GREEN

`contest/shared/`:
```
shared/
├── ml/
│   ├── encoder.py        # embedding_service.py + MRL
│   ├── model_loader.py   # embedding_model.py + device detection
│   └── batch.py          # embedding_batch.py + checkpointing
├── nlp/
│   └── lemmatizer.py     # lemmatizer.py (Russian)
├── eval/
│   ├── metrics.py        # nDCG/recall/MRR/hit + PR-AUC
│   ├── runner.py         # eval_runner.py
│   ├── writer.py         # eval_results_writer.py
│   ├── control.py        # control_test.py (random baseline)
│   ├── ab_test.py        # ab_test.py (paired t-test)
│   ├── report.py         # eval_report.py (markdown + limitations)
│   └── figures.py        # figure_generator.py
├── sweep/
│   ├── runner.py         # sweep_runner.py
│   └── aggregator.py     # sweep_aggregator.py
├── pipeline/
│   └── orchestrator.py   # index_builder.py pattern
└── __init__.py
```

## RED

```bash
pytest shared/tests/ -v  # 20+ tests pass
python -c "from shared.ml.encoder import BiEncoder"  # must work
python -c "from shared.eval.ab_test import compute_lift"  # must work
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/ -v
# All shared tests pass
```

## Compliance

- **R1** (open-source code): Pure stdlib + numpy + polars
- **R5** (no plagiarism): Re-implements mlaw-rag patterns (same author — see ADR-003)
- **R7** (single HF pipeline): No inference server

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

