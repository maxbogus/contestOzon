---
id: M-003
phase: S0
title: contest — migrate embedding infrastructure (encoder + model + batch)
priority: P0
effort: 2.0
unit: hours
rice:
  R: 4
  I: 3
  C: 0.9
  score: 5.4
depends_on: [M-001]
blocks: [T1.0.9, T1.0.10, T1.0.11, T1.2.2, T1.4.7, T3.5.3]
tags: [migration, ml]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate embedding infrastructure from mlaw-rag (3 modules, ~200 lines).
Foundation for T1.0.9 BiEncoder and T1.0.10 batch inference.

Source modules:
- application/embedding_service.py (51 lines) — MRL truncation
- application/embedding_model.py (56 lines) — BGE-M3 loader
- application/embedding_batch.py (94 lines) — batch + checkpoint

## GREEN

`shared/ml/`:
- `encoder.py`: BiEncoder with MRL + L2 normalize
- `model_loader.py`: load_model with device detection (CUDA/CPU+offload)
- `batch.py`: encode_batches with per-batch checkpoint

## RED

```bash
pytest shared/tests/ml/ -v  # 15+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/ml/ -v
# All tests pass
```

## Compliance

- **R1** (open-source code): transformers + torch, Apache 2.0/BSD
- **R5** (no plagiarism): Re-implements mlaw-rag embeddings (same author)
- **R6** (reproducible): Pinned model versions
- **R7** (single HF pipeline): HF transformers in-process

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

