---
id: M-004
phase: S0
title: contest — migrate sparse retrieval (BM25 + lemmatizer)
priority: P0
effort: 1.5
unit: hours
rice:
  R: 3
  I: 3
  C: 0.9
  score: 5.4
depends_on: [M-001]
blocks: [T1.2.2, T1.4.7, T3.5.3]
tags: [migration, sparse]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate sparse retrieval from mlaw-rag (3 modules, ~234 lines).
Foundation for hybrid retrieval in T1.4.7 cascade.

Source modules:
- infrastructure/bm25_index.py (71 lines) — bm25s + lemmatizer
- infrastructure/bm25_search.py (34 lines) — sparse search
- infrastructure/lemmatizer.py (106 lines) — Russian lemmatizer

## GREEN

`shared/nlp/` + `shared/sparse/`:
- `lemmatizer.py`: RussianLemmatizer (mawo-pymorphy3)
- `bm25_index.py`: BM25Index build/save/load
- `bm25_search.py`: sparse_search → RerankedItem

## RED

```bash
pytest shared/tests/sparse/ shared/tests/nlp/ -v  # 15+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/sparse/ shared/tests/nlp/ -v
# All tests pass
```

## Compliance

- **R1** (open-source code): rank_bm25 / custom BM25, MIT
- **R5** (no plagiarism): Re-implements mlaw-rag sparse (same author)
- **R7** (single HF pipeline): CPU sparse retrieval

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

