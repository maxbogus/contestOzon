---
id: M-002
phase: S0
title: contest — migrate eval framework from mlaw-rag (8 modules)
priority: P0
effort: 2.0
unit: hours
rice:
  R: 4
  I: 3
  C: 0.95
  score: 5.7
depends_on: [M-001]
blocks: [T1.5.1, T1.5.2, T1.5.4, T1.5.6, T1.5.7]
tags: [migration, eval]
repo: contest
status: done
created: 2026-08-15
---

## Context

Migrate the eval framework from mlaw-rag (8 modules, ~1500 lines) to contest/shared/eval/.
This is the foundation for T1.5.x eval pipeline.

Source modules (all done in mlaw-rag):
- application/metrics.py (83 lines) — nDCG/recall/MRR/hit
- application/eval_runner.py (103 lines) — latency tracking
- application/eval_results_writer.py (54 lines) — CSV/JSON
- application/control_test.py (87 lines) — random baseline
- application/ab_test.py (213 lines) — paired t-test
- application/eval_report.py (266 lines) — markdown report
- application/figure_generator.py (107 lines) — matplotlib
- application/ablation.py (103 lines) — variant comparison

## GREEN

`shared/eval/` with all 8 modules adapted:
- Replace LegalAct/Chunk with Item/ItemPair entities
- Add PR-AUC wrapper for E-CUP T1 metric
- Keep all statistical tests (paired t-test via _betai)
- Mandatory limitations section in report

## RED

```bash
pytest shared/tests/eval/ -v  # 30+ tests pass
```

## Verification

```bash
cd /home/maxbogus/Repositories/contest
uv run pytest shared/tests/eval/ -v --tb=short
# 30+ tests pass
```

## Compliance

- **R1** (open-source code): scikit-learn + scipy
- **R5** (no plagiarism): Re-implements mlaw-rag eval (same author)
- **R6** (reproducible): Pinned metric versions
- **R7** (single HF pipeline): Pure CPU eval

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

