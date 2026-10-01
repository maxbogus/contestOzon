---
id: M-008
phase: S0
title: contest — RUNBOOK template + GPU lessons from T-057/T-015
priority: P0
effort: 0.5
unit: hours
rice:
  R: 2
  I: 3
  C: 0.95
  score: 11.4
depends_on: []
blocks: []
tags: [docs, runbook, gpu]
repo: contest
status: done
created: 2026-08-15
---

## Context

Document GPU lessons learned from mlaw-rag T-057/T-015:
- 8GB VRAM OOM (reranker fp16 + max_length=2048 + batch=16 + expandable_segments)
- chunk_id collisions (15% chunks lost silently)

Create RUNBOOK template for each E-CUP submission track.

## GREEN

`docs/RUNBOOK.md` (master template):
- §0 Предусловия (nvidia-smi, uv sync, .env)
- §1 Инфраструктура (Docker, no internet)
- §2 Сборка индекса (model load, batch size, OOM prevention)
- §3 Eval (metrics, control)
- §4 Submission (Docker build, time limits)
- §5 Gotchas (chunk_id uniqueness, OOM fixes)

`docs/notes/001-gpu-oom-8gb.md` — урок про 8GB VRAM
`docs/notes/002-chunk-id-collision.md` — урок про уникальные ID

## RED

```bash
cat docs/RUNBOOK.md  # must exist
cat docs/notes/001-gpu-oom-8gb.md  # must exist
```

## Verification

```bash
ls docs/RUNBOOK.md docs/notes/00*.md
```


## Acceptance Criteria

- [x] docs/notes/001-gpu-oom-8gb.md created
- [x] docs/notes/002-chunk-id-collision.md created
- [x] docs/RUNBOOK.md created with all 7 sections

## Verification

```bash
ls docs/RUNBOOK.md docs/notes/00*.md
# RUNBOOK.md exists
# 001-gpu-oom-8gb.md exists
# 002-chunk-id-collision.md exists
```

✅ VERIFIED 2026-08-15: All 3 files created with substantive content.

## Compliance

- **R1** (open-source code): Markdown documentation
- **R5** (no plagiarism): Self-authored
- **R7** (single HF pipeline): No inference code

## Forbidden Actions

- ❌ Do NOT introduce dependencies on proprietary code (R1).
- ❌ Do NOT skip rule-references in tickets (W1, W2).

