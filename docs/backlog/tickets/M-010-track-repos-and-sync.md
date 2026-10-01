---
id: M-010
phase: S1
title: contest — repo tracking + docs/backlog synchronization hygiene
priority: P2
effort: 1.0
unit: hours
rice:
  R: 1
  I: 2
  C: 1.0
  score: 2.0
depends_on: []
blocks: []
tags: [meta, infra, hygiene]
repo: contest
status: backlog
created: 2026-08-16
---

## Context

Two structural hygiene issues surfaced during 2026-08-16 session:

### Issue A: pre-existing modified pyproject.toml in ecup26-user-value

As of 2026-08-16, `ecup26-user-value/pyproject.toml` shows as `M` (modified)
in `git status`. The diff has two lines:

```
+ "lifetimes>=0.11.3",       # pre-existing (added before 2026-08-16 session)
+ "lightgbm>=4.7.0",        # added during 2026-08-16 session via `uv add --dev lightgbm`
```

The `lightgbm` line is now safe to commit (T3.6.12 + T3.6.18 require it for
ablation). The `lifetimes` line is pre-existing and was used by T3.6.4
(`models/btyd_baseline.py` imports lifetimes). Without it, the project does
not reproduce from a fresh checkout.

**Status**: Issue A resolved in commit f69801e (lifetimes dep pinned).
Remaining: Issue B (git init in contest/) - deferred to user decision.

**Action**: commit both lines in one `chore: pin lifetimes + lightgbm deps`
commit, citing the tickets that depend on each.

### Issue B: contest/ workspace is untracked

`/home/maxbogus/Repositories/contest/` is NOT a git repository. The
following files are untracked and lose changes on every cleanup:

- `Makefile` — orchestrates the 4 sub-repos (matching/quality/user-value/shared)
- `ECUP_RULES.md` — rule cheatsheet (source of truth for R1-R10)
- `README.md` — workspace-level README
- `docs/` — contest-level docs (backlog mirror, tasks, notes, research)
- `scripts/check_plan.py` (and friends: help.py, rice.py, inventory.py, etc.)
- `scripts/tests/test_check_plan.py` (and other tests)
- `ai/`, `models/` — contest-level infrastructure (e.g. AI prompts)

Each sub-repo (ecup26-matching, ecup26-quality, ecup26-user-value, shared)
has its own .git/ — but the workspace above is duplicated state with no
history. Every change to `check_plan.py` or `ECUP_RULES.md` is invisible
to git until manually saved elsewhere.

**Action**: decide policy:
- (A) `git init` `contest/` and track all of it (Makefile, ECUP_RULES.md,
  scripts/, docs/, ai/, models/, README.md).
- (B) Symlink contest/docs/backlog -> ecup26-user-value/docs/backlog (kills
  the mirror duplication but loses contest-level structure).
- (C) Add `contest/` as a new git repo with a curated subset
  (Makefile, ECUP_RULES.md, scripts/, docs/backlog only).
- (D) Leave as-is, document the manual `cp` sync convention, accept that
  `check_plan.py` lives outside version control.

The 2026-08-16 session has been doing (D) with manual `cp` after every
archive operation. This works for 1 person but breaks for collaborators.

## GREEN

This ticket documents the issues; the actual fix is a separate decision
that the user must approve. After user picks a path:

- (A): `cd contest && git init && git add Makefile ECUP_RULES.md README.md scripts/ docs/ ai/ models/ && git commit -m '...'`. Then `check_plan.py` edits are tracked.
- (B): `cd contest/docs && rm -rf backlog && ln -s ../../ecup26-user-value/docs/backlog backlog`. Risky if sub-repo structure changes.
- (C): subset only — needs to enumerate which files are tracked.
- (D): leave as-is, document the `cp` sync in README.

## Verification

After decision:
- `cd contest && git status` shows no uncommitted changes for the chosen subset
- `check_plan.py --status T3.6.16` returns the same result whether run from
  `/home/maxbogus/Repositories/contest` or from inside the new repo
- `cp` is no longer needed after every archive operation

## Compliance

No R-rules apply directly (this is meta-infrastructure). But:
- R6 (reproducible): pinning `lifetimes` in pyproject.toml is required so
  fresh checkouts reproduce
- R7 (single HF pipeline): no inference code involved

## Forbidden Actions

- ❌ Do NOT commit pre-existing changes attributed to other contributors
  without explicit review.
- ❌ Do NOT `git init` `contest/` without explicit user approval (creates new
  repository outside scope of any sub-repo).
- ❌ Do NOT delete `docs/backlog/` mirror files without a decision on (A)/(B)/(C).
