#!/usr/bin/env python3
"""Cache cleanup for E-CUP T3 (policy v2, 2026-08-20).

Single source of truth for all run caches: ``<repo>/data/`` (gitignored).
- fold caches : ``<repo>/data/folds/<name>``
- resume cache: ``<repo>/data/.cache``  (content-addressed hash groups)

Policy v2:
1. Limit: 10 GB total for ``data/folds`` + ``data/.cache`` (default, override
   with ``--limit-gb``).
2. Keep MORE hashes - evict ONLY when the total is over the limit.
3. Never delete hashes / fold dirs listed in ``data/.cache/top_candidates.json``
   (hashes of top-LB candidates, so a winning submission stays reproducible).
4. Eviction order: oldest mtime first; for the resume cache the most recent
   hash group (active config) is protected too.

``top_candidates.json`` shape::

    {"hashes": ["43822300...", ...], "folds": ["giftG", ...]}

Compliance: R1 (stdlib only), R6 (deterministic), R8 (dev-only).
Usage::

    uv run python scripts/cache_cleanup.py            # report + evict if over limit
    uv run python scripts/cache_cleanup.py --limit-gb 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS = REPO_ROOT / "data" / "folds"
RESUME = REPO_ROOT / "data" / ".cache"
TOP_FILE = REPO_ROOT / "docs" / "research" / "top_candidates.json"
DEFAULT_LIMIT_GB = 10.0


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _load_top() -> dict:
    if TOP_FILE.exists():
        return json.loads(TOP_FILE.read_text(encoding="utf-8"))
    return {"hashes": [], "folds": []}


def _hash_of(filename: str) -> str:
    """Extract the content-addressed hash from a resume-cache filename."""
    stem = filename.split(".")[0]
    for part in stem.split("_"):
        if len(part) >= 16 and part.isalnum():
            return part
    return "?"


def _resume_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not RESUME.exists():
        return groups
    for f in RESUME.iterdir():
        if f.is_file() and "_" in f.name:
            groups.setdefault(_hash_of(f.name), []).append(f)
    return groups


def _fmt(nbytes: int) -> str:
    return f"{nbytes / 2**30:.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-gb", type=float, default=DEFAULT_LIMIT_GB)
    ap.add_argument("--dry-run", action="store_true", help="report only, delete nothing")
    args = ap.parse_args()

    top = _load_top()
    protected_hashes = set(top.get("hashes", []))
    protected_folds = set(top.get("folds", []))

    folds_size = _dir_size(FOLDS)
    resume_size = _dir_size(RESUME)
    total = folds_size + resume_size
    limit = int(args.limit_gb * 2**30)

    print(f"data/folds  : {_fmt(folds_size)}  ({len(list(FOLDS.glob('*'))) if FOLDS.exists() else 0} dirs)")
    print(f"data/.cache : {_fmt(resume_size)}  ({len(_resume_groups())} hash groups)")
    print(f"total       : {_fmt(total)} / limit {args.limit_gb:.1f} GB")

    if total <= limit:
        print("under limit - nothing to evict")
        return 0

    # eviction candidates: fold dirs (oldest first), resume hash groups (oldest first)
    candidates: list[tuple[float, str, Path]] = []  # (mtime, label, path)
    if FOLDS.exists():
        for d in sorted(FOLDS.iterdir(), key=lambda p: p.stat().st_mtime):
            if d.is_dir() and d.name not in protected_folds:
                candidates.append((d.stat().st_mtime, f"fold:{d.name}", d))
    groups = _resume_groups()
    if groups:
        newest_hash = max(groups, key=lambda h: max(p.stat().st_mtime for p in groups[h]))
        for h, files in groups.items():
            if h in protected_hashes or h == newest_hash:
                continue
            mtime = max(p.stat().st_mtime for p in files)
            candidates.append((mtime, f"hash:{h[:12]}", files[0].parent / h))

    candidates.sort()
    freed = 0
    for _, label, path in candidates:
        if total - freed <= limit:
            break
        size = _dir_size(path)
        if args.dry_run:
            print(f"[dry-run] would delete {label} ({_fmt(size)})")
        else:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                for f in path.parent.glob(path.name + "*"):
                    f.unlink()
            print(f"deleted {label} ({_fmt(size)})")
        freed += size

    print(f"freed {_fmt(freed)}; remaining total {_fmt(total - freed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
