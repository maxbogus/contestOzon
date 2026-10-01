#!/usr/bin/env python3
"""T0.0.2 — Verify downloaded data integrity.

Checks every dataset:
  - existence + file size
  - parquet: schema + row count (via matching repo's uv venv)
  - CSV: header + row count
  - ZIP: can be opened

Exit 0 = all OK, 1 = problems found.
Run: python scripts/verify_data.py [--repo matching|quality|user-value|all]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# {rel_path: {min_mb, cols?}}
EXPECTED = {
    "ecup26-matching/data/matches.parquet":      {"min_mb": 1,  "cols": ["id1", "id2", "target"]},
    "ecup26-matching/data/items.parquet":         {"min_mb": 100, "cols": ["id", "name", "attributes", "category"]},
    "ecup26-matching/data/items_human.parquet":   {"min_mb": 50,  "cols": ["id", "name", "attributes", "category"]},
    "ecup26-matching/data/matches_llm.parquet":   {"min_mb": 10,  "cols": ["id1", "id2", "target"]},
    "ecup26-quality/data/data.csv":               {"min_mb": 1},
    "ecup26-quality/data/images.zip":            {"min_mb": 500},
    "ecup26-user-value/data/train.parquet":      {"min_mb": 50},
    "ecup26-user-value/data/sample_submit.csv":   {"min_mb": 0.01},
}


def _parquet_via_uv(path: Path, spec: dict) -> tuple[bool, str]:
    """Read parquet using the matching repo's uv venv (has polars)."""
    cols_list = spec.get("cols", [])
    cols_py = repr(cols_list)
    check_script = (
        f"import polars as pl; "
        f"df = pl.read_parquet('{path}'); "
        f"print(f'rows={{df.height:,}},cols={{df.columns}}'); "
        f"missing = set({cols_py}) - set(df.columns); "
        f"exit(1 if missing else 0)"
    )
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", check_script],
            cwd="/home/maxbogus/Repositories/contest/ecup26-matching",
            capture_output=True, text=True, timeout=120,
        )
        out = result.stdout.strip()
        if result.returncode == 0:
            return True, out
        return False, f"{out}  (missing cols)"
    except subprocess.TimeoutExpired:
        return False, "timeout reading parquet"
    except Exception as e:
        return False, f"error: {e}"


def _csv_via_uv(path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c",
             f"import polars as pl; df = pl.read_csv('{path}'); "
             f"print(f'rows={{df.height:,}},cols={{df.columns}}')"],
            cwd="/home/maxbogus/Repositories/contest/ecup26-matching",
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, f"error: {e}"


def _zip_check(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if names:
                return True, f"entries={len(names)}, sample={names[:2]}"
            return False, "empty"
    except Exception as e:
        return False, f"error: {e}"


def _check(path: Path, spec: dict) -> tuple[bool, str]:
    if not path.exists():
        return False, "not found"
    size_mb = path.stat().st_size / 1e6
    if size_mb < spec.get("min_mb", 0):
        return False, f"size={size_mb:.1f} MB < min={spec['min_mb']} MB"
    s = path.suffix.lower()
    if s == ".parquet":
        return _parquet_via_uv(path, spec)
    if s == ".csv":
        return _csv_via_uv(path)
    if s == ".zip":
        return _zip_check(path)
    return True, f"size={size_mb:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", choices=["matching", "quality", "user-value", "all"], default="all")
    args = ap.parse_args()

    repos = {"matching", "quality", "user-value"} if args.repo == "all" else {args.repo}
    all_ok = True

    for repo in sorted(repos):
        print(f"\n{'='*60}\n  ecup26-{repo}\n{'='*60}")
        for rel, spec in EXPECTED.items():
            if repo not in rel:
                continue
            path = Path(f"/home/maxbogus/Repositories/contest/{rel}")
            ok, detail = _check(path, spec)
            size = path.stat().st_size / 1e6 if path.exists() else 0
            print(f"  {'[OK]' if ok else '[FAIL]'} {path.name}: {detail}  ({size:.0f} MB)")
            if not ok:
                all_ok = False

    print(f"\n{'='*60}")
    print("All checks passed" if all_ok else "Some checks FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
