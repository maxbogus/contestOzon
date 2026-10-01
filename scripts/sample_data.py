#!/usr/bin/env python3
"""Sample data with token/row limits for safe experimentation.

Usage:
    python scripts/sample_data.py              # list available datasets
    python scripts/sample_data.py --all        # sample all datasets
    python scripts/sample_data.py --matches    # sample matches.parquet
    python scripts/sample_data.py --items      # sample items_human.parquet
    python scripts/sample_data.py --train      # sample train.parquet
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

DEFAULT_LIMITS = {
    "matches.parquet": 5_000,
    "matches_llm.parquet": 5_000,
    "items_human.parquet": 1_000,
    "data.csv": 2_000,
    "train.parquet": 5_000,
}


def find_data_dir() -> Path:
    for name in ["ecup26-matching", "ecup26-quality", "ecup26-user-value"]:
        candidate = Path(__file__).parents[1] / name / "data"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("data directory not found in any sub-repo")


def sample_parquet(path: Path, limit: int) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, skipping parquet")
        return

    df = pd.read_parquet(path)
    original = len(df)
    if original <= limit:
        print(f"  {path.name}: {original} rows (under limit {limit}, no sampling)")
        return

    sampled = df.sample(n=limit, random_state=42)
    out = path.parent / f"{path.stem}_sample.parquet"
    sampled.to_parquet(out, index=False)
    print(f"  {path.name}: {original} -> {len(sampled)} rows -> {out}")


def sample_csv(path: Path, limit: int) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, skipping csv")
        return

    df = pd.read_csv(path)
    original = len(df)
    if original <= limit:
        print(f"  {path.name}: {original} rows (under limit {limit}, no sampling)")
        return

    sampled = df.sample(n=limit, random_state=42)
    out = path.parent / f"{path.stem}_sample.csv"
    sampled.to_csv(out, index=False)
    print(f"  {path.name}: {original} -> {len(sampled)} rows -> {out}")


def main() -> None:
    args = set(sys.argv[1:])

    data_dir = find_data_dir()
    print(f"Data directory: {data_dir}")
    print(f"Default limits: {json.dumps(DEFAULT_LIMITS, indent=2)}")
    print()

    if not args or "--all" in args:
        targets = DEFAULT_LIMITS
    else:
        targets = {}
        for key in DEFAULT_LIMITS:
            flag = "--" + key.replace(".parquet", "").replace(".csv", "")
            if flag in args:
                targets[key] = DEFAULT_LIMITS[key]

    for fname, limit in targets.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  {fname}: not found, skipping")
            continue
        if fname.endswith(".parquet"):
            sample_parquet(path, limit)
        elif fname.endswith(".csv"):
            sample_csv(path, limit)
        else:
            print(f"  {fname}: unknown extension, skipping")


if __name__ == "__main__":
    main()
