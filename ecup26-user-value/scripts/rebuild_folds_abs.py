#!/usr/bin/env python3
"""T3.10.4: rebuild fold cache WITH absence/gap features (ECUP26_ABSENCE_FEATURES=1).

Creates data/folds/abs with the same fold cutoffs as the w180 cache
(max-60 -> A, max-30 -> B), plus the 5 new gap/absence columns. The baseline
cache data/folds/w180 stays untouched for the paired A/B.

Usage: ECUP26_ABSENCE_FEATURES=1 uv run python scripts/rebuild_folds_abs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import benchmark_t3 as bench  # noqa: E402

OUT = Path(os.environ.get("ECUP26_FOLD_OUT", "data/folds/abs"))


def main() -> int:
    data = bench._load_clean(Path("data/train.parquet"))
    max_date = data["event_date"].max()
    print(f"max_date={max_date}")
    for name, days in (("A", 60), ("B", 30)):
        cutoff = max_date - pd.Timedelta(days=days)
        bench._build_fold(data, cutoff, name, OUT)
        print(f"fold {name} built (cutoff {cutoff.date()})")

    # alignment check vs baseline cache (paired bootstrap needs identical order)
    from scripts import ab_validate as av

    Xa, ya, Xva, yta = av._load("A", OUT)
    Xb, yb, Xvb, ytb = av._load("B", OUT)
    base = Path("data/folds/w180")
    Xa0, _, _, _ = av._load("A", base)
    Xb0, _, _, _ = av._load("B", base)
    ok_a = Xa.index.equals(Xa0.index)
    ok_b = Xb.index.equals(Xb0.index)
    new_feats = [c for c in Xb.columns if c not in Xb0.columns]
    print(f"fold A aligned: {ok_a} | fold B aligned: {ok_b} | new feats: {new_feats}")
    assert ok_a and ok_b, "row order mismatch vs baseline cache"
    return 0


if __name__ == "__main__":
    sys.exit(main())
