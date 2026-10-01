#!/usr/bin/env python3
"""T3.10.15: rebuild fold cache WITH gift-holiday pre/post flags.

Creates data/folds/holiday with the same fold cutoffs as the actor3
cache (B = max-30, A = max-60, P = promo 2025-11-01) plus the 6 new
is_pre/post_holiday_* columns (ECUP26_HOLIDAY_PREPOST_FEATURES=1) on top of
the production actor features (ECUP26_ACTOR_FEATURES=1). The actor3 cache stays
touched for the paired A/B (dual-cache comparison in ab_validate).

Usage: uv run python scripts/rebuild_folds_holiday.py
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

OUT = Path(os.environ.get("ECUP26_FOLD_OUT", "data/folds/holiday"))
BASELINE = Path("data/folds/actor3")


def main() -> int:
    os.environ["ECUP26_ACTOR_FEATURES"] = "1"
    os.environ["ECUP26_HOLIDAY_PREPOST_FEATURES"] = "1"
    data = bench._load_clean(Path("data/train.parquet"))
    max_date = data["event_date"].max()
    print(f"max_date={max_date}")
    # P = promo spike fold (cutoff 2025-11-01) - matches the actor3 cache.
    for name, cutoff in (("A", max_date - pd.Timedelta(days=60)),
                         ("B", max_date - pd.Timedelta(days=30)),
                         ("P", pd.Timestamp("2025-11-01"))):
        bench._build_fold(data, cutoff, name, OUT)
        print(f"fold {name} built (cutoff {cutoff.date()})")

    # alignment check vs baseline cache (paired bootstrap needs identical order)
    from scripts import ab_validate as av

    new_feats: set[str] = set()
    for name in ("A", "B", "P"):
        X, _, _, _ = av._load(name, OUT)
        X0, _, _, _ = av._load(name, BASELINE)
        ok = X.index.equals(X0.index)
        print(f"fold {name} aligned: {ok} | new feats: {sorted(set(X.columns) - set(X0.columns))}")
        new_feats |= set(X.columns) - set(X0.columns)
        assert ok, f"row order mismatch for fold {name}"
    assert new_feats == {"is_pre_holiday_7d_feb23", "is_pre_holiday_2d_feb23",
                         "is_post_holiday_7d_feb23", "is_pre_holiday_7d_mar8",
                         "is_pre_holiday_2d_mar8", "is_post_holiday_7d_mar8"}, \
        f"unexpected new feats: {new_feats}"
    print("OK: holiday folds built, 6 new flag columns verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
