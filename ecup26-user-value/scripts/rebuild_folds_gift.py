#!/usr/bin/env python3
"""T3.10.28 E1 - build extended gift-season fold caches (G / G2 / G3).

Fold G truth = (2025-02-14, 2025-03-15] = gift season 2025 (target window shifted
-1y). Data starts 2025-01-01, so walk-forward origins are inherently limited; a
small step (7d) maximises the origin count. Uses the CURRENT production feature
set (ECUP26_ACTOR_FEATURES=1) so the production blend can be evaluated on it.

Builds into OUT (=data/folds/giftG, override via ECUP26_FOLD_OUT):
  G   cutoff 2025-02-14  truth = Feb 14 - Mar 15 2025 (exact gift proxy)
  G2  cutoff 2025-02-23  truth = Feb 23 - Mar 24 2025 (Feb-23 holiday window)
  G3  cutoff 2025-03-08  truth = Mar 8 - Apr 7 2025  (Mar-8 holiday window)

Compliance: R1/R6/R8. Forbidden: no 2026 target days used for features (all
origins are <= the 2025 cutoffs; the train parquet only holds facts <= 2026-02-13
but folds cut at 2025 dates).

Usage: ECUP26_ACTOR_FEATURES=1 uv run python scripts/rebuild_folds_gift.py
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

OUT = Path(os.environ.get("ECUP26_FOLD_OUT", "data/folds/giftG"))
#: max origins / small step to fit as many origins as possible inside 2025-01-01..
CUTOFFS = (("G", "2025-02-14"), ("G2", "2025-02-23"), ("G3", "2025-03-08"))


def _report_fold(name: str, cache_dir: Path) -> None:
    train = pd.read_parquet(cache_dir / f"fold_{name}_train.parquet")
    n_origins = train["origin"].nunique() if "origin" in train.columns else 0
    truth = pd.read_parquet(cache_dir / f"fold_{name}_truth.parquet")["y_true"]
    infer = pd.read_parquet(cache_dir / f"fold_{name}_infer.parquet")
    print(
        f"fold {name}: train_rows={len(train):,} users={train['user_id'].nunique():,} "
        f"origins={n_origins} | infer_users={len(infer):,} | "
        f"truth_users={len(truth):,} zero_frac={float((truth == 0).mean()):.4f} "
        f"mean={float(truth.mean()):.1f} n_feats={infer.shape[1]}"
    )


def main() -> int:
    # Current production feature set + extended walk-forward grid.
    os.environ["ECUP26_ACTOR_FEATURES"] = "1"
    os.environ["ECUP26_WALK_FORWARD_ORIGINS"] = "12"
    os.environ["ECUP26_WALK_FORWARD_STEP_DAYS"] = "7"
    print(f"OUT={OUT}")

    data = bench._load_clean(Path("data/train.parquet"))
    print(f"loaded train: {len(data):,} rows, max_date={data['event_date'].max().date()}")

    for name, cutoff_str in CUTOFFS:
        cutoff = pd.Timestamp(cutoff_str)
        bench._build_fold(data, cutoff, name, OUT)
        _report_fold(name, OUT)

    del data
    print("\nOK: gift folds built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
