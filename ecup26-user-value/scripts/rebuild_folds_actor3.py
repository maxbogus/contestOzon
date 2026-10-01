#!/usr/bin/env python3
"""Rebuild the production actor-feature fold cache (B/A/P) into data/folds/actor3.

Fold cutoffs match the production actor3 cache (used by gap_analysis,
conformal_threshold, shap_diagnostic and the B+A+P+G decision gate):
  A  cutoff = max_date - 60d
  B  cutoff = max_date - 30d
  P  cutoff = 2025-11-01 (promo spike)
Built with ECUP26_ACTOR_FEATURES=1 (153 features, same as production).

Compliance: R1/R6/R8. Forbidden: no 2026 target days used for features (folds
cut at 2025/early-2026 dates; train parquet holds facts <= 2026-02-13).

Usage: uv run python scripts/rebuild_folds_actor3.py
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

OUT = Path(os.environ.get("ECUP26_FOLD_OUT", "data/folds/actor3"))


def main() -> int:
    os.environ["ECUP26_ACTOR_FEATURES"] = "1"
    data = bench._load_clean(Path("data/train.parquet"))
    max_date = data["event_date"].max()
    print(f"max_date={max_date}")
    # P = promo spike fold (cutoff 2025-11-01) - matches the actor3 cache.
    for name, cutoff in (
        ("A", max_date - pd.Timedelta(days=60)),
        ("B", max_date - pd.Timedelta(days=30)),
        ("P", pd.Timestamp("2025-11-01")),
    ):
        bench._build_fold(data, cutoff, name, OUT)
        print(f"fold {name} built (cutoff {cutoff.date()})")
    print(f"OK: actor3 folds built -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
