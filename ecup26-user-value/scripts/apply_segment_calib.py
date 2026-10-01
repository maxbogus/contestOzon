#!/usr/bin/env python3
"""T3.16: segment calibration for reactivated users (point 4 of the reactivation plan).

Segment "reactivated" = reactivated_last_14d == 1 OR recency_ratio > 2 (overdue
vs own cadence). For these users the model under-predicts the post-reactivation
surge, so we apply a SEPARATE log-space shrink factor c_react (> 1), everyone
else keeps the global c.

    pred_c = expm1(c_react * log1p(pred0))   for reactivated
    pred_c = expm1(c_glob  * log1p(pred0))   for others

Input: a production CSV with c=1.0 (log-space base). The segment flags are
recomputed from the full train (ref = 2026-02-13) with ECUP26_REACTIVATION_FEATURES.

Usage:
    ECUP26_REACTIVATION_FEATURES=1 uv run python scripts/apply_segment_calib.py \
        submissions/submit_..._c1.0.csv 1.045 1.15 [out_tag]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecup26_user_value.features.build_features import build_features  # noqa: E402
from scripts import benchmark_t3 as bench  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    c_glob = float(sys.argv[2])
    c_react = float(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else f"segcalib_cr{c_react}"
    df = pd.read_csv(src)
    assert df["user_id"].is_unique and len(df) == 250000
    os.environ.setdefault("ECUP26_REACTIVATION_FEATURES", "1")
    os.environ.setdefault("ECUP26_ACTOR_FEATURES", "1")
    os.environ.setdefault("ECUP26_CALENDAR_V2_FEATURES", "1")
    data = bench._load_clean(Path("data/train.parquet"))
    feats = build_features(data, ref_date=pd.Timestamp("2026-02-13"))
    seg = (
        (feats["reactivated_last_14d"] > 0.5) | (feats["recency_ratio"] > 2.0)
    ).reindex(df["user_id"]).fillna(False).to_numpy()
    pred0 = df["predict"].to_numpy(dtype=np.float64)
    log0 = np.log1p(np.clip(pred0, 0.0, None))
    log1 = np.where(seg, c_react * log0, c_glob * log0)
    pred = np.clip(np.expm1(log1), 0.0, None)
    out = df.copy()
    out["predict"] = pred
    out_name = src.with_name(f"{src.stem}_{tag}.csv")
    out.to_csv(out_name, index=False)
    print(f"segment: reactivated={(seg).mean():.1%} | c_glob={c_glob} c_react={c_react} "
          f"| mean={pred.mean():.2f} | {out_name.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
