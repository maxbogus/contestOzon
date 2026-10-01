#!/usr/bin/env python3
"""T3.19 v2: build per-user static feature cache (102 clean features) for the two-input model.

For each walk-forward ref (7 training origins + fold-B + TEST) compute
build_features(ref) -> apply feature_policy (drop + trend) -> [n, 102] float32,
saved as npz. The trainer reindexes to the sequence user_ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecup26_user_value.features.build_features import build_features  # noqa: E402
from scripts import benchmark_t3 as bench  # noqa: E402

ORIGINS = ["2025-06-18", "2025-07-18", "2025-08-17", "2025-09-16",
           "2025-10-16", "2025-11-15", "2025-12-15"]
VAL_REF = "2026-01-14"
TEST_REF = "2026-02-13"


def _apply_policy(df: pd.DataFrame) -> pd.DataFrame:
    pol = json.loads(Path("scripts/feature_policy.json").read_text())
    drop = [c for c in pol["drop"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    if {"to_ord_90d", "to_ord_30d"} <= set(df.columns):
        df["__trend_90_30"] = df["to_ord_90d"] - df["to_ord_30d"]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.parquet")
    ap.add_argument("--out-dir", default="data/static")
    args = ap.parse_args()
    df = bench._load_clean(Path(args.data))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ref in ORIGINS + [VAL_REF, TEST_REF]:
        name = {"2026-01-14": "B", "2026-02-13": "TEST"}.get(ref, ref)
        feats = build_features(df, ref_date=pd.Timestamp(ref))
        feats = _apply_policy(feats).set_index("user_id")
        X = feats.to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        path = out / f"static_{name}.npz"
        np.savez_compressed(path, X=X, user_ids=feats.index.to_numpy())
        print(f"{path.name}: X={X.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
