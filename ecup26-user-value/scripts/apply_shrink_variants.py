#!/usr/bin/env python3
"""T3.14: generate c-shrink variants from one log-space production CSV.

The production run with --shrink 1.0 writes pred0 = expm1(log1p-space pred).
Applying a shrink factor c in log space is equivalent to
    pred_c = expm1(c * log1p(pred0))
which matches experiment_blend_production.py's `blend_log * shrink` exactly,
so we reuse ONE expensive model fit to emit several c-variants for the same-day
LB dial without retraining.

Usage:
    uv run python scripts/apply_shrink_variants.py <input_csv> c1 c2 c3 ...
Example:
    uv run python scripts/apply_shrink_variants.py \
        submissions/submit_blend_actor_retuned_2026-08-23_*.csv 1.03 1.045 1.06
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    df = pd.read_csv(src)
    assert {"user_id", "predict"}.issubset(df.columns), f"bad schema in {src}"
    assert df["user_id"].is_unique, "duplicate user_ids (R10)"
    pred0 = df["predict"].to_numpy(dtype=np.float64)
    log0 = np.log1p(np.clip(pred0, 0.0, None))
    base = src.stem
    for c in sys.argv[2:]:
        cf = float(c)
        pred_c = np.clip(np.expm1(cf * log0), 0.0, None)
        out = df.copy()
        out["predict"] = pred_c
        out_name = src.with_name(f"{base}_c{cf}.csv")
        out.to_csv(out_name, index=False)
        print(f"c={cf}: mean={pred_c.mean():.2f} -> {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
