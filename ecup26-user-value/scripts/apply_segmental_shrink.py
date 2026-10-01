#!/usr/bin/env python3
"""M1.A2: apply segmental log-space shrink to a submission CSV (post-processing).

Reads a production submission (user_id,predict), applies the segmental
correction in log space: non-whale users (bottom (1-w_frac) by prediction)
are shrunk by c; the whale segment (top w_frac by prediction) is kept or
boosted by beta. This is the production form of m_shrink_except_whales.

Why segmental: the global shrink (T3.10.13) degraded on LB (1.6702) because
it also shrank the under-predicted whales (bias +1.74 on fold B, +2.34 on G).
C1.A3 confirmed the split (zeros shrink helps, whale shrink hurts); M1.A1
validated shrink_except_whales on B (+0.0097) and G (+0.0133), M=500, P=0.00.

No truth leak: the whale mask uses the prediction quantile, not labels.

Compliance: R1 (pandas/numpy), R6 (deterministic), R8 (dev-only), R10 (unique
user_id, predict >= 0 asserted on output).

Usage:
  uv run python scripts/apply_segmental_shrink.py \
      --input submissions/submit_blend_xxx.csv \
      --output /tmp/submit_seg_shrink.csv --c 0.95 --beta 1.0 --w-frac 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None,
                    help="output CSV path (default: derived from input + --task)")
    ap.add_argument("--c", type=float, default=0.95, help="log-space shrink for non-whales")
    ap.add_argument("--beta", type=float, default=1.0, help="log-space multiplier for whales")
    ap.add_argument("--w-frac", type=float, default=0.05, help="whale top fraction by prediction")
    ap.add_argument("--task", type=str, default="",
                    help="ledger task code appended to the output filename (e.g. M1.A4-add004)")
    ap.add_argument("--add-shift", type=float, default=0.0,
                    help="if != 0: ADDITIVE log-space shift (pred_log + add) for ALL users "
                         "instead of multiplicative c/beta split (shape test)")
    args = ap.parse_args()

    if args.output is None and args.task:
        args.output = args.input.with_name(f"{args.input.stem}_{args.task}.csv")

    df = pd.read_csv(args.input)
    assert list(df.columns) == ["user_id", "predict"], df.columns
    assert df["user_id"].is_unique, "R10: duplicate user_id"
    pred = np.maximum(0.0, df["predict"].to_numpy(dtype=np.float64))
    pred_log = np.log1p(pred)

    thr = np.quantile(pred_log, 1.0 - args.w_frac)
    whale = pred_log >= thr
    if args.add_shift:
        out_log = pred_log + args.add_shift  # additive shape test (all users)
    else:
        out_log = pred_log.copy()
        out_log[~whale] *= args.c
        out_log[whale] *= args.beta
    out = np.expm1(out_log)
    out = np.maximum(0.0, out)

    result = pd.DataFrame({"user_id": df["user_id"], "predict": out})
    result.to_csv(args.output, index=False)
    print(
        f"segmental shrink: n={len(result):,} c={args.c} beta={args.beta} "
        f"w_frac={args.w_frac} | mean {result.predict.mean():.2f} "
        f"({df.predict.mean():.2f} before) | whale_share={whale.mean():.3f}"
    )
    assert result["user_id"].is_unique, "R10: output duplicate user_id"
    assert (result["predict"] >= 0).all(), "R10: negative predictions"
    print(f"R10 OK: {len(result)} rows, unique ids, predict >= 0 -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
