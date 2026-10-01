"""T3.9.12: apply per-bucket bias calibration to a submission.

Biases are fit on fold B (the closest-to-target proxy) by predicted-GMV
bucket; SHRINK hedges the transfer to the gift-season target (fold B is
over-predicted, but the target may differ). Cross-fold A->B gave -0.0100.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

EDGES = [0.0, 10.0, 100.0, 1000.0, 1e12]
BIASES = [-0.181, -0.244, -0.214, -0.073]  # fit on fold B (T3.9.12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, default="submit_final_refined.csv")
    ap.add_argument("--output", type=str, default="submit_final_calibrated.csv")
    ap.add_argument("--shrink", type=float, default=1.0)
    args = ap.parse_args()
    sub = pd.read_csv(args.input)
    raw = sub["predict"].values
    idx = np.digitize(raw, EDGES[1:-1])
    corr_log = np.log1p(raw) + args.shrink * np.array(BIASES)[idx]
    sub["predict"] = np.clip(np.expm1(corr_log), 0.0, None)
    sub.to_csv(args.output, index=False)
    print(f"calibrated x{args.shrink} -> {args.output}: mean {sub.predict.mean():.2f}, "
          f"unique {sub.user_id.is_unique}, neg {(sub.predict < 0).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
