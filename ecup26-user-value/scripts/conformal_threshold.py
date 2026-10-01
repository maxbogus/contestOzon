#!/usr/bin/env python3
"""T3.10.23 - split-conformal diagnostic for the zero/positive threshold.

Hard-zero was already FALSIFIED (T3.10.2: best heuristic t=0.05 = no threshold),
so this is DIAGNOSTIC ONLY: it computes the calibrated conformal threshold for
P(y=0) with a finite-sample coverage guarantee and compares it with the heuristic.
NO zeroing is applied (forbidden by the ticket).

Protocol (split conformal, classification):
  1. fit the binary P(y>0) classifier on fold B train;
  2. calibration split (20% seeded): nonconformity score of the true label;
  3. q_hat = higher-quantile of scores at level ceil((n+1)(1-alpha))/n;
  4. label 0 is in the prediction set iff P(y>0) <= q_hat (so q_hat IS the
     calibrated zero-membership threshold); report vs heuristic t=0.05.

Compliance: R1 (numpy/lightgbm), R6 (seeded), R8 (dev-only).
Usage: uv run python scripts/conformal_threshold.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import ab_validate as av  # noqa: E402

CACHE_ACTOR = Path("data/folds/actor3")


def _nonconformity(p1, y_bin) -> np.ndarray:
    """Nonconformity score of the true label (0: P(y>0); 1: 1 - P(y>0))."""
    p1 = np.asarray(p1, dtype=np.float64)
    y = np.asarray(y_bin)
    return np.where(y == 0, p1, 1.0 - p1)


def _conformal_qhat(scores, alpha: float) -> float:
    """Finite-sample conformal quantile at level ceil((n+1)(1-alpha))/n."""
    n = len(scores)
    q_level = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="B")
    ap.add_argument("--cache", type=Path, default=CACHE_ACTOR)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, Xv, yt_log = av._load(args.fold, args.cache)
    y_arr = np.asarray(y, dtype=np.float64)
    y_bin = (y_arr > 0).astype(np.int32)
    print(f"fold {args.fold}: train={len(y_bin):,} infer={len(Xv):,} zero_frac={float((y_bin == 0).mean()):.4f}")

    clf = av._train_clf(X, y_bin, args.seed)
    p1 = np.asarray(clf.predict(X.fillna(0.0)), dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    cal = rng.choice(len(y_bin), size=int(args.cal_frac * len(y_bin)), replace=False)
    scores = _nonconformity(p1[cal], y_bin[cal])
    q_hat = _conformal_qhat(scores, args.alpha)

    # infer side: empirical coverage + implied zero-membership rate.
    p1_test = np.asarray(clf.predict(Xv.fillna(0.0)), dtype=np.float64)
    y_test = (np.expm1(yt_log) > 0).astype(np.int32)
    set_zero = p1_test <= q_hat        # label 0 in the prediction set
    set_pos = p1_test >= 1.0 - q_hat   # label 1 in the prediction set
    covered = (set_zero & (y_test == 0)) | (set_pos & (y_test == 1))
    emp_cov = float(covered.mean())

    heuristic = 0.05  # T3.10.2 best zero-threshold (no threshold effectively)
    print("=" * 60)
    print(f"calibration n={len(scores):,}  alpha={args.alpha}  q_hat={q_hat:.4f}")
    print(f"conformal zero-membership threshold (P(y>0) <= q_hat): {q_hat:.4f}")
    print(f"heuristic zero-threshold (T3.10.2 best t):             {heuristic:.4f}")
    print(f"implied zero-membership rate (diagnostic): {float(set_zero.mean()):.4f}")
    print(f"empirical coverage on infer users: {emp_cov:.4f} (target >= {1 - args.alpha:.2f})")
    print(f"conformal set size / prediction: {float((set_zero + set_pos).mean()):.4f}")
    print("role: DIAGNOSTIC - no zeroing applied (T3.10.2 FALSIFIED).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
