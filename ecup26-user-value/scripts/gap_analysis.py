#!/usr/bin/env python3
"""T3.10.27 E1 - decompose RMSLE error^2 into zeros / positives / whales.

Loads fold B (production actor-feature cache) and the gift-proxy fold G, trains the
production-representative model (blend_retuned = actor_retuned config) and splits the
total log-space squared error by (zero vs positive) and whale tier (top-1% / top-5% by
raw 30d GMV truth). Output is a ranked lever list that feeds T3.10.19 / T3.10.28 /
T3.10.15 / T3.10.16.

Conventions:
  - bias = truth_log - pred_log  (bias > 0 => under-predicted, < 0 => over-predicted)
  - whale = user whose true 30d GMV is >= the (1 - w_frac) quantile of the fold truth
  - error^2 = (pred_log - truth_log)^2 summed over users; RMSLE = sqrt(mean(error^2))

Compliance: R1 (lightgbm/catboost/numpy, open-source), R6 (seeded), R8 (dev-only
script, not imported by run.py). Forbidden: no 2026 target days used for features
(the fold caches only contain facts <= their cutoff).

Usage:
  uv run python scripts/gap_analysis.py               # full: blend_retuned on B + G
  uv run python scripts/gap_analysis.py --fast --folds G   # light smoke (single LGBM)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import ab_validate as av  # noqa: E402

#: Production actor-feature fold cache (B/A/P) - matches actor_retuned submission.
CACHE_ACTOR = Path("data/folds/actor3")
#: Gift-proxy fold cache (G, cutoff 2025-02-14) - data-limited (T3.10.28 extends it).
CACHE_GIFT = Path("data/folds/gp")

DEFAULT_FOLDS = ("B", "G")


def _seg_summary(mask, err2, total_err2, bias, yt_log, pred_log, y_raw) -> dict:
    """Aggregate one user segment: counts, shares and log-space means."""
    n = int(mask.sum())
    return {
        "n": n,
        "share": float(n / len(mask)) if len(mask) else 0.0,
        "err2_share": float(err2[mask].sum() / total_err2) if total_err2 > 0 else 0.0,
        "bias_mean": float(bias[mask].mean()) if n else 0.0,
        "yt_log_mean": float(yt_log[mask].mean()) if n else 0.0,
        "pred_log_mean": float(pred_log[mask].mean()) if n else 0.0,
        "y_raw_mean": float(y_raw[mask].mean()) if n else 0.0,
    }


def build_segments(y_raw, yt_log, pred_log, whale_fracs=(0.01, 0.05)) -> dict:
    """Decompose a fold's log-space squared error into segments.

    Returns a dict with overall stats plus a ``segments`` map keyed by segment name.
    """
    y_raw = np.asarray(y_raw, dtype=np.float64)
    yt_log = np.asarray(yt_log, dtype=np.float64)
    pred_log = np.asarray(pred_log, dtype=np.float64)
    err2 = (pred_log - yt_log) ** 2
    bias = yt_log - pred_log
    total_err2 = float(err2.sum())
    is_zero = y_raw == 0

    out = {
        "n": int(len(y_raw)),
        "rmsle": float(np.sqrt(err2.mean())) if len(err2) else 0.0,
        "zero_frac": float(is_zero.mean()) if len(is_zero) else 0.0,
        "total_err2": total_err2,
    }
    segs = {
        "zeros": _seg_summary(is_zero, err2, total_err2, bias, yt_log, pred_log, y_raw),
        "positives": _seg_summary(~is_zero, err2, total_err2, bias, yt_log, pred_log, y_raw),
    }
    for w in whale_fracs:
        thr = float(np.quantile(y_raw, 1.0 - w)) if len(y_raw) else 0.0
        whale = y_raw >= thr
        key = f"top{int(round(w * 100)):02d}pct"
        segs[key] = _seg_summary(whale, err2, total_err2, bias, yt_log, pred_log, y_raw)
        segs[f"{key}_nonwhale"] = _seg_summary(
            ~whale, err2, total_err2, bias, yt_log, pred_log, y_raw
        )
        segs[f"{key}_zeros"] = _seg_summary(
            whale & is_zero, err2, total_err2, bias, yt_log, pred_log, y_raw
        )
        segs[f"{key}_positives"] = _seg_summary(
            whale & ~is_zero, err2, total_err2, bias, yt_log, pred_log, y_raw
        )
    out["segments"] = segs
    return out


def _print_fold(name: str, cache: Path, summary: dict, model_name: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"fold {name}  ({cache})  model={model_name}")
    print(
        f"n={summary['n']:,}  rmsle={summary['rmsle']:.6f}  "
        f"zero_frac={summary['zero_frac']:.4f}"
    )
    print(
        f"{'segment':<20} {'n':>9} {'share':>7} {'err2%':>7} {'bias':>9} "
        f"{'truth':>8} {'pred':>8} {'y_raw':>10}"
    )
    order = ["zeros", "positives", "top01pct", "top05pct"]
    for k in order:
        s = summary["segments"][k]
        print(
            f"{k:<20} {s['n']:>9,} {s['share']:>7.3f} {s['err2_share'] * 100:>7.2f} "
            f"{s['bias_mean']:>+9.4f} {s['yt_log_mean']:>8.3f} "
            f"{s['pred_log_mean']:>8.3f} {s['y_raw_mean']:>10.1f}"
        )


def _levers(fold_b: dict | None, fold_g: dict | None) -> list[str]:
    """Ranked lever list from the two decompositions (feeds E3 prioritization)."""
    if fold_b is None:
        return []
    levers: list[str] = []
    zb = fold_b["segments"]["zeros"]
    levers.append(
        f"zeros carry {zb['err2_share'] * 100:.1f}% of fold-B error^2 "
        f"(bias {zb['bias_mean']:+.3f}) -> log-space shrink / calibration"
    )
    w5 = fold_b["segments"]["top05pct"]
    levers.append(
        f"whales top-5% hold {w5['err2_share'] * 100:.1f}% of fold-B error^2 "
        f"(bias {w5['bias_mean']:+.3f}) -> whale handling (T3.10.19)"
    )
    if fold_g is not None:
        levers.append(
            f"gift-proxy fold G zero_frac={fold_g['zero_frac']:.4f} "
            f"(fold B={fold_b['zero_frac']:.4f}) -> season coverage (T3.10.28)"
        )
    return levers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-b-cache", type=Path, default=CACHE_ACTOR)
    ap.add_argument("--fold-g-cache", type=Path, default=CACHE_GIFT)
    ap.add_argument("--folds", default=",".join(DEFAULT_FOLDS), help="comma list, e.g. B,G")
    ap.add_argument("--whale-fracs", default="0.01,0.05")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="single LGBM instead of blend_retuned")
    ap.add_argument("--output", type=Path, default=None, help="JSON output path")
    args = ap.parse_args()

    os.environ.setdefault("ECUP26_NUM_THREADS", "4")
    whale_fracs = tuple(float(x) for x in args.whale_fracs.split(",") if x.strip())
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    model = av.m_baseline if args.fast else av.m_blend_retuned
    model_name = "baseline" if args.fast else "blend_retuned"

    results: dict[str, dict] = {}
    for name in folds:
        cache = args.fold_b_cache if name != "G" else args.fold_g_cache
        X, y, Xv, yt_log = av._load(name, cache)
        pred_log = np.asarray(model(X, y, Xv, seed=args.seed), dtype=np.float64)
        y_raw = np.clip(np.expm1(yt_log), 0.0, None)
        summary = build_segments(y_raw, yt_log, pred_log, whale_fracs)
        _print_fold(name, cache, summary, model_name)
        results[name] = summary

    print(f"\n{'=' * 78}\nRanked levers (E3 output):")
    for i, lev in enumerate(_levers(results.get("B"), results.get("G")), 1):
        print(f"  {i}. {lev}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nJSON saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
