#!/usr/bin/env python3
"""T3.20: sweep for Mamba / RWKV / xLSTM sequence encoders (resume-safe).

docs/results/neural_ssm_sweep.csv is the source of truth: rows APPENDED, never
truncated; re-run skips done configs. NEW BEST is printed when a config beats
the best fold-B seen so far; the model itself is saved immediately by
experiment_neural_wf.py (--save-thresh >= current best).

Default grid = the current-best neural options (h128 l2 lr0.0005 w409, epochs 20,
patience 5) for each of the three T3.20 models. Follow-up passes narrow around
the winner with --model-list / --lr-list / --hidden-list.

Usage:
  uv run python scripts/sweep_neural_ssm.py                     # 3 models x best cfg
  uv run python scripts/sweep_neural_ssm.py \\
      --model-list mamba --lr-list 0.0003,0.0008 --hidden-list 112,128,144
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

CSV = "docs/results/neural_ssm_sweep.csv"
HEADER = "model,hidden,layers,lr,window,epochs,best_rmsle,best_epoch,date,saved"


def _append_row(out: Path, row: str) -> None:
    with open(out, "a") as fh:
        fh.write(row + "\n")


def load_done(out: Path) -> tuple[set[tuple], float]:
    """Return (done_keys, best_so_far). Keys: (model, hidden, layers, lr, window)."""
    done: set[tuple] = set()
    best = float("inf")
    if not out.exists():
        return done, best
    for line in out.read_text().splitlines()[1:]:
        p = line.strip().split(",")
        if len(p) < 8 or p[0] == "model":
            continue
        try:
            key = (p[0], int(p[1]), int(p[2]), float(p[3]), int(p[4]))
        except ValueError:
            continue
        done.add(key)
        try:
            r = float(p[6])  # best_rmsle column (0=model..6=best_rmsle,7=best_epoch)
            if r < best:
                best = r
        except ValueError:
            pass
    return done, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-list", type=str, default="mamba,rwkv,xlstm")
    ap.add_argument("--lr-list", type=str, default="0.0005")
    ap.add_argument("--hidden-list", type=str, default="128")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=409)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--save-thresh", type=float, default=1.68)
    ap.add_argument("--timeout", type=int, default=3 * 60 * 60)
    args = ap.parse_args()

    models = [x for x in args.model_list.split(",") if x.strip()]
    lrs = [float(x) for x in args.lr_list.split(",") if x.strip()]
    hids = [int(x) for x in args.hidden_list.split(",") if x.strip()]
    out = Path(CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        _append_row(out, HEADER)

    done, best_so_far = load_done(out)
    grid = [(m, h, args.layers, lr, args.window) for m in models for h in hids for lr in lrs]
    todo = [c for c in grid if c not in done]
    print(f"resume: {len(done)} done (best {best_so_far:.4f}), {len(todo)} remaining", flush=True)

    for i, (model, h, lay, lr, w) in enumerate(todo, start=1):
        t0 = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{i}/{len(todo)}] {model} h{h} l{lay} lr{lr} w{w} ({t0})", flush=True)
        cmd = [
            sys.executable,
            "scripts/experiment_neural_wf.py",
            "--model",
            model,
            "--hidden",
            str(h),
            "--layers",
            str(lay),
            "--lr",
            str(lr),
            "--window",
            str(w),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--save-thresh",
            str(args.save_thresh),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            _append_row(
                out,
                f"{model},{h},{lay},{lr},{w},{args.epochs},TIMEOUT,0,"
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')},0",
            )
            continue
        m = re.search(r"BEST fold-B (\S+) @ ep (\d+)", r.stdout)
        saved = 1 if "SAVED" in r.stdout else 0
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if m:
            rmsle = float(m.group(1))
            new_best = rmsle < best_so_far
            if new_best:
                best_so_far = rmsle
            tag = "  *** NEW BEST ***" if new_best else ""
            print(
                f"  -> BEST fold-B {rmsle:.4f} @ ep {m.group(2)} (saved={saved}){tag}",
                flush=True,
            )
            _append_row(
                out,
                f"{model},{h},{lay},{lr},{w},{args.epochs},{rmsle},{m.group(2)},{date},{saved}",
            )
        else:
            print("  -> FAILED", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{args.epochs},FAIL,0,{date},0")
    print(f"sweep done -> {CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
