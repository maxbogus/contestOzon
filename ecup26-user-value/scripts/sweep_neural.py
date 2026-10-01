#!/usr/bin/env python3
"""T3.19: parameter sweep (RESUME + PROGRESS + APPEND-safe).
docs/results/neural_sweep.csv is the source of truth: re-run skips done
configs. Rows are APPENDED (never truncated) - interrupt-safe.
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

GRID = [
    ("gru", 128, 1, 0.0008, 409, 15),
    ("gru", 124, 2, 0.0008, 409, 15),
    ("gru", 132, 2, 0.0008, 409, 15),
    ("gru", 128, 3, 0.0008, 409, 15),
]


def _append_row(out: Path, row: str) -> None:
    with open(out, "a") as fh:
        fh.write(row + chr(10))


def load_done(out: Path) -> set:
    done = set()
    if not out.exists():
        return done
    for line in out.read_text().splitlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 6 and p[0] != "model":
            try:
                done.add((p[0], int(p[1]), int(p[2]), float(p[3]), int(p[4])))
            except ValueError:
                pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    grid = GRID[:4] if args.quick else GRID
    out = Path("docs/results/neural_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        _append_row(out, "model,hidden,layers,lr,window,epochs,best_rmsle,best_epoch,date,saved")
    done = load_done(out)
    todo = [c for c in grid if (c[0], c[1], c[2], c[3], c[4]) not in done]
    print(f"resume: {len(done)} done, {len(todo)} remaining", flush=True)
    for i, (model, h, lay, lr, w, ep) in enumerate(todo, start=1):
        t0 = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{i}/{len(todo)}] {model} h{h} l{lay} lr{lr} w{w} ({t0})", flush=True)
        batch = 256 if h >= 256 else 512
        cmd = [sys.executable, "scripts/experiment_neural_wf.py",
               "--model", model, "--hidden", str(h), "--layers", str(lay),
               "--lr", str(lr), "--window", str(w), "--epochs", str(ep),
               "--batch", str(batch), "--save-thresh", "1.80"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8*60*60)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},TIMEOUT,0,{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},0")
            continue
        m = re.search(r"BEST fold-B (\S+) @ ep (\d+)", r.stdout)
        saved = 1 if "SAVED" in r.stdout else 0
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if m:
            print(f"  -> BEST fold-B {m.group(1)} @ ep {m.group(2)} (saved={saved})", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},{m.group(1)},{m.group(2)},{date},{saved}")
        else:
            print("  -> FAILED", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},FAIL,0,{date},0")
    print("sweep done -> docs/results/neural_sweep.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
