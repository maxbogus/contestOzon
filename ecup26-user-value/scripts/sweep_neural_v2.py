#!/usr/bin/env python3
"""T3.19 v2 sweep: two-input neural model (sequence + static 102 features).

Config-driven grid over the top-3 leader configs x norm flag. Append-mode CSV
(resume-safe), saves models that beat --save-thresh via experiment_neural_wf_v2.py.
"""
from __future__ import annotations

import datetime
import re
import subprocess
import sys
from pathlib import Path

# (model, hidden, layers, lr, window, epochs, norm)
GRID = [
    ("gru", 56, 2, 0.0005, 409, 15, 0),
    ("gru", 56, 2, 0.0005, 409, 15, 1),
    ("gru", 80, 2, 0.0005, 409, 15, 0),
    ("gru", 80, 2, 0.0005, 409, 15, 1),
    ("gru", 96, 1, 0.0005, 409, 15, 0),
    ("gru", 96, 1, 0.0005, 409, 15, 1),
    ("gru", 104, 1, 0.0005, 409, 15, 0),
    ("gru", 104, 1, 0.0005, 409, 15, 1),
    ("lstm", 56, 2, 0.0005, 409, 15, 0),
    ("lstm", 56, 2, 0.0005, 409, 15, 1),
    ("lstm", 80, 2, 0.0005, 409, 15, 0),
    ("lstm", 80, 2, 0.0005, 409, 15, 1),
    ("lstm", 96, 1, 0.0005, 409, 15, 0),
    ("lstm", 96, 1, 0.0005, 409, 15, 1),
    ("lstm", 104, 1, 0.0005, 409, 15, 0),
    ("lstm", 104, 1, 0.0005, 409, 15, 1),
]

CSV_HEADER = "model,hidden,layers,lr,window,epochs,norm,best_rmsle,best_epoch,date,saved"


def _append_row(out: Path, row: str) -> None:
    with out.open("a") as f:
        f.write(row + "\n")


def load_done(out: Path) -> set:
    if not out.exists():
        return set()
    done = set()
    for line in out.read_text().splitlines()[1:]:
        p = line.split(",")
        if len(p) >= 7:
            done.add((p[0], int(p[1]), int(p[2]), float(p[3]), int(p[4]), int(p[5]), int(p[6])))
    return done


def main() -> int:
    out = Path("docs/results/neural_sweep_v2.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        _append_row(out, CSV_HEADER)
    done = load_done(out)
    todo = [c for c in GRID if c not in done]
    print(f"resume: {len(done)} done, {len(todo)} remaining", flush=True)
    for i, (model, h, lay, lr, w, ep, norm) in enumerate(todo, start=1):
        t0 = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{i}/{len(todo)}] {model} h{h} l{lay} lr{lr} w{w} norm{norm} ({t0})", flush=True)
        cmd = [sys.executable, "scripts/experiment_neural_wf_v2.py",
               "--model", model, "--hidden", str(h), "--layers", str(lay),
               "--lr", str(lr), "--window", str(w), "--epochs", str(ep),
               "--norm", str(norm), "--save-thresh", "1.80"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5 * 60 * 60)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},{norm},TIMEOUT,0,"
                        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')},0")
            continue
        m = re.search(r"BEST fold-B (\S+) @ ep (\d+)", r.stdout)
        saved = 1 if "SAVED" in r.stdout else 0
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if m:
            print(f"  -> BEST fold-B {m.group(1)} @ ep {m.group(2)} (saved={saved})", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},{norm},{m.group(1)},"
                        f"{m.group(2)},{date},{saved}")
        else:
            print("  -> FAILED", flush=True)
            _append_row(out, f"{model},{h},{lay},{lr},{w},{ep},{norm},FAIL,0,{date},0")
    print("sweep v2 done -> docs/results/neural_sweep_v2.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
