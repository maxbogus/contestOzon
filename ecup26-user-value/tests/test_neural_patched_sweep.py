"""RED tests for T3.19 patched sweep (early stop / resume / NEW BEST parse).

Per .clinerules: R6 (seeded/deterministic), R10-adjacent (CSV integrity).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiment_neural_wf import early_stop_after  # noqa: E402
from sweep_neural_patched import load_done  # noqa: E402

HEADER = "model,hidden,layers,lr,window,patch,epochs,best_rmsle,best_epoch,date,saved"
BEST_RE = re.compile(r"BEST fold-B (\S+) @ ep (\d+)")


class TestEarlyStop:
    def test_disabled(self) -> None:
        assert early_stop_after(10, 4, 0) is False

    def test_not_yet(self) -> None:
        assert early_stop_after(8, 4, 5) is False

    def test_trigger(self) -> None:
        assert early_stop_after(9, 4, 5) is True

    def test_at_best_epoch(self) -> None:
        assert early_stop_after(4, 4, 5) is False

    def test_patience_one(self) -> None:
        assert early_stop_after(5, 4, 1) is True


class TestLoadDone:
    def test_missing_file(self, tmp_path) -> None:
        done, best = load_done(tmp_path / "nope.csv")
        assert done == set()
        assert best == float("inf")

    def test_parses_keys_with_patch(self, tmp_path) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text(
            HEADER + "\n"
            "patched,128,2,0.0003,409,7,15,1.6758,4,2026-08-29 20:34,1\n"
            "patched,128,2,0.0002,409,7,20,TIMEOUT,0,2026-08-29 21:00,0\n"
        )
        done, best = load_done(csv)
        assert ("patched", 128, 2, 0.0003, 409, 7) in done
        assert ("patched", 128, 2, 0.0002, 409, 7) in done
        assert best == 1.6758

    def test_timeout_row_counts_done_but_not_best(self, tmp_path) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text(HEADER + "\npatched,128,2,0.0002,409,7,20,TIMEOUT,0,2026-08-29 21:00,0\n")
        done, best = load_done(csv)
        assert len(done) == 1
        assert best == float("inf")

    def test_skips_garbage_rows(self, tmp_path) -> None:
        csv = tmp_path / "s.csv"
        csv.write_text(HEADER + "\nnot-a-row\npatched,128,2,0.0003,409,7,15,1.6758,4,d,1\n")
        done, _ = load_done(csv)
        assert len(done) == 1


class TestParseBest:
    def test_parse_result_line(self) -> None:
        m = BEST_RE.search("... BEST fold-B 1.6758 @ ep 4 ...")
        assert m is not None
        assert float(m.group(1)) == 1.6758
        assert int(m.group(2)) == 4
