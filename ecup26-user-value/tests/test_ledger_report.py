"""Tests for scripts/ledger_report.py (decision-ledger v2 reader)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ledger_report  # noqa: E402


def _row(
    rid: str,
    status: str = "TODO",
    conditions: str = "",
    parents: list[str] | None = None,
) -> dict:
    return {
        "id": rid,
        "type": "atomic_check",
        "parents": parents or ["T"],
        "status": status,
        "hypothesis": "h",
        "verdict_conditions": conditions,
        "revisit_trigger": "trigger",
        "command": "cmd",
        "gate_pre": "WIN",
    }


def test_load_rows_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps(_row("A1")) + "\n\n" + json.dumps(_row("A2")) + "\n")
    rows = ledger_report.load_rows(p)
    assert len(rows) == 2


def test_reopen_candidates_flags_conditions_now_available() -> None:
    rows = [
        _row("A1", status="FALSIFIED", conditions="fold G не был доступен"),
        _row("A2", status="FALSIFIED", conditions="только один фолд"),
        _row("A3", status="CONFIRMED", conditions="actor фичи"),
        _row("A4", status="TIE", conditions="сегментная коррекция"),
        _row("A5", status="TODO", conditions="fold G"),
    ]
    hits = ledger_report.reopen_candidates(rows)
    assert {r["id"] for r in hits} == {"A1", "A3", "A4"}


def test_todo_checks_excluded_from_reopen() -> None:
    rows = [_row("A1", status="TODO", conditions="fold G")]
    assert ledger_report.reopen_candidates(rows) == []


def test_ledger_file_parses_and_has_expected_schema() -> None:
    ledger = REPO_ROOT / "docs/research/decision_ledger_v2.jsonl"
    rows = ledger_report.load_rows(ledger)
    assert len(rows) >= 30
    required = {
        "id", "type", "parents", "status", "hypothesis",
        "command", "metric", "gate_pre", "verdict_conditions",
        "revisit_trigger",
    }
    for row in rows:
        assert required <= set(row), row["id"]
        assert row["type"] == "atomic_check"
        assert row["status"] in {"TODO", "RUNNING", "CONFIRMED", "FALSIFIED", "TIE", "INVALID", "PARTIAL", "OBSOLETE"}
        assert row["gate_pre"], row["id"]
