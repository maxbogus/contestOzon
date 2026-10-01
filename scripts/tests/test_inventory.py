#!/usr/bin/env python3
"""Tests for inventory.py — LLM-optimized backlog profiling.

TDD: RED first. These tests define the expected output format.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parents[1]))

SCRIPT = Path(__file__).parents[1] / "inventory.py"


def run_inventory(args=None) -> tuple[int, str, str]:
    """Run inventory.py and return (returncode, stdout, stderr)."""
    cmd = ["python3", str(SCRIPT)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


class TestInventoryRuns(unittest.TestCase):
    def test_runs_without_args(self):
        """inventory.py should run successfully with no args."""
        rc, out, err = run_inventory()
        assert rc == 0, f"Expected exit 0, got {rc}. stderr: {err[:200]}"

    def test_uses_markdown_headers(self):
        """Output should contain LLM-friendly markdown headings."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Must have major section headers
        assert "## SUMMARY" in out or "## Backlog" in out.upper(), (
            f"Missing SUMMARY header. Output:\n{out[:500]}"
        )

    def test_uses_tables(self):
        """Output should contain markdown tables for structured data."""
        rc, out, err = run_inventory()
        assert rc == 0
        assert "|" in out, "Output should contain markdown tables"


class TestInventoryContent(unittest.TestCase):
    def test_shows_total_count(self):
        """Must show total ticket count."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Look for total count
        assert any(word in out for word in ["total", "Total", "TOTAL", "41"]), (
            f"Missing total count. Output:\n{out[:300]}"
        )

    def test_shows_status_breakdown(self):
        """Must show tickets by status (done, backlog, in_progress)."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Should mention statuses
        status_words = ["backlog", "done", "in_progress"]
        found = [w for w in status_words if w in out]
        assert len(found) >= 2, (
            f"Expected at least 2 status types. Found: {found}. Output:\n{out[:500]}"
        )

    def test_shows_by_repo(self):
        """Must show ticket counts per repo."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Should mention repo names
        repos = ["matching", "quality", "user-value", "shared"]
        found = [r for r in repos if r in out]
        assert len(found) >= 2, (
            f"Expected at least 2 repos. Found: {found}. Output:\n{out[:500]}"
        )

    def test_shows_top_ready(self):
        """Must show ready-to-implement tickets (sorted by RICE)."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Should show ticket IDs
        import re
        ticket_ids = re.findall(r"T\d+\.\d+\.\d+", out)
        assert len(ticket_ids) >= 1, (
            f"Expected at least 1 ticket ID in output. Output:\n{out[:500]}"
        )


class TestInventoryJsonBlock(unittest.TestCase):
    def test_has_json_summary_block(self):
        """Must include a JSON block with machine-readable summary."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Look for fenced code block with json
        assert "```json" in out or '"total":' in out or '"backlog":' in out, (
            f"Missing JSON block. Output:\n{out[:500]}"
        )


class TestInventoryCriticalPath(unittest.TestCase):
    def test_shows_critical_path(self):
        """Should show the critical path (longest P0 chain)."""
        rc, out, err = run_inventory()
        assert rc == 0
        # Should mention critical path or blocked
        critical_keywords = ["critical", "Critical", "CRITICAL", "path", "blocked"]
        found = any(kw in out for kw in critical_keywords)
        assert found, f"Missing critical path info. Output:\n{out[:500]}"


if __name__ == "__main__":
    unittest.main()
