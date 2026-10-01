"""Tests for .gitignore hygiene (M-009).

The .gitignore used to have `models/` which matched BOTH:
- /models/ (artifact directory at repo root, intended to ignore)
- src/ecup26_user_value/models/ (Python source code, NOT intended)

Fixed in M-009 to `/models/` (anchored to root). This test guards against
regression to the unanchored pattern.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_check_ignore(rel_path: str) -> bool:
    """Return True if git would ignore rel_path."""
    result = subprocess.run(
        ["git", "check-ignore", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # git check-ignore exit codes: 0 = ignored, 1 = not ignored, 128 = error
    return result.returncode == 0


class TestGitignoreSourceModelsNotIgnored:
    """Python source under src/.../models/ MUST be tracked (R1 / R6 hygiene)."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "src/ecup26_user_value/models/btyd_baseline.py",
            "src/ecup26_user_value/models/btyd_segmented.py",
            "src/ecup26_user_value/models/ziln.py",
            "src/ecup26_user_value/ablation/model_wrappers.py",
        ],
    )
    def test_source_path_not_ignored(self, rel_path: str) -> None:
        assert not _git_check_ignore(rel_path), (
            f"{rel_path} should NOT be ignored by .gitignore (M-009 fix: models/ -> /models/)"
        )


class TestGitignoreArtifactModelsIgnored:
    """The /models/ artifact directory at repo root MUST still be ignored (R8)."""

    def test_root_models_dir_ignored(self) -> None:
        # Pattern is "/models/" (anchored, dir-only); files inside it ignored.
        assert _git_check_ignore("models/"), "/models/ dir should be ignored"
        assert _git_check_ignore("models/some_artifact.bin"), (
            "Files under /models/ should be ignored"
        )


class TestGitignorePattern:
    """Guard the literal pattern: it MUST be anchored (start with /)."""

    def test_models_pattern_is_anchored(self) -> None:
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        # Find the line declaring models/ (allow leading whitespace, comments before)
        for line in gitignore.splitlines():
            stripped = line.strip()
            if stripped.startswith("models/") or stripped == "models/":
                pytest.fail(
                    f"Found unanchored 'models/' pattern in .gitignore line: {line!r}. "
                    f"M-009 fix should make it '/models/'."
                )
