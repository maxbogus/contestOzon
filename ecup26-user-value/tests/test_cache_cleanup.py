"""Tests for scripts/cache_cleanup.py (cache policy v2)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cache_cleanup  # noqa: E402


def _touch(p: Path, size: int, mtime: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    import os
    os.utime(p, (mtime, mtime))


def test_dir_size_counts_files(tmp_path: Path) -> None:
    d = tmp_path / "folds" / "giftG"
    _touch(d / "a.parquet", 100, 1.0)
    _touch(d / "sub" / "b.parquet", 50, 1.0)
    assert cache_cleanup._dir_size(tmp_path / "folds") == 150


def test_hash_of_extracts_content_hash() -> None:
    assert cache_cleanup._hash_of("train_43822300b8b46497e543526ff6d1e0f1.parquet") == "43822300b8b46497e543526ff6d1e0f1"
    assert cache_cleanup._hash_of("model_x.txt") == "?"


def test_resume_groups_groups_by_hash(tmp_path: Path, monkeypatch) -> None:
    h = "43822300b8b46497e543526ff6d1e0f1"
    cache_cleanup.RESUME = tmp_path
    cache_cleanup.TOP_FILE = tmp_path / "top_candidates.json"
    _touch(tmp_path / f"train_{h}.parquet", 10, 1.0)
    _touch(tmp_path / f"model_{h}.txt", 10, 1.0)
    _touch(tmp_path / "model_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.txt", 10, 2.0)
    groups = cache_cleanup._resume_groups()
    assert set(groups) == {h, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}


def test_eviction_keeps_protected(tmp_path: Path, monkeypatch) -> None:
    cache_cleanup.FOLDS = tmp_path / "folds"
    cache_cleanup.RESUME = tmp_path / ".cache"
    cache_cleanup.TOP_FILE = cache_cleanup.RESUME / "top_candidates.json"
    cache_cleanup.RESUME.mkdir(parents=True, exist_ok=True)
    cache_cleanup.TOP_FILE.write_text(json.dumps({"hashes": [], "folds": ["giftG"]}))

    _touch(cache_cleanup.FOLDS / "giftG" / "x.parquet", 10 * 2**20, 1.0)
    _touch(cache_cleanup.FOLDS / "old" / "y.parquet", 10 * 2**20, 1.0)

    monkeypatch.setattr(sys, "argv", ["cache_cleanup", "--limit-gb", "0.01"])
    rc = cache_cleanup.main()
    assert rc == 0
    assert (cache_cleanup.FOLDS / "giftG").exists(), "protected fold must survive"
    assert not (cache_cleanup.FOLDS / "old").exists(), "unprotected fold must be evicted"
