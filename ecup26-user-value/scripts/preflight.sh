#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== ruff ==="
uv run ruff check src/ tests/ || { echo "ruff FAILED"; exit 1; }
echo "=== mypy ==="
uv run mypy src/ || { echo "mypy FAILED"; exit 1; }
echo "=== pytest ==="
uv run pytest tests/ -v || { echo "pytest FAILED"; exit 1; }
echo "=== All checks passed ==="
