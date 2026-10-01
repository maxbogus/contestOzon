#!/usr/bin/env bash
# Coverage check for all 3 sub-repos
# Usage: bash scripts/coverage_check.sh
set -e

cd "$(dirname "$0")/.."

for repo in ecup26-matching ecup26-quality ecup26-user-value; do
    echo ""
    echo "=========================================="
    echo "  Coverage: $repo"
    echo "=========================================="
    cd "$repo"

    if [ ! -f pyproject.toml ]; then
        echo "  no pyproject.toml, skipping"
        cd ..
        continue
    fi

    if ! uv run python -c "import pytest_cov" 2>/dev/null; then
        echo "  Installing pytest-cov..."
        uv add --group dev pytest-cov 2>/dev/null || true
    fi

    uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=50 -q 2>&1 || echo "  FAIL: coverage < 50%"

    cd ..
done

echo ""
echo "=========================================="
echo "  Coverage check complete"
echo "=========================================="
