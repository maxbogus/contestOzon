#!/usr/bin/env bash
# E-CUP 2026 — unified download script for all 3 tasks.
# Usage:
#   ./download_data.sh --check-only    # exit 0 if all present
#   ./download_data.sh --task {1,2,3}  # download specific task
#   ./download_data.sh                 # download all

set -euo pipefail

DATA_DIR="${E-CUP_DATA_DIR:-/home/maxbogus/Repositories/contest/data}"
TASK1_DIR="$DATA_DIR/task1-matching"
TASK2_DIR="$DATA_DIR/task2-quality"
TASK3_DIR="$DATA_DIR/task3-user-value"

# =============================================================================
# URLs — update if competition data moves
# =============================================================================

TASK1_ITEMS_URL="https://odsai.s3.amazonaws.com/e-cup-2026/matching/items.parquet"
TASK1_MATCHES_URL="https://odsai.s3.amazonaws.com/e-cup-2026/matching/matches.parquet"
TASK1_ITEMS_HUMAN_URL="https://odsai.s3.amazonaws.com/e-cup-2026/matching/items_human.parquet"
TASK1_MATCHES_LLM_URL="https://odsai.s3.amazonaws.com/e-cup-2026/matching/matches_llm.parquet"

TASK2_DATA_URL="https://odsai.s3.amazonaws.com/e-cup-2026/quality/data.csv"
TASK2_IMAGES_URL="https://odsai.s3.amazonaws.com/e-cup-2026/quality/images.zip"

TASK3_DATA_URL="https://odsai.s3.amazonaws.com/e-cup-2026/user-value/train.csv"
TASK3_SAMPLE_URL="https://odsai.s3.amazonaws.com/e-cup-2026/user-value/sample_submission.csv"

# =============================================================================
# Helpers
# =============================================================================

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARNING: $*" >&2; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required: $1 (not installed)"; }

wget_resume() {
    local url="$1"; local out="$2"
    log "Downloading: $(basename "$out")"
    wget -c -q --show-progress -O "$out" "$url" || die "Failed: $url"
}

# =============================================================================
# Check mode
# =============================================================================

check_task1() {
    local missing=0
    for f in items.parquet matches.parquet items_human.parquet matches_llm.parquet; do
        local path="$TASK1_DIR/$f"
        if [[ -f "$path" && -s "$path" ]]; then
            log "  [OK] $f ($(du -sh "$path" | cut -f1))
"
        else
            log "  [MISSING] $f"
            ((missing++))
        fi
    done
    return $missing
}

check_task2() {
    local missing=0
    for f in data.csv images.zip; do
        local path="$TASK2_DIR/$f"
        if [[ -f "$path" && -s "$path" ]]; then
            log "  [OK] $f ($(du -sh "$path" | cut -f1))"
        else
            log "  [MISSING] $f"
            ((missing++))
        fi
    done
    return $missing
}

check_task3() {
    local missing=0
    for f in train.csv sample_submission.csv; do
        local path="$TASK3_DIR/$f"
        if [[ -f "$path" && -s "$path" ]]; then
            log "  [OK] $f ($(du -sh "$path" | cut -f1))"
        else
            log "  [MISSING] $f"
            ((missing++))
        fi
    done
    return $missing
}

# =============================================================================
# Download functions
# =============================================================================

download_task1() {
    log "=== Task 1: Matching ==="
    mkdir -p "$TASK1_DIR"
    need wget
    wget_resume "$TASK1_ITEMS_URL"    "$TASK1_DIR/items.parquet"
    wget_resume "$TASK1_MATCHES_URL"  "$TASK1_DIR/matches.parquet"
    wget_resume "$TASK1_ITEMS_HUMAN_URL"  "$TASK1_DIR/items_human.parquet"
    wget_resume "$TASK1_MATCHES_LLM_URL"  "$TASK1_DIR/matches_llm.parquet"
    log "Task 1: done"
}

download_task2() {
    log "=== Task 2: Quality ==="
    mkdir -p "$TASK2_DIR"
    need wget
    wget_resume "$TASK2_DATA_URL"   "$TASK2_DIR/data.csv"
    wget_resume "$TASK2_IMAGES_URL" "$TASK2_DIR/images.zip"
    log "Task 2: done"
}

download_task3() {
    log "=== Task 3: User Value ==="
    mkdir -p "$TASK3_DIR"
    need wget
    wget_resume "$TASK3_DATA_URL"   "$TASK3_DIR/train.csv"
    wget_resume "$TASK3_SAMPLE_URL" "$TASK3_DIR/sample_submission.csv"
    log "Task 3: done"
}

# =============================================================================
# Main
# =============================================================================

MODE="all"
TASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) MODE="check"; shift ;;
        --dry-run)    MODE="dry-run"; shift ;;
        --task)       TASK="$2"; MODE="one"; shift 2 ;;
        --dir)        DATA_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--check-only] [--dry-run] [--task {1,2,3}] [--dir PATH]"
            echo "  --check-only   check presence + size, exit 0 if all present"
            echo "  --dry-run      show URLs without downloading"
            echo "  --task N       download only task N (1, 2 or 3)"
            echo "  --dir PATH     data root (default: \$E-CUP_DATA_DIR or ./data)"
            exit 0
            ;;
        *) die "Unknown arg: $1" ;;
    esac
done

log "Data root: $DATA_DIR"

case "$MODE" in
    check)
        log "=== Checking ==="
        code=0
        log "Task 1:"; check_task1 || code=$?
        log "Task 2:"; check_task2 || code=$?
        log "Task 3:"; check_task3 || code=$?
        if [[ $code -eq 0 ]]; then
            log "All data present ✓"
        else
            warn "$code file(s) missing"
        fi
        exit $code
        ;;
    dry-run)
        echo "=== URLs ==="
        echo "Task 1:"
        echo "  $TASK1_ITEMS_URL"
        echo "  $TASK1_MATCHES_URL"
        echo "Task 2:"
        echo "  $TASK2_DATA_URL"
        echo "  $TASK2_IMAGES_URL"
        echo "Task 3:"
        echo "  $TASK3_DATA_URL"
        echo "  $TASK3_SAMPLE_URL"
        ;;
    one)
        case "$TASK" in
            1) download_task1 ;;
            2) download_task2 ;;
            3) download_task3 ;;
            *) die "--task must be 1, 2 or 3" ;;
        esac
        ;;
    all)
        download_task1
        download_task2
        download_task3
        log "=== All done ==="
        ;;
esac
