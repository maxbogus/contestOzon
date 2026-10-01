#!/usr/bin/env python3
"""E-CUP 2026 Task 3 (user-value) — submission entry point.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lifetimes + scikit-learn, Apache 2.0/BSD)
- R4: no internet at runtime (all deps installed at build time)
- R6: reproducible (seeded BTYD via lifetimes defaults)
- R7: CPU tabular models only (no LLM, no vLLM)
- R8: entry point `python -u run.py --test_data_path <path> --output_path <path>`
- R10: assert `df.iloc[:, 0].is_unique` before writing submit.csv

Reads:
  --test_data_path  CSV with column `user_id` listing users to predict for
                    (in dry-run mode: data/sample_submit.csv)

Trains BTYDBaseline on train.parquet in repo root (same dir as run.py) when
present (development mode). Production eval-time training data is mounted by
the evaluator under a path passed via --test_data_path's parent or via env
var ECUP26_TRAIN_PATH.

Writes:
  --output_path  CSV with columns `user_id,predict` (one row per user)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# R8: make run.py self-contained as a submission entry point.
# Add repo's src/ to sys.path so `import ecup26_user_value.run_pipeline`
# works regardless of how the evaluator invokes this script
# (`python -u run.py ...` from /app, no PYTHONPATH set).
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402 (imports follow the R8 sys.path bootstrap above)
import pandas as pd  # noqa: E402

# R8: early env check. parquet support requires pyarrow or fastparquet.
# Without this, pd.read_parquet fails with a confusing ImportError.
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET_ENGINE = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET_ENGINE = True
    except ImportError:
        _HAS_PARQUET_ENGINE = False

if not _HAS_PARQUET_ENGINE:
    sys.stderr.write(
        "ERROR: run.py requires pyarrow or fastparquet to read train.parquet.\n"
        "       Run via `uv run python -u run.py ...` (uses repo venv with pyarrow),\n"
        "       or install pyarrow in your Python environment: pip install pyarrow\n"
    )
    sys.exit(2)

from ecup26_user_value.run_pipeline import predict_for_users  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_train_data(test_data_path: Path) -> Path | None:
    """Locate train.parquet for model fitting.

    Order:
    1. ECUP26_TRAIN_PATH env var (if set, evaluator mounts train there)
    2. <test_data_path>/../train.parquet (same parent dir)
    3. <repo_root>/data/train.parquet (dev mode)
    """
    env_path = os.environ.get("ECUP26_TRAIN_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    sibling = test_data_path.parent / "train.parquet"
    if sibling.exists():
        return sibling
    dev = SCRIPT_DIR / "data" / "train.parquet"
    if dev.exists():
        return dev
    return None


def _load_transactions(train_path: Path) -> pd.DataFrame:
    """Load train parquet and adapt to BTYDBaseline schema.

    Required columns after mapping: user_id, event_date, type, price.
    Source columns per docs/tasks/task3_user_value_data.md:
      event_date, user_id, search, cat, ..., gmv_search, gmv_cat, gmv
    """
    # Lazy check: defense in depth (early check at module load catches 99% of cases)
    if train_path.suffix == ".parquet" and not _HAS_PARQUET_ENGINE:
        raise RuntimeError(
            "parquet engine (pyarrow or fastparquet) not available. "
            "Run via `uv run python -u run.py ...`."
        )
    df = pd.read_parquet(train_path) if train_path.suffix == ".parquet" \
        else pd.read_csv(train_path)
    if "type" not in df.columns and "search" in df.columns and "cat" in df.columns:  # noqa: SIM102
        df["type"] = np.where(df["search"] > 0, "search", "cat")
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"].astype(float)
    # Rich feature engineering needs the activity columns, but the full
    # 18-column frame (all float64/int64) blows up RAM (~25+ GB) during groupby
    # feature building on 30M rows. Keep only the columns used and downcast.
    required = ["user_id", "event_date", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"train.parquet missing required columns {missing}; have {df.columns.tolist()}"
        )
    _keep = [
        "user_id", "event_date", "price", "gmv", "search", "cat",
        "searches", "to_cart", "to_ord", "search_to_cart", "search_to_ord",
        "cat_to_cart", "cat_to_ord", "gmv_search", "gmv_cat",
    ]
    df = df[[c for c in _keep if c in df.columns]]
    for c in ("price", "gmv", "gmv_search", "gmv_cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    for c in ("searches", "to_cart", "to_ord", "search_to_cart",
              "search_to_ord", "cat_to_cart", "cat_to_ord"):
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    for c in ("search", "cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.int8)
    return df


def _subsample_train_df(train_df: pd.DataFrame, train_frac: float) -> pd.DataFrame:
    """Optionally subsample train users for faster iteration in tests.

    Args:
        train_df: full train DataFrame (after _load_transactions).
        train_frac: fraction in (0, 1]. 1.0 = no subsampling.
    Returns:
        Subsampled DataFrame (deterministic, seed=42).
    """
    if train_frac >= 1.0:
        return train_df
    rng = np.random.default_rng(42)
    unique_users = train_df["user_id"].unique()
    n_keep = int(len(unique_users) * train_frac)
    sample_users = set(rng.choice(unique_users, size=n_keep, replace=False))
    return train_df[train_df["user_id"].isin(sample_users)].copy()


def _fit_predict_pipeline(
    train_df: pd.DataFrame,
    user_ids: pd.Series,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Run T3.7.6 production pipeline: features + LightGBM.

    Args:
        train_df: training transactions (user_id, event_date, type, price).
        user_ids: users to predict for.
        cache_dir: dev-resume cache dir (None disables caching).
        use_cache: False forces a full recompute even with cache_dir.

    Returns:
        np.ndarray of per-user predictions aligned to user_ids.
    """
    return predict_for_users(
        train_df, user_ids.values,
        cache_dir=cache_dir, use_cache=use_cache,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E-CUP 2026 Task 3: User Value Prediction"
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True,
        help="CSV with column user_id (users to predict for)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output CSV path (columns: user_id,predict)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=None,
        help="Optional: use only this fraction of train users (0, 1]. "
             "For testing speed. Default: 1.0 (full dataset). "
             "Override with ECUP26_TRAIN_FRAC env var.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the dev-resume cache (feature matrix + LightGBM model). "
             "Default: cache enabled at $ECUP26_CACHE_DIR or <repo>/data/.cache.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override the resume-cache directory. "
             "Default: $ECUP26_CACHE_DIR or <repo>/data/.cache.",
    )
    args = parser.parse_args()

    default_cache_dir = os.environ.get("ECUP26_CACHE_DIR", str(SCRIPT_DIR / "data" / ".cache"))
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(default_cache_dir)
    use_cache = not args.no_cache
    if use_cache:
        logger.info(
            "Resume cache enabled (dir=%s); pass --no-cache to disable", cache_dir
        )
    else:
        logger.info("Resume cache disabled (--no-cache)")

    test_path = Path(args.test_data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        logger.error("test_data_path not found: %s", test_path)
        return 2

    logger.info("Loading test users from %s", test_path)
    test_df = pd.read_csv(test_path)
    if "user_id" not in test_df.columns:
        logger.error("test_data_path missing 'user_id' column; have %s", test_df.columns.tolist())
        return 2
    user_ids = test_df["user_id"].drop_duplicates().sort_values().reset_index(drop=True)
    logger.info("Predicting for %d unique users", len(user_ids))

    train_path = _find_train_data(test_path)
    if train_path is None:
        logger.warning("No train.parquet found; emitting zero predictions")
        preds = pd.Series(0.0, index=user_ids)
    else:
        train_df = _load_transactions(train_path)
        # Get train_frac: CLI > env > default 1.0
        if args.train_frac is None:
            train_frac = float(os.environ.get("ECUP26_TRAIN_FRAC", "1.0"))
        else:
            train_frac = args.train_frac
        if train_frac < 1.0:
            train_df = _subsample_train_df(train_df, train_frac)
            logger.info(
                "Subsampled to %d users (%.1f%% of full)",
                train_df["user_id"].nunique(), train_frac * 100,
            )
        preds = _fit_predict_pipeline(
            train_df, user_ids,
            cache_dir=cache_dir, use_cache=use_cache,
        )
        # Wrap as Series for consistent downstream handling
        preds = pd.Series(preds, index=user_ids).clip(lower=0.0)

    # R10: assert unique IDs before write
    assert user_ids.is_unique, (
        f"user_ids from test_data_path have duplicates: "
        f"{user_ids[user_ids.duplicated()].head().tolist()}"
    )

    out_df = pd.DataFrame({"user_id": user_ids, "predict": preds.values})
    assert out_df.iloc[:, 0].is_unique, "R10: duplicate user_ids in output"

    logger.info("Writing %d rows to %s", len(out_df), output_path)
    out_df.to_csv(output_path, index=False)

    # Sanity log
    rmsle_naive = float(np.sqrt(np.mean(np.log1p(out_df["predict"].values) ** 2)))
    logger.info("Sanity: mean predict=%.2f, naive-RMSLE-against-zero=%.4f",
                out_df["predict"].mean(), rmsle_naive)
    return 0


if __name__ == "__main__":
    import os  # noqa: E402 (used inside _find_train_data)
    sys.exit(main())
