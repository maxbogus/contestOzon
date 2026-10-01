"""Dev-resume cache: content-addressed feature matrix + LightGBM model.

Per .clinerules/02-ecup-rules.md:
- R4: local disk only (never network)
- R6: deterministic; the cache key covers the train-data signature AND the
      feature/model config versions, so stale artifacts are never reused
- R8: in the eval Docker the cache dir is fresh/empty -> cache miss -> the
      full pipeline runs once (correct); local iteration re-runs skip the
      expensive feature build and LightGBM fit

API:
    feature_cache_key(train_df, rolling_windows, model_config) -> str
    load_cached_features(cache_dir, key) -> pd.DataFrame | None
    save_cached_features(cache_dir, key, features) -> Path
    load_cached_model(cache_dir, key) -> Any | None
    save_cached_model(cache_dir, key, booster) -> Path
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Bump when build_features output schema/semantics change (invalidates all caches).
FEATURE_CACHE_VERSION = "v4"
#: Bump when the LightGBM training logic changes (target, params, transforms).
MODEL_CACHE_VERSION = "v5"

_DEFAULT_USER_ID_COL = "user_id"
_DEFAULT_DATE_COL = "event_date"
_DEFAULT_PRICE_COL = "price"


def _content_signature(train_df: pd.DataFrame) -> str:
    """Cheap, robust signature of the training frame content.

    Uses aggregate statistics instead of hashing 30M rows: fast enough to
    compute on every run, and any real change to the data flips at least one
    field (row count, date range, user count, total price).
    """
    return "|".join(
        [
            str(len(train_df)),
            str(train_df[_DEFAULT_DATE_COL].max()),
            str(train_df[_DEFAULT_DATE_COL].min()),
            str(train_df[_DEFAULT_USER_ID_COL].nunique()),
            repr(float(train_df[_DEFAULT_PRICE_COL].sum())),
        ]
    )


def feature_cache_key(
    train_df: pd.DataFrame,
    rolling_windows: tuple[int, ...] = (7, 14, 30),
    model_config: dict[str, Any] | None = None,
) -> str:
    """Content-addressed cache key for the feature matrix + fitted model.

    Covers: data signature, feature schema version, rolling windows, and the
    full model config (params + num boost rounds). Any of them changing
    produces a different key, so a cache hit is only possible when both the
    data and the pipeline config are unchanged.
    """
    h = hashlib.sha256()
    h.update(FEATURE_CACHE_VERSION.encode("utf-8"))
    h.update(MODEL_CACHE_VERSION.encode("utf-8"))
    h.update(_content_signature(train_df).encode("utf-8"))
    h.update(repr(rolling_windows).encode("utf-8"))
    if model_config is not None:
        h.update(repr(sorted(model_config.items(), key=lambda kv: kv[0])).encode("utf-8"))
    return h.hexdigest()[:32]


def _features_path(cache_dir: Path, key: str, kind: str) -> Path:
    return cache_dir / f"features_{kind}_{key}.parquet"


def _model_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"model_{key}.txt"


def load_cached_features(cache_dir: Path, key: str, kind: str = "train") -> pd.DataFrame | None:
    """Load the cached per-user feature matrix, or None on miss/corruption.

    Args:
        cache_dir: cache directory.
        key: content-addressed cache key.
        kind: "train" (features built at feature_cutoff, aligned with the
            30-day label) or "infer" (features built at ref_date, freshest
            signal used for scoring).
    """
    path = _features_path(cache_dir, key, kind)
    if not path.exists():
        return None
    try:
        features = pd.read_parquet(path)
        if _DEFAULT_USER_ID_COL not in features.columns:
            logger.warning("Cache file %s missing user_id column; ignoring", path)
            return None
        return features
    except Exception as exc:  # noqa: BLE001 - a broken cache must never break the run
        logger.warning("Failed to load feature cache %s: %s", path, exc)
        return None


def save_cached_features(
    cache_dir: Path, key: str, features: pd.DataFrame, kind: str = "train"
) -> Path:
    """Persist the feature matrix for later resume runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _features_path(cache_dir, key, kind)
    features.to_parquet(path, index=False)
    logger.info("Saved %d x %d feature matrix to %s", features.shape[0], features.shape[1], path)
    return path


def load_cached_model(cache_dir: Path, key: str) -> Any | None:
    """Load the cached LightGBM Booster, or None on miss/corruption."""
    path = _model_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load model cache %s: %s", path, exc)
        return None


def save_cached_model(cache_dir: Path, key: str, booster: Any) -> Path:
    """Persist the fitted LightGBM Booster (all boosting rounds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _model_path(cache_dir, key)
    booster.save_model(str(path))
    logger.info("Saved LightGBM model to %s", path)
    return path


def _train_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"train_{key}.parquet"


def load_cached_train(cache_dir: Path, key: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load the cached walk-forward training set (features + label).

    The label is stored as the column "y_log" inside the parquet so the
    feature rows and labels stay positionally aligned across origins.

    Returns:
        (X, y) where X has a user_id column and y is aligned to X rows,
        or None on miss/corruption.
    """
    path = _train_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        X = pd.read_parquet(path)
        if "y_log" not in X.columns:
            logger.warning("Cache file %s missing y_log column; ignoring", path)
            return None
        y = X.pop("y_log").astype(float)
        return X, y
    except Exception as exc:  # noqa: BLE001 - a broken cache must never break the run
        logger.warning("Failed to load train cache %s: %s", path, exc)
        return None


def save_cached_train(cache_dir: Path, key: str, X: pd.DataFrame, y: pd.Series) -> Path:
    """Persist the walk-forward training set for later resume runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _train_path(cache_dir, key)
    out = X.copy()
    out["y_log"] = np.asarray(y, dtype=np.float64)
    out.to_parquet(path, index=False)
    logger.info("Saved %d-row walk-forward train set to %s", len(out), path)
    return path


def _bias_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"bias_{key}.json"


def save_cached_bias(cache_dir: Path, key: str, bias: float) -> Path:
    """Persist the log-space calibration bias (T3.8.6) for resume runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _bias_path(cache_dir, key)
    path.write_text(json.dumps({"calibration_bias": float(bias)}))
    logger.info("Saved calibration bias %s to %s", bias, path)
    return path


def load_cached_bias(cache_dir: Path, key: str) -> float | None:
    """Load the cached calibration bias, or None on miss/corruption."""
    path = _bias_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["calibration_bias"])
    except Exception:  # noqa: BLE001
        return None
