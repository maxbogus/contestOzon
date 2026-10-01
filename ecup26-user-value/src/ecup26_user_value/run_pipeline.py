"""Production pipeline for T3.7.6 (Cycle #3) - features + LightGBM.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lightgbm, scikit-learn, lifetimes)
- R4: no internet at runtime
- R6: reproducible (seeded)
- R7: CPU tabular
- R8: dev-only allowed; run.py uses it but pipeline is separate

Uses:
- build_features (T3.7.3, vectorized) for feature engineering
- LightGBM with default params (T3.7.4: defaults near-optimal) and explicit
  num_threads (ECUP26_NUM_THREADS env override, else os.cpu_count())
- Dev-resume cache (caching.py): content-addressed feature matrix + model,
  enabled when the caller passes cache_dir (run.py does by default)

API:
    predict_for_users(train_df, user_ids, cache_dir=None, use_cache=True) -> np.ndarray
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecup26_user_value import caching, hybrid
from ecup26_user_value.features.build_features import (
    DEFAULT_ROLLING_WINDOWS,
    build_features,
)

logger = logging.getLogger(__name__)

SEED = 42

#: Number of boosting rounds (T3.9.5 re-tune + T3.9.7 fine: 550 @ lr 0.02 ->
#: fold A 1.7416 / fold B 1.6857, CV 1.7137).
N_ESTIMATORS = 550

#: Prediction horizon in days. The E-CUP target is GMV over the next 30 days;
#: since the future window is unknown at train time, the label is the most
#: recent 30-day GMV window and features are built strictly before it.
TARGET_DAYS = 30

#: Target transform: "log1p" trains L2 on log1p(30d GMV) so it directly
#: minimizes the contest RMSLE; "tweedie" uses LightGBM's tweedie objective
#: on the raw 30d GMV (handles the heavy zero-inflation + long tail).
TARGET_TRANSFORM = "log1p"

#: Walk-forward: train on multiple historical origins (features at time T,
#: label = GMV over (T, T+30]) instead of a single recent window.
WALK_FORWARD_ORIGINS = 8
WALK_FORWARD_STEP_DAYS = 30

#: Minimum free RAM (GB) required to build a walk-forward origin; origins are
#: skipped below this guard to avoid OOM-killing the host.
MIN_ORIGIN_MEM_GUARD_GB = 6.0

#: T3.8.10 hybrid stacking: add leak-free OOF P(Y>0) ('purchase_prob') as a
#: feature of the base LGBM. CV showed -0.0032 but LB DEGRADED (1.6672 vs
#: 1.6588) - REJECTED on LB. Default OFF; re-enable only for dev A/B via
#: ECUP26_USE_HYBRID=1 (cache key changes accordingly).
USE_HYBRID = os.environ.get("ECUP26_USE_HYBRID", "0") == "1"


def _default_num_threads() -> int:
    """LightGBM thread count: ECUP26_NUM_THREADS env override, else cpu count."""
    env = os.environ.get("ECUP26_NUM_THREADS")
    if env is not None and env.strip().isdigit():
        return max(1, int(env.strip()))
    return max(1, os.cpu_count() or 1)


def _lightgbm_params() -> dict[str, Any]:
    """LightGBM training params (shared by fit and the resume-cache key).

    objective: 'regression' on the log1p(30d GMV) label (directly minimizes
    RMSLE), or 'tweedie' on the raw label (zero-inflated heavy-tailed GMV).
    """
    params: dict[str, Any] = {
        "objective": "regression" if TARGET_TRANSFORM == "log1p" else "tweedie",
        "num_leaves": 97,
        "learning_rate": 0.02,
        "min_child_samples": 21,
        "feature_fraction": 0.66,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": SEED,
        "num_threads": _default_num_threads(),
    }
    if params["objective"] == "tweedie":
        params["tweedie_variance_power"] = 1.5
    return params


def _model_config() -> dict[str, Any]:
    """Full model configuration (params + rounds + target) for the cache key."""
    return {
        "params": _lightgbm_params(),
        "num_boost_round": N_ESTIMATORS,
        "target": TARGET_TRANSFORM,
        "target_days": TARGET_DAYS,
        "walk_forward": {
            "origins": int(os.environ.get("ECUP26_WALK_FORWARD_ORIGINS", WALK_FORWARD_ORIGINS)),
            "step_days": int(
                os.environ.get("ECUP26_WALK_FORWARD_STEP_DAYS", WALK_FORWARD_STEP_DAYS)
            ),
        },
        "hybrid": USE_HYBRID,
    }


def _build_features_for_users(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-user feature matrix from training transactions.

    Per R8: run.py is a submission entry point. Feature construction on
    30M rows takes ~7s (vectorized), so we wrap in tqdm to show progress.
    """
    from ecup26_user_value.features.build_features import build_features

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    ref = train_df["event_date"].max()
    n_rows = len(train_df)
    logger.info("Building features on %d transaction rows (ref_date=%s)...", n_rows, ref)

    if tqdm is not None:
        with tqdm(total=4, desc="Features", unit="step", file=sys.stderr, mininterval=0.5) as pbar:
            pbar.set_description("Features: starting")
            pbar.update(1)
            # Step 1: build_features returns the per-user matrix
            # (calendar + rolling + cross + tenure happen inside build_features)
            result = build_features(train_df, ref_date=ref)
            pbar.update(1)
            pbar.set_description("Features: done")
            pbar.update(2)
            return result
    else:
        # No tqdm fallback
        return build_features(train_df, ref_date=ref)


def _add_hybrid_feature(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features_infer: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T3.8.10: add leak-free OOF 'purchase_prob' to train and infer frames.

    The binary classifier's leave-origin-out P(Y>0) becomes a feature of the
    base LGBM. The origin column (added by _build_walk_forward_training with
    include_origin=True) is consumed here. Deterministic (seeded).
    """
    origin_col = X_train.pop("origin")
    y_bin = (np.asarray(y_train, dtype=np.float64) > 0).astype(np.int32)
    Xf = X_train.fillna(0.0)
    X_train["purchase_prob"] = hybrid.oof_purchase_prob(Xf.values, y_bin, origin_col.values, SEED)
    clf_full = hybrid.fit_purchase_prob(Xf.values, y_bin, SEED)
    features_infer["purchase_prob"] = np.asarray(
        clf_full.predict(features_infer.fillna(0.0).values), dtype=np.float64
    )
    logger.info(
        "T3.8.10 hybrid: purchase_prob added (leave-origin-out OOF, %d rows)",
        len(X_train),
    )
    return X_train, features_infer


def _fit_lightgbm(features: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit LightGBM via lgb.train() low-level API with real progress callback.

    Per R8: run.py is a submission entry point that may run for minutes.
    Users need visibility into what's happening. THREE layers:
    1. lgb.train() low-level: emits progress via custom callback (env.iteration).
    2. tqdm: visual progress bar updated by the callback.
    3. logger.info: structured logs at start/end and every 10 iterations.

    sklearn-API LGBMRegressor.fit() does NOT support custom callbacks
    the way lgb.train() does. We use lgb.train() directly with a dict
    of params, then return a small wrapper that exposes .predict() like
    sklearn-API for _predict_per_user.
    """
    import lightgbm as lgb

    X = features.copy().fillna(0.0)
    # y_train is already aligned positionally with X (both are built from the
    # same walk-forward rows); force positional alignment so duplicate user_id
    # index labels are safe (reindex by label would collapse duplicates).
    y = pd.Series(np.asarray(y_train, dtype=np.float64), index=X.index)
    n_estimators = N_ESTIMATORS
    logger.info(
        "Fitting LightGBMRegressor on %d rows x %d features (n_estimators=%d)...",
        len(X),
        X.shape[1],
        n_estimators,
    )

    # Heartbeat: emit timestamp every 10s during fit so the user sees
    # progress even if lgb.train() doesn't invoke our callback for some
    # reason. Runs in a daemon thread; joined after fit().
    import threading
    import time

    stop_heartbeat = threading.Event()

    def _heartbeat():
        t0 = time.time()
        while not stop_heartbeat.is_set():
            time.sleep(10)
            if stop_heartbeat.is_set():
                break
            logger.info(
                "LightGBM fit heartbeat: %.0fs elapsed, still running...",
                time.time() - t0,
            )

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    # Build lgb.Dataset from features
    train_set = lgb.Dataset(X, label=y)

    # LightGBM params dict (shared with the resume-cache key).
    params = _lightgbm_params()

    # tqdm wrapper for visual progress (best-effort, no-op if missing).
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # LightGBM callback protocol: callbacks receive env with .iteration
    # (0-indexed: 0..num_boost_round-1). We update tqdm on each iteration.
    callbacks = []
    if tqdm is not None:
        with tqdm(
            total=n_estimators, desc="LightGBM fit", unit="iter", file=sys.stderr, mininterval=1.0
        ) as pbar:

            def _tqdm_callback(env):
                # env.iteration is 0-indexed; display 1-indexed for users
                pbar.n = env.iteration + 1
                pbar.refresh()
                if env.iteration % 10 == 0:
                    logger.info("LightGBM iter %d/%d", env.iteration + 1, n_estimators)

            callbacks.append(_tqdm_callback)
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=n_estimators,
                callbacks=callbacks,
            )
            pbar.n = n_estimators
            pbar.refresh()
    else:
        # Fallback: no tqdm, still log every 10 iterations.
        def _log_callback(env):
            if env.iteration % 10 == 0:
                logger.info("LightGBM iter %d/%d", env.iteration + 1, n_estimators)

        callbacks.append(_log_callback)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            callbacks=callbacks,
        )

    stop_heartbeat.set()
    hb_thread.join(timeout=2)
    logger.info("LightGBM fit complete (booster.n_features=%d)", booster.num_feature())

    # Return a sklearn-API-compatible wrapper so _predict_per_user works
    # unchanged (model.predict(X, num_iteration=...) -- booster.predict
    # accepts num_iteration via keyword).
    return booster


def _predict_per_user(
    model: Any, features: pd.DataFrame, user_ids: pd.Index, calibration_bias: float = 0.0
) -> np.ndarray:
    """Predict 30-day GMV for each user_id.

    Inverts the training target transform (expm1 for log1p labels; tweedie
    already outputs GMV) and clips negatives to 0 (non-negative output).
    """
    X = features.copy().fillna(0.0)
    X = X.reindex(user_ids).fillna(0.0)
    raw = np.asarray(model.predict(X), dtype=np.float64)
    # T3.8.6: log-space bias correction (b fit on the walk-forward train).
    pred = np.expm1(raw + calibration_bias) if TARGET_TRANSFORM == "log1p" else raw
    return np.maximum(pred, 0.0)


def _split_for_30d_target(
    train_df: pd.DataFrame, target_days: int = TARGET_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Split train into feature/target windows for a no-leakage 30-day label.

    The contest asks for GMV over the next 30 days (after the last train date).
    Since the true future window is unknown at train time, the label is the
    most recent 30-day GMV window, and features must be built strictly before
    that window -- otherwise the features would literally contain the label
    (train-time leakage) and the model would collapse at inference time.

    Returns:
        feature_df: rows with event_date <= feature_cutoff (feature source).
        target_df: rows in (feature_cutoff, ref_date] (the 30-day label).
        feature_cutoff: last date allowed for features.
        ref_date: last date in the data (start of the future window).
    """
    work = train_df.copy()
    work["event_date"] = pd.to_datetime(work["event_date"])
    ref_date = work["event_date"].max()
    feature_cutoff = ref_date - pd.Timedelta(days=target_days)
    feature_df = work[work["event_date"] <= feature_cutoff]
    target_df = work[(work["event_date"] > feature_cutoff) & (work["event_date"] <= ref_date)]
    return feature_df, target_df, feature_cutoff, ref_date


def _transform_target(y_30d: pd.Series) -> pd.Series:
    """Transform the 30-day GMV label per TARGET_TRANSFORM."""
    if TARGET_TRANSFORM == "tweedie":
        return y_30d.astype(float)
    return np.log1p(y_30d).astype(float)


def _walk_forward_origins(
    ref_date: pd.Timestamp,
    n_origins: int = WALK_FORWARD_ORIGINS,
    step_days: int = WALK_FORWARD_STEP_DAYS,
) -> list[pd.Timestamp]:
    """Origin dates T such that (T, T+TARGET_DAYS] is a valid label window.

    The last origin is ref_date - TARGET_DAYS (its label is the most recent 30
    days); earlier origins step back by `step_days`.
    """
    # T3.8.4: env overrides for dev A/B (ECUP26_WALK_FORWARD_ORIGINS / _STEP_DAYS).
    n_origins = int(os.environ.get("ECUP26_WALK_FORWARD_ORIGINS", n_origins))
    step_days = int(os.environ.get("ECUP26_WALK_FORWARD_STEP_DAYS", step_days))
    last = ref_date - pd.Timedelta(days=TARGET_DAYS)
    origins = [last - pd.Timedelta(days=i * step_days) for i in range(n_origins)]
    # T3.13/Г3#1: inject historical gift-season origins (e.g. 2025-02-13, whose
    # label window is the fold-G gift window) so walk-forward training sees a
    # gift-season order boom. Guarded: only origins whose FULL label window
    # (t, t+30] fits in data <= ref_date are added, so fold-G training (ref =
    # 2025-02-13) cannot leak its own truth window.
    extra = os.environ.get("ECUP26_EXTRA_ORIGINS", "")
    for s in [x.strip() for x in extra.split(",") if x.strip()]:
        t = pd.Timestamp(s)
        if t + pd.Timedelta(days=TARGET_DAYS) <= ref_date:
            origins.append(t)
    return sorted(set(origins), reverse=True)


def _available_mem_gb() -> float:
    """Available RAM in GB (Linux /proc/meminfo); inf if unknown."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB -> GB
    except OSError:
        pass
    return float("inf")


def _build_walk_forward_training(
    train_df: pd.DataFrame,
    ref_date: pd.Timestamp,
    include_origin: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a walk-forward training set from multiple historical origins.

    For each origin date T: features are built from data <= T and the label is
    GMV over (T, T+TARGET_DAYS]. Concatenating origins gives
    ~WALK_FORWARD_ORIGINS rows per user, which (a) multiplies the training
    signal and (b) makes the model robust to feature-distribution drift.

    When include_origin is True, X gains an 'origin' column (datetime of each
    row's origin) so stacking can split rows by origin without label leakage
    (T3.8.10). Callers must drop it before fitting a model.

    Returns:
        (X, y) with a user_id column in X and y positionally aligned to X rows.
    """
    origins = _walk_forward_origins(ref_date, WALK_FORWARD_ORIGINS, WALK_FORWARD_STEP_DAYS)
    frames: list[pd.DataFrame] = []
    built_origins: list[pd.Timestamp] = []
    for origin in origins:
        if _available_mem_gb() < MIN_ORIGIN_MEM_GUARD_GB:
            logger.warning(
                "Skipping walk-forward origin %s: low free RAM (guard %.1f GB)",
                origin.date(),
                MIN_ORIGIN_MEM_GUARD_GB,
            )
            continue
        feature_df = train_df[train_df["event_date"] <= origin]
        if feature_df.empty:
            continue
        feats = build_features(feature_df, ref_date=origin)
        if feats.empty:
            continue
        target_df = train_df[
            (train_df["event_date"] > origin)
            & (train_df["event_date"] <= origin + pd.Timedelta(days=TARGET_DAYS))
        ]
        y_30d = target_df.groupby("user_id")["price"].sum()
        feats["y_log"] = feats["user_id"].map(_transform_target(y_30d)).fillna(0.0)
        frames.append(feats)
        if include_origin:
            built_origins.append(origin)
        logger.info(
            "Walk-forward origin %s: %d users, %d features",
            origin.date(),
            len(feats),
            feats.shape[1] - 1,
        )
    if not frames:
        return pd.DataFrame(), pd.Series(dtype="float64")
    X = pd.concat(frames, ignore_index=True)
    if include_origin:
        X["origin"] = np.repeat(
            np.asarray(built_origins, dtype="datetime64[ns]"),
            [len(f) for f in frames],
        )
    y = X.pop("y_log").astype(float)
    return X, y


def predict_for_users(
    train_df: pd.DataFrame,
    user_ids: list[int] | pd.Series | np.ndarray,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Train pipeline on train_df, predict 30-day GMV for each user_id.

    Target: sum(price) over the next 30 days (matches the E-CUP RMSLE metric).
    Training uses a walk-forward design: for each historical origin date T,
    features are built from data <= T (no leakage into the label window) and
    the label is GMV over (T, T+30]. Concatenating origins multiplies the
    training signal and makes the model robust to feature drift. The label is
    transformed per TARGET_TRANSFORM (log1p directly minimizes RMSLE; tweedie
    handles the zero-inflated heavy tail) and inverted at prediction time.
    Inference uses features at ref_date (freshest signal for scoring).

    Dev-resume behavior (enabled when cache_dir is provided and use_cache is
    True): the walk-forward training set, the inference features and the fitted
    LightGBM Booster are stored in cache_dir under a content-addressed key. On
    re-runs with unchanged data the cache hit skips the feature builds and fit.

    Args:
        train_df: transaction-level DataFrame with user_id, event_date,
                  price (and optionally the full daily-schema activity columns).
        user_ids: list/Series/array of user_ids to predict for.
        cache_dir: optional directory for the dev-resume cache. When None,
                  caching is disabled (full recompute every run).
        use_cache: set False to force a full recompute even with cache_dir.

    Returns:
        np.ndarray of length len(user_ids) with non-negative floats.
    """
    if len(train_df) == 0:
        logger.warning("Empty train_df; emitting zero predictions")
        return np.zeros(len(user_ids), dtype=np.float64)

    work = train_df.copy()
    work["event_date"] = pd.to_datetime(work["event_date"])
    ref_date = work["event_date"].max()

    # Recent 30-day GMV mean (fallback when there is not enough history).
    last_win = work[
        (work["event_date"] > ref_date - pd.Timedelta(days=TARGET_DAYS))
        & (work["event_date"] <= ref_date)
    ]
    y_last = last_win.groupby("user_id")["price"].sum()

    cache_dir_p = Path(cache_dir) if cache_dir is not None else None
    cache_key: str | None = None
    X_train: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    features_infer: pd.DataFrame | None = None
    model: Any = None

    if use_cache and cache_dir_p is not None:
        cache_key = caching.feature_cache_key(
            work,
            rolling_windows=DEFAULT_ROLLING_WINDOWS,
            model_config=_model_config(),
        )
        parts = caching.load_cached_train(cache_dir_p, cache_key)
        features_infer = caching.load_cached_features(cache_dir_p, cache_key, kind="infer")
        model = caching.load_cached_model(cache_dir_p, cache_key)
        if (
            parts is not None
            and not parts[0].empty
            and features_infer is not None
            and not features_infer.empty
            and model is not None
        ):
            X_train, y_train = parts
            logger.info("Cache hit (%s): reusing walk-forward train + infer + model", cache_key)
        else:
            logger.info("Cache miss (%s): building walk-forward features + model", cache_key)

    if X_train is None or features_infer is None:
        logger.info(
            "Walk-forward: building features at %d origins (step=%dd, target=%s)...",
            WALK_FORWARD_ORIGINS,
            WALK_FORWARD_STEP_DAYS,
            TARGET_TRANSFORM,
        )
        X_train, y_train = _build_walk_forward_training(work, ref_date, include_origin=USE_HYBRID)
        features_infer = _build_features_for_users(work)
        if X_train.empty or features_infer.empty:
            # Not enough history to build a 30-day model (e.g. tiny dev data).
            logger.warning(
                "No training features (history < %d days); predicting global 30d mean",
                TARGET_DAYS,
            )
            mean_30d = float(y_last.mean()) if len(y_last) else 0.0
            return np.full(len(user_ids), mean_30d, dtype=np.float64)
        if USE_HYBRID:
            # T3.8.10: leak-free OOF purchase_prob feature (leave-origin-out).
            X_train, features_infer = _add_hybrid_feature(X_train, y_train, features_infer)
        if cache_dir_p is not None and cache_key is not None:
            caching.save_cached_train(cache_dir_p, cache_key, X_train, y_train)
            caching.save_cached_features(cache_dir_p, cache_key, features_infer, kind="infer")

    # T3.8.10: the walk-forward train may carry an 'origin' column (needed to
    # compute the leak-free OOF purchase_prob); it is never a model feature.
    if "origin" in X_train.columns:
        X_train = X_train.drop(columns=["origin"])

    X_train_indexed = X_train.set_index("user_id")
    features_infer_indexed = features_infer.set_index("user_id")

    calibration_bias: float = 0.0
    if model is None:
        model = _fit_lightgbm(X_train_indexed, y_train)
        # T3.8.6: log-space bias = mean(log1p(y) - log1p(pred)) over the train set.
        raw_train = np.asarray(model.predict(X_train_indexed.fillna(0.0)), dtype=np.float64)
        calibration_bias = float(np.mean(np.asarray(y_train, dtype=np.float64) - raw_train))
        if cache_dir_p is not None and cache_key is not None:
            caching.save_cached_model(cache_dir_p, cache_key, model)
            caching.save_cached_bias(cache_dir_p, cache_key, calibration_bias)
            logger.info(
                "Saved walk-forward train + infer + model + bias %.4f to %s",
                calibration_bias,
                cache_dir_p,
            )
    elif cache_dir_p is not None and cache_key is not None:
        calibration_bias = caching.load_cached_bias(cache_dir_p, cache_key) or 0.0

    user_ids_arr = pd.Index(user_ids)
    return _predict_per_user(model, features_infer_indexed, user_ids_arr, calibration_bias)
