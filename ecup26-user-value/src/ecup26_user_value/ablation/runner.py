"""Runner for T3.6.12 ablation matrix.

Computes leave-one-out combinations of feature groups, runs each model on
time-based CV folds, and returns a DataFrame:

    model | features | fold | rmsle | n_samples

Per .clinerules/02-ecup-rules.md R6: deterministic, no leakage.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY
from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def _time_folds(
    df: pd.DataFrame,
    n_folds: int,
    holdout_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Build time-based CV folds.

    Each fold:
        train = df[event_date <= cut_end]
        val   = df[cut_end < event_date <= cut_end + holdout_days]
    """
    if len(df) == 0:
        return []
    dates = df["event_date"].sort_values().unique()
    if len(dates) < n_folds + 1:
        return [(dates[0], dates[-1], df, df.copy())]

    span = dates[-1] - dates[0]
    step = span / (n_folds + 1)
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds):
        cut_end = dates[0] + step * (i + 1)
        val_end = cut_end + pd.Timedelta(days=holdout_days)
        train = df[df["event_date"] <= cut_end]
        val = df[(df["event_date"] > cut_end) & (df["event_date"] <= val_end)]
        if len(train) == 0 or len(val) == 0:
            continue
        folds.append((cut_end, val_end, train, val))
    return folds


def _build_feature_matrix(
    df: pd.DataFrame,
    group_names: Iterable[str],
) -> pd.DataFrame:
    """Build per-user feature matrix from a list of feature group names."""
    parts: list[pd.DataFrame] = []
    for name in group_names:
        parts.append(FEATURE_GROUP_REGISTRY[name].transform(df))
    if not parts:
        return pd.DataFrame({"user_id": df["user_id"].unique()})
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="user_id", how="outer")
    return out.fillna(0.0)


def _aggregate_per_user(X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user feature matrix to one row per user (sum)."""
    if len(X) == 0:
        return pd.DataFrame()
    return X.groupby("user_id").sum(numeric_only=True)


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE per task3_user_value.md: log1p, max(0, y)."""
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def run_ablation(
    df: pd.DataFrame,
    target: pd.Series,
    folds: int = 2,
    holdout_days: int = 30,
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run ablation: leave-one-out feature combinations x models x time-fold CV.

    Args:
        df: transaction-level DataFrame with user_id, event_date, type, price.
        target: per-user target series (indexed by user_id) used to score.
        folds: number of time-CV folds.
        holdout_days: validation window length in days.
        model_names: subset of MODEL_REGISTRY to run (default: all).

    Returns:
        DataFrame with columns: model, features, fold, rmsle, n_samples.
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    all_groups = list(FEATURE_GROUP_REGISTRY.keys())
    # Leave-one-out: 'all' + skip each group in turn.
    combos: list[frozenset[str]] = [frozenset(all_groups)]
    for skip in all_groups:
        combos.append(frozenset(g for g in all_groups if g != skip))

    cv_folds = _time_folds(df, n_folds=folds, holdout_days=holdout_days)
    if not cv_folds:
        return pd.DataFrame(columns=["model", "features", "fold", "rmsle", "n_samples"])

    rows: list[dict[str, object]] = []
    for fold_idx, (_cut_end, _val_end, train_df, val_df) in enumerate(cv_folds):
        # Build per-combo train and val feature matrices
        train_X_per_combo: dict[frozenset[str], pd.DataFrame] = {}
        val_X_per_combo: dict[frozenset[str], pd.DataFrame] = {}
        for combo in combos:
            train_X_per_combo[combo] = _build_feature_matrix(train_df, combo)
            val_X_per_combo[combo] = _build_feature_matrix(val_df, combo)

        # val_user_index = unique users in val
        val_user_index = pd.Series(val_df["user_id"].unique()).values
        y_true = target.reindex(val_user_index).fillna(0.0).values

        for mname in model_names:
            proto = MODEL_REGISTRY[mname]
            model = proto.build()

            if model.kind == "transaction":
                model.fit(train_df)
                pred_per_user = model.predict_per_user(val_df)
                pred_series = pd.Series(
                    pred_per_user,
                    index=pd.Series(val_df["user_id"].unique()).values,
                )
                pred_per_user_aligned = pred_series
            else:
                pred_per_user_aligned = None  # computed per-combo below

            for combo in combos:
                if model.kind == "transaction":
                    # BTYD ignores feature LOO (it sees raw transactions);
                    # we report one row per (model, fold), not per combo.
                    if combo != combos[0]:
                        continue
                    pred_aligned = (
                        pred_per_user_aligned.reindex(pd.Series(val_user_index)).fillna(0.0).values
                    )
                else:
                    X_train = _aggregate_per_user(train_X_per_combo[combo])
                    y_train = target.reindex(X_train.index).fillna(0.0)
                    model.fit(X_train, y_train)
                    val_X = _aggregate_per_user(val_X_per_combo[combo])
                    val_X = val_X.reindex(X_train.columns, axis=1, fill_value=0.0)
                    pred = model.predict(val_X)
                    pred_aligned = (
                        pd.Series(pred, index=val_X.index)
                        .reindex(pd.Series(val_user_index))
                        .fillna(0.0)
                        .values
                    )

                rmsle = _rmsle(y_true, pred_aligned)
                feat_label = (
                    "(transactions, no LOO)"
                    if model.kind == "transaction"
                    else ("+".join(sorted(combo)) if combo else "<empty>")
                )
                rows.append(
                    {
                        "model": mname,
                        "features": feat_label,
                        "fold": fold_idx,
                        "rmsle": rmsle,
                        "n_samples": len(val_user_index),
                    }
                )

    return pd.DataFrame(rows)
