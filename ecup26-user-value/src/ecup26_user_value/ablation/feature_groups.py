"""FeatureGroup registry for T3.6.12 ablation matrix.

A FeatureGroup takes a transaction-level DataFrame and returns a per-user
feature DataFrame (one row per user_id). Groups are pure-data transforms -
no model training, no leakage.

Provides:
- FeatureGroup ABC
- register() decorator
- BaseFeatures: historical aggregates from train.parquet
- BTYDFeatures: per-segment BG/NBD parameters from features.btyd_segments
- FEATURE_GROUP_REGISTRY: dict[str, FeatureGroup]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ecup26_user_value.features.btyd_segments import compute_btyd_segments


class FeatureGroup(ABC):
    """Abstract base for a feature group."""

    name: str = ""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-user (user_id, ...) DataFrame.

        Args:
            df: transaction-level DataFrame with columns:
                user_id, event_date, type, price (configurable via constructor).
        Returns:
            DataFrame with user_id as one column and feature values for the rest.
        """


_REGISTRY: dict[str, FeatureGroup] = {}


def register(group: FeatureGroup) -> FeatureGroup:
    """Register a FeatureGroup instance in the global registry by its .name."""
    if not group.name:
        raise ValueError("FeatureGroup.name must be non-empty")
    _REGISTRY[group.name] = group
    return group


FEATURE_GROUP_REGISTRY: dict[str, FeatureGroup] = _REGISTRY


class BaseFeatures(FeatureGroup):
    """Historical GMV aggregates per user (no lookahead, no model).

    Computes per-user sums over the *entire* input df. Time-based CV in the
    runner handles train/val splits to avoid leakage.
    """

    name = "base_features"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("user_id").agg(
            gmv_total=("price", "sum"),
            orders_total=("price", "count"),
            gmv_mean=("price", "mean"),
        )
        return agg.reset_index()


class BTYDFeatures(FeatureGroup):
    """Per-segment BG/NBD parameters from features.btyd_segments.

    Returns one row per user with aggregated segment predictions.
    """

    name = "btyd_features"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        seg_df = compute_btyd_segments(df, ref_date=df["event_date"].max())
        if seg_df.empty:
            return pd.DataFrame({"user_id": df["user_id"].unique()})
        # Aggregate per-user across segments
        agg = seg_df.groupby("user_id").agg(
            predicted_ltv=("predicted_ltv", "sum"),
            p_alive_max=("p_alive", "max"),
            btyd_freq_sum=("frequency", "sum"),
            btyd_monetary_mean=("monetary", "mean"),
        )
        return agg.reset_index()


# Eager-register on import
register(BaseFeatures())
register(BTYDFeatures())
