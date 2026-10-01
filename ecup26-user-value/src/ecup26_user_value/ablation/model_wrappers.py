"""ModelWrapper registry for T3.6.12 ablation matrix (v2: includes ZILN composite).

Each wrapper exposes a unified interface over heterogeneous backends:
- BTYD models: fit on transaction-level df, predict_per_user() aggregates rows
- LightGBM models: fit on per-user feature matrix X + y, predict(X) returns per-user

The runner in runner.py dispatches based on the wrapper's kind attribute.

Provides:
- ModelWrapper ABC
- register() decorator
- Active models (v2):
    * BTYDBaselineWrapper (T3.6.4)
    * SegmentedBTYDWrapper (T3.6.14)
    * LightGBMRegressorMSE (sklearn-API)
    * LightGBMZILNComposite (T3.6.18) - 3 single-output LightGBM models
      composing ZILN formula (z_prob, mu, sigma) -> E[y]
- MODEL_REGISTRY: dict[str, ModelWrapper]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd

from ecup26_user_value.models.btyd_baseline import BTYDBaseline
from ecup26_user_value.models.btyd_segmented import SegmentedBTYD
from ecup26_user_value.models.ziln import ziln_predict

logger = logging.getLogger(__name__)

WrapperKind = Literal["transaction", "tabular"]


class ModelWrapper(ABC):
    """Unified fit/predict interface (kind-dispatched by runner)."""

    name: str = ""
    kind: WrapperKind = "tabular"

    @abstractmethod
    def build(self) -> ModelWrapper:
        """Return a fresh, unfitted instance."""


class BTYDBaselineWrapper(ModelWrapper):
    """Global BTYDBaseline - one model for all users (transaction-level)."""

    name = "btyd_global"
    kind = "transaction"

    def build(self) -> BTYDBaselineWrapper:
        w = BTYDBaselineWrapper()
        w.model = BTYDBaseline(penalizer_coef=0.01, time_period_days=30)
        return w

    def fit(self, df: pd.DataFrame) -> BTYDBaselineWrapper:
        self.model.fit(df, customer_id_col="user_id", order_date_col="event_date")
        return self

    def predict_per_user(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(df, customer_id_col="user_id", order_date_col="event_date")
        if len(pred) != len(df):
            raise RuntimeError(f"Model {self.name} returned {len(pred)} preds for {len(df)} rows")
        out = pd.Series(pred, index=df["user_id"].values).groupby(level=0).sum()
        return np.maximum(out.values, 0.0)


class SegmentedBTYDWrapper(ModelWrapper):
    """SegmentedBTYD - per-search-vs-cat model + global fallback (transaction-level)."""

    name = "btyd_segmented"
    kind = "transaction"

    def build(self) -> SegmentedBTYDWrapper:
        w = SegmentedBTYDWrapper()
        w.model = SegmentedBTYD(penalizer_coef=0.01, time_period_days=30)
        return w

    def fit(self, df: pd.DataFrame) -> SegmentedBTYDWrapper:
        self.model.fit(df, order_date_col="event_date")
        return self

    def predict_per_user(self, df: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(df, order_date_col="event_date")
        if len(pred) != len(df):
            raise RuntimeError(f"Model {self.name} returned {len(pred)} preds for {len(df)} rows")
        out = pd.Series(pred, index=df["user_id"].values).groupby(level=0).sum()
        return np.maximum(out.values, 0.0)


class LightGBMRegressorMSE(ModelWrapper):
    """LightGBM regressor with MSE objective (tabular features)."""

    name = "lgbm_mse"
    kind = "tabular"

    def build(self) -> LightGBMRegressorMSE:
        import lightgbm as lgb

        w = LightGBMRegressorMSE()
        w._booster = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            verbose=-1,
        )
        return w

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMRegressorMSE:
        self._booster.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self._booster.predict(X)
        return np.maximum(np.asarray(pred, dtype=np.float64), 0.0)


class LightGBMZILNComposite(ModelWrapper):
    """ZILN distribution via 3 single-output LightGBM models (T3.6.18).

    Sub-models:
    - _zprob_model: binary classifier, P(y > 0); z_prob = 1 - P
    - _mu_model: regression on log1p(y) for y > 0 (with zeros mapped to 0)
    - _sigma_model: regression on squared log-residuals; clipped to >=1e-3

    Prediction: E[y] = (1 - sigmoid(zprob)) * exp(mu + sigma^2 / 2).
    """

    name = "lgbm_ziln"
    kind = "tabular"

    def build(self) -> LightGBMZILNComposite:
        import lightgbm as lgb

        w = LightGBMZILNComposite()
        w._zprob_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            verbose=-1,
        )
        w._mu_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            verbose=-1,
        )
        w._sigma_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=15,
            verbose=-1,
        )
        return w

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMZILNComposite:
        import numpy as np

        y_arr = np.asarray(y, dtype=np.float64)
        # Sub-model 1: P(y > 0)
        y_binary = (y_arr > 0).astype(np.float64)
        self._zprob_model.fit(X, y_binary)

        # Sub-model 2: mu = log1p(y) for y > 0; 0 otherwise
        mu_target = np.log1p(y_arr)
        self._mu_model.fit(X, mu_target)

        # Sub-model 3: sigma^2 = (log1p(y) - mu_hat)^2; clip to >=1e-3 at predict
        mu_hat = self._mu_model.predict(X)
        sigma_sq = (mu_target - mu_hat) ** 2
        self._sigma_model.fit(X, sigma_sq)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z_prob_raw = self._zprob_model.predict_proba(X)[:, 1]  # P(y > 0)
        z_prob = 1.0 - z_prob_raw  # P(y = 0)
        mu = self._mu_model.predict(X)
        sigma_sq = np.maximum(self._sigma_model.predict(X), 1e-3)
        sigma = np.sqrt(sigma_sq)
        return np.asarray(ziln_predict(z_prob, mu, sigma), dtype=np.float64)


_REGISTRY: dict[str, ModelWrapper] = {}
MODEL_REGISTRY: dict[str, ModelWrapper] = _REGISTRY


def register(wrapper: ModelWrapper) -> ModelWrapper:
    if not wrapper.name:
        raise ValueError("ModelWrapper.name must be non-empty")
    _REGISTRY[wrapper.name] = wrapper
    return wrapper


# Eager-register prototype instances (factory pattern via .build())
register(BTYDBaselineWrapper())
register(SegmentedBTYDWrapper())
register(LightGBMRegressorMSE())
register(LightGBMZILNComposite())
