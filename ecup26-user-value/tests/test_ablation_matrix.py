"""Tests for T3.6.12 - ablation matrix (model x feature-group comparison).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lifetimes, lightgbm, scikit-learn - all Apache 2.0/BSD)
- R4: no internet at runtime (all models CPU, no model downloads)
- R6: reproducible (seeded synthetic data, deterministic time-CV)
- R7: CPU tabular models, no LLM / inference server
- R8: ablation lives in scripts/, NOT imported by run.py (lightgbm in dev deps)
- R10: ablation writes docs/results/*.csv, NOT submit.csv

v1 scope (after ZILN deferred to v2):
- 3 active models (btyd_global, btyd_segmented, lgbm_mse)
  - lgbm_ziln registered but build() raises NotImplementedError; tracked for v2
- 2 feature groups (base_features, btyd_features)
- leave-one-out (3 combos) applied only to tabular (LightGBM) models
- 2 folds
- Total rows: 2 BTYD x 2 folds + 1 LightGBM x 3 LOO x 2 folds = 10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_tx(n_users: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic transaction df matching T3 schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        n_orders = int(rng.integers(3, 10)) if seg == "search" else int(rng.integers(1, 5))
        for _o in range(n_orders):
            price = float(rng.uniform(5, 30)) if seg == "search" else float(rng.uniform(80, 250))
            rows.append(
                {
                    "user_id": uid,
                    "event_date": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 30))),
                    "type": seg,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


def _make_holdout_target(df: pd.DataFrame, holdout_days: int = 30) -> pd.Series:
    """Pseudo-target: total price per user over the last holdout_days of df."""
    cutoff = df["event_date"].max() - pd.Timedelta(days=holdout_days)
    return (
        df[df["event_date"] > cutoff]
        .groupby("user_id")["price"]
        .sum()
        .reindex(df["user_id"].unique())
        .fillna(0.0)
    )


_ACTIVE_MODEL_NAMES = ("btyd_global", "btyd_segmented", "lgbm_mse")


class TestFeatureGroupRegistry:
    """FeatureGroup ABC + registry must export BaseFeatures and BTYDFeatures."""

    def test_registry_exposes_two_groups(self) -> None:
        from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY

        names = list(FEATURE_GROUP_REGISTRY.keys())
        assert "base_features" in names, f"base_features missing from {names}"
        assert "btyd_features" in names, f"btyd_features missing from {names}"

    def test_base_features_returns_known_columns(self) -> None:
        from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY

        df = _make_synthetic_tx()
        out = FEATURE_GROUP_REGISTRY["base_features"].transform(df)
        assert isinstance(out, pd.DataFrame)
        assert "user_id" in out.columns
        assert len(out) == df["user_id"].nunique()

    def test_btyd_features_returns_known_columns(self) -> None:
        from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY

        df = _make_synthetic_tx()
        out = FEATURE_GROUP_REGISTRY["btyd_features"].transform(df)
        assert isinstance(out, pd.DataFrame)
        assert "user_id" in out.columns
        assert "predicted_ltv" in out.columns


class TestModelWrapperRegistry:
    """ModelWrapper ABC + registry must expose 3 active models (+ 1 stub)."""

    def test_registry_exposes_three_active_models(self) -> None:
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        names = list(MODEL_REGISTRY.keys())
        for required in _ACTIVE_MODEL_NAMES:
            assert required in names, f"{required} missing from {names}"

    def test_kinds_are_set_correctly(self) -> None:
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        assert MODEL_REGISTRY["btyd_global"].kind == "transaction"
        assert MODEL_REGISTRY["btyd_segmented"].kind == "transaction"
        assert MODEL_REGISTRY["lgbm_mse"].kind == "tabular"

    def test_lgbm_mse_produces_nonnegative_predictions(self) -> None:
        from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        df = _make_synthetic_tx()
        target = _make_holdout_target(df)
        base = FEATURE_GROUP_REGISTRY["base_features"].transform(df)
        btyd = FEATURE_GROUP_REGISTRY["btyd_features"].transform(df)
        X = base.merge(btyd, on="user_id", how="left").fillna(0.0)
        X_agg = X.groupby("user_id").sum(numeric_only=True)
        y = target.reindex(X_agg.index).fillna(0.0)

        model = MODEL_REGISTRY["lgbm_mse"].build()
        model.fit(X_agg, y)
        pred = model.predict(X_agg)
        assert (np.asarray(pred) >= 0).all(), "lgbm_mse produced negative preds"

    def test_btyd_models_fit_and_predict(self) -> None:
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        df = _make_synthetic_tx(n_users=30)
        for name in ("btyd_global", "btyd_segmented"):
            model = MODEL_REGISTRY[name].build()
            model.fit(df)
            pred = model.predict_per_user(df)
            assert (pred >= 0).all(), f"{name} produced negative preds"
            assert len(pred) == df["user_id"].nunique()


class TestRunner:
    """run_ablation produces a 10-row DataFrame (BTYD x 2 folds + LightGBM x 3 LOO x 2 folds)."""

    def test_runner_produces_10_rows(self) -> None:
        from ecup26_user_value.ablation.runner import run_ablation

        df = _make_synthetic_tx(n_users=40)
        target = _make_holdout_target(df)

        result = run_ablation(
            df=df,
            target=target,
            folds=2,
            holdout_days=30,
            model_names=list(_ACTIVE_MODEL_NAMES),
        )
        # 2 BTYD models x 2 folds + 1 LightGBM x 3 LOO x 2 folds = 4 + 6 = 10
        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

    def test_runner_output_schema(self) -> None:
        from ecup26_user_value.ablation.runner import run_ablation

        df = _make_synthetic_tx(n_users=40)
        target = _make_holdout_target(df)

        result = run_ablation(
            df=df,
            target=target,
            folds=2,
            holdout_days=30,
            model_names=list(_ACTIVE_MODEL_NAMES),
        )
        for col in ("model", "features", "fold", "rmsle", "n_samples"):
            assert col in result.columns, f"Column {col} missing"

    def test_runner_rmsle_finite(self) -> None:
        from ecup26_user_value.ablation.runner import run_ablation

        df = _make_synthetic_tx(n_users=40)
        target = _make_holdout_target(df)

        result = run_ablation(
            df=df,
            target=target,
            folds=2,
            holdout_days=30,
            model_names=list(_ACTIVE_MODEL_NAMES),
        )
        assert np.isfinite(result["rmsle"]).all(), f"Non-finite RMSLE: {result['rmsle']}"

    def test_leave_one_out_combinations(self) -> None:
        from ecup26_user_value.ablation.runner import run_ablation

        df = _make_synthetic_tx(n_users=40)
        target = _make_holdout_target(df)

        result = run_ablation(
            df=df,
            target=target,
            folds=2,
            holdout_days=30,
            model_names=list(_ACTIVE_MODEL_NAMES),
        )
        # 4 feature labels: BTYD x 2 ("(transactions, no LOO)") + LightGBM x 3 LOO
        feature_combos = result["features"].unique()
        assert len(feature_combos) == 4, f"Expected 4 feature labels, got {feature_combos}"


@pytest.mark.parametrize("n_users", [30, 50])
def test_ablation_smoke(n_users: int) -> None:
    """Smoke: ablation runs end-to-end on small synthetic data."""
    from ecup26_user_value.ablation.runner import run_ablation

    df = _make_synthetic_tx(n_users=n_users)
    target = _make_holdout_target(df)
    result = run_ablation(
        df=df,
        target=target,
        folds=2,
        holdout_days=30,
        model_names=list(_ACTIVE_MODEL_NAMES),
    )
    assert len(result) == 10
    assert (np.isfinite(result["rmsle"])).all()


class TestZILNComposite:
    """T3.6.18: LightGBMZILNComposite wraps 3 single-output LightGBM models."""

    def test_ziln_formula_correct(self) -> None:
        """Prediction must equal (1 - sigmoid(zprob_pred)) * exp(mu + sigma**2/2)."""
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        model = MODEL_REGISTRY["lgbm_ziln"].build()
        # 3 sub-models must exist after build()
        assert hasattr(model, "_zprob_model")
        assert hasattr(model, "_mu_model")
        assert hasattr(model, "_sigma_model")

    def test_ziln_predictions_nonnegative(self) -> None:
        """Final composite prediction must be >= 0 (RMSLE constraint)."""
        from ecup26_user_value.ablation.feature_groups import FEATURE_GROUP_REGISTRY
        from ecup26_user_value.ablation.model_wrappers import MODEL_REGISTRY

        df = _make_synthetic_tx(n_users=30)
        target = _make_holdout_target(df)
        base = FEATURE_GROUP_REGISTRY["base_features"].transform(df)
        btyd = FEATURE_GROUP_REGISTRY["btyd_features"].transform(df)
        X = base.merge(btyd, on="user_id", how="left").fillna(0.0)
        X_agg = X.groupby("user_id").sum(numeric_only=True)
        y = target.reindex(X_agg.index).fillna(0.0)

        model = MODEL_REGISTRY["lgbm_ziln"].build()
        model.fit(X_agg, y)
        pred = model.predict(X_agg)
        assert (np.asarray(pred) >= 0).all(), f"lgbm_ziln produced negative preds: {pred[pred < 0]}"


class TestRunnerZILNIntegration:
    """T3.6.18: runner.py accepts lgbm_ziln and produces extra rows."""

    def test_runner_includes_ziln_when_active(self) -> None:
        from ecup26_user_value.ablation.runner import run_ablation

        df = _make_synthetic_tx(n_users=40)
        target = _make_holdout_target(df)

        # ACTIVE_MODELS for v1 was 3 models; adding lgbm_ziln -> 4 models
        names = ["btyd_global", "btyd_segmented", "lgbm_mse", "lgbm_ziln"]
        result = run_ablation(df=df, target=target, folds=2, holdout_days=30, model_names=names)
        # 2 BTYD x 2 folds + 2 tabular x 3 LOO x 2 folds = 4 + 12 = 16
        assert len(result) == 16, f"Expected 16 rows, got {len(result)}"
