#!/usr/bin/env python3
"""T3.8.1 benchmarker — combinatorial hyperparameter search on walk-forward CV.

Finds the best pipeline config for the ecup26-user-value submission via
walk-forward CV over multiple historical holdout windows. The winner is then
baked into `run_pipeline.py` as fixed constants; the search itself is NOT
embedded in the submission (R9 time limits, R6 reproducibility).

Structure borrowed from the author's own `BenchmarkService`
(`backend/src/application/benchmark_service.py` in diplomMagistrate): dataclass
results + runner + `format_results_table` + CSV.

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lightgbm Apache-2.0)
- R6: reproducible (seeded)
- R8: dev-only tool, NOT imported by run.py
- R9: search runs offline; submission keeps fixed constants
- Leakage: uses only the provided train data + own holdout splits (never the
  organizers' test labels)

Usage:
  # grid (curated coordinate-style), 2-fold walk-forward CV
  uv run python scripts/benchmark_t3.py --strategy grid --max-configs 12 \\
      --output docs/results/benchmark_<date>.md
  # random search
  uv run python scripts/benchmark_t3.py --strategy random --max-configs 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value import hybrid  # noqa: E402
from ecup26_user_value.features.build_features import build_features  # noqa: E402
from ecup26_user_value.run_pipeline import (  # noqa: E402
    TARGET_DAYS,
    _build_walk_forward_training,
    _default_num_threads,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_t3")

SEED = 42

#: Search space (dev-only). Winner is baked into run_pipeline constants.
SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [50, 100, 150, 200, 300],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.03, 0.05, 0.1],
    "feature_fraction": [0.7, 0.8, 1.0],
    "bagging_fraction": [0.7, 0.8, 1.0],
    "min_child_samples": [10, 20, 50],
}

#: Current production config (reference baseline for every run).
BASE_CONFIG: dict[str, float | int] = {
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "min_child_samples": 20,
}

#: Default cache dir for walk-forward fold datasets.
FOLD_CACHE_DIR = Path("data/folds")

_KEEP = [
    "user_id",
    "event_date",
    "price",
    "gmv",
    "search",
    "cat",
    "searches",
    "to_cart",
    "to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
]


def _load_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    if "price" not in df.columns and "gmv" in df.columns:
        df["price"] = df["gmv"].astype(np.float32)
    df = df[[c for c in _KEEP if c in df.columns]]
    for c in ("price", "gmv", "gmv_search", "gmv_cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    for c in (
        "searches",
        "to_cart",
        "to_ord",
        "search_to_cart",
        "search_to_ord",
        "cat_to_cart",
        "cat_to_ord",
    ):
        if c in df.columns:
            df[c] = df[c].astype(np.int32)
    for c in ("search", "cat"):
        if c in df.columns:
            df[c] = df[c].astype(np.int8)
    return df


@dataclass
class ConfigResult:
    """Result of evaluating one config on the walk-forward CV folds."""

    config: dict
    rmsle_mean: float
    rmsle_std: float
    rmsle_folds: list[float]
    pred_mean: float
    pred_median: float
    fit_time_sec: float


def curated_grid() -> list[dict]:
    """Coordinate-style grid: base config + one-parameter variations."""
    base = BASE_CONFIG.copy()
    configs = [base]
    variations: dict[str, list] = {
        "n_estimators": [100, 150, 300],
        "num_leaves": [15, 63],
        "learning_rate": [0.03, 0.07, 0.1],
        "feature_fraction": [0.7, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.9, 1.0],
        "min_child_samples": [10, 50],
    }
    for key, values in variations.items():
        for v in values:
            cfg = base.copy()
            cfg[key] = v
            configs.append(cfg)
    configs.append({**base, "n_estimators": 300, "learning_rate": 0.03})
    configs.append(
        {
            **base,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
        }
    )
    # old single-origin baseline for reference
    configs.append(
        {
            "n_estimators": 50,
            "num_leaves": 15,
            "learning_rate": 0.1,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "min_child_samples": 20,
        }
    )
    return configs


def random_configs(n: int, seed: int = SEED) -> list[dict]:
    """Sample n unique random configs from the full space (seeded)."""
    rng = random.Random(seed)
    seen: set = set()
    configs: list[dict] = []
    while len(configs) < n and len(seen) < 10_000_000:
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def make_params(cfg: dict, seed: int) -> dict:
    """Build LightGBM train params from a config dict."""
    return {
        "objective": "regression",
        "num_leaves": int(cfg["num_leaves"]),
        "learning_rate": float(cfg["learning_rate"]),
        "min_child_samples": int(cfg["min_child_samples"]),
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": int(seed),
        "num_threads": _default_num_threads(),
    }


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def _build_fold(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    name: str,
    cache_dir: Path = FOLD_CACHE_DIR,
) -> None:
    """Build + cache one walk-forward CV fold (train / infer / truth)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_df = data[data["event_date"] <= cutoff]
    X, y = _build_walk_forward_training(train_df, cutoff, include_origin=True)
    out = X.copy()
    out["y_log"] = np.asarray(y, dtype=np.float64)
    out.to_parquet(cache_dir / f"fold_{name}_train.parquet", index=False)
    X_infer = build_features(train_df, ref_date=cutoff)
    X_infer.to_parquet(cache_dir / f"fold_{name}_infer.parquet", index=False)
    truth = data[
        (data["event_date"] > cutoff)
        & (data["event_date"] <= cutoff + pd.Timedelta(days=TARGET_DAYS))
    ]
    truth.groupby("user_id")["price"].sum().rename("y_true").to_frame().to_parquet(
        cache_dir / f"fold_{name}_truth.parquet"
    )
    logger.info("Fold %s built: cutoff=%s train_rows=%d", name, cutoff.date(), len(out))


def _eval_fold(
    cfg: dict,
    name: str,
    seed: int,
    cache_dir: Path = FOLD_CACHE_DIR,
) -> tuple[float, float, float, float]:
    """Train cfg on a cached fold; return (rmsle, pred_mean, pred_median, time)."""
    import lightgbm as lgb

    t0 = time.time()
    X = pd.read_parquet(cache_dir / f"fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(cache_dir / f"fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(cache_dir / f"fold_{name}_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    params = make_params(cfg, seed)
    model = lgb.train(
        params,
        lgb.Dataset(X.fillna(0.0), label=np.asarray(y)),
        num_boost_round=int(cfg["n_estimators"]),
    )
    Xv = infer.reindex(all_users).fillna(0.0)
    pred = np.expm1(np.asarray(model.predict(Xv), dtype=np.float64))
    y_true = truth.reindex(all_users).fillna(0.0).values
    fit_time = time.time() - t0
    return rmsle(y_true, pred), float(pred.mean()), float(np.median(pred)), fit_time


def run_two_stage(
    cfg: dict,
    folds: list[str],
    cache_dir: Path,
    seed: int,
) -> dict[str, float]:
    """Two-stage v2 + bias-correction diagnostic (Phase 1).

    Split hyperparameters (classifier vs regressor). Returns mean RMSLE for
    raw, raw-space-corrected (x mean(y)/mean(pred)) and log-space-corrected
    (expm1(log1p(pred)+b)). If corrected > 1.9 -> abandon two-stage.
    """
    import lightgbm as lgb

    raw: list[float] = []
    raw_cf: list[float] = []
    log_cal: list[float] = []
    for name in folds:
        X, y, infer, truth, all_users = _load_fold(name, cache_dir)
        Xf = X.fillna(0.0)
        y_arr = np.asarray(y, dtype=np.float64)
        y_bin = (y_arr > 0).astype(np.int32)
        pos = y_bin == 1
        n_pos = int(pos.sum())
        n_neg = len(y_bin) - n_pos
        scale_pos = (n_neg / max(1, n_pos)) * 1.1
        logger.info(
            "  two-stage fold %s: %d rows, %d pos, scale_pos=%.3f",
            name,
            len(Xf),
            n_pos,
            scale_pos,
        )

        clf_params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 19,
            "max_depth": 7,
            "learning_rate": 0.05,
            "min_child_samples": 70,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.75,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 1.0,
            "scale_pos_weight": scale_pos,
            "boost_from_average": True,
            "verbose": -1,
            "random_state": int(seed),
            "num_threads": _default_num_threads(),
        }
        clf = lgb.train(clf_params, lgb.Dataset(Xf, label=y_bin), num_boost_round=150)

        X_pos = Xf[pos]
        y_pos = y_arr[pos]
        val_n = max(1, int(len(X_pos) * 0.05))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_pos))
        val_idx, tr_idx = idx[:val_n], idx[val_n:]
        reg_params = {
            "objective": "huber",
            "alpha": 0.9,
            "metric": "rmse",
            "num_leaves": 41,
            "max_depth": 10,
            "learning_rate": 0.07,
            "min_child_samples": 15,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
            "boost_from_average": False,
            "verbose": -1,
            "random_state": int(seed),
            "num_threads": _default_num_threads(),
        }
        dtr = lgb.Dataset(X_pos.iloc[tr_idx], label=y_pos[tr_idx])
        dva = lgb.Dataset(X_pos.iloc[val_idx], label=y_pos[val_idx], reference=dtr)
        reg = lgb.train(
            reg_params,
            dtr,
            num_boost_round=500,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50)],
        )

        Xv = infer.reindex(all_users).fillna(0.0)
        p_buy = np.asarray(clf.predict(Xv), dtype=np.float64)
        amount = np.expm1(
            np.asarray(
                reg.predict(Xv, num_iteration=reg.best_iteration),
                dtype=np.float64,
            )
        )
        pred = p_buy * amount
        y_true = truth.reindex(all_users).fillna(0.0).values
        raw.append(rmsle(y_true, pred))
        cf = float(np.mean(y_true) / np.mean(pred)) if np.mean(pred) > 0 else 1.0
        raw_cf.append(rmsle(y_true, pred * cf))
        b = float(np.mean(np.log1p(y_true) - np.log1p(pred)))
        log_cal.append(rmsle(y_true, np.expm1(np.log1p(pred) + b)))
    return {
        "raw": float(np.mean(raw)),
        "raw_cf": float(np.mean(raw_cf)),
        "log_cal": float(np.mean(log_cal)),
    }


def run_weighted(
    cfg: dict, folds: list[str], cache_dir: Path, seed: int
) -> tuple[float, list[float]]:
    """Weighted single-stage (Phase 2): weight = clip(1+log1p(gmv), 1, 6)."""
    import lightgbm as lgb

    scores: list[float] = []
    for name in folds:
        X, y, infer, truth, all_users = _load_fold(name, cache_dir)
        Xf = X.fillna(0.0)
        y_arr = np.asarray(y, dtype=np.float64)
        w = np.clip(1.0 + y_arr, 1.0, 6.0)
        params = make_params(cfg, seed)
        model = lgb.train(
            params,
            lgb.Dataset(Xf, label=y_arr, weight=w),
            num_boost_round=int(cfg["n_estimators"]),
        )
        Xv = infer.reindex(all_users).fillna(0.0)
        pred = np.expm1(np.asarray(model.predict(Xv), dtype=np.float64))
        y_true = truth.reindex(all_users).fillna(0.0).values
        scores.append(rmsle(y_true, pred))
    return float(np.mean(scores)), scores


def run_calibrated(
    cfg: dict, folds: list[str], cache_dir: Path, seed: int
) -> tuple[float, list[float]]:
    """Single-stage + log-space bias correction (Phase 3)."""
    import lightgbm as lgb

    scores: list[float] = []
    for name in folds:
        X, y, infer, truth, all_users = _load_fold(name, cache_dir)
        Xf = X.fillna(0.0)
        y_arr = np.asarray(y, dtype=np.float64)
        params = make_params(cfg, seed)
        model = lgb.train(
            params,
            lgb.Dataset(Xf, label=y_arr),
            num_boost_round=int(cfg["n_estimators"]),
        )
        Xv = infer.reindex(all_users).fillna(0.0)
        pred = np.expm1(np.asarray(model.predict(Xv), dtype=np.float64))
        y_true = truth.reindex(all_users).fillna(0.0).values
        b = float(np.mean(np.log1p(y_true) - np.log1p(pred)))
        scores.append(rmsle(y_true, np.expm1(np.log1p(pred) + b)))
    return float(np.mean(scores)), scores


def run_hybrid(
    cfg: dict, folds: list[str], cache_dir: Path, seed: int
) -> tuple[float, list[float]]:
    """T3.8.10: OOF P(Y>0) as a feature of the base single-stage LGBM.

    The classifier's out-of-fold probability (leave-origin-out) is added to the
    base regressor's features instead of being used as a multiplier. Trees learn
    non-linear interactions with it (e.g., prob>0.8 AND late Dec -> boost).
    """
    import lightgbm as lgb

    scores: list[float] = []
    for name in folds:
        X, y, origin, infer, truth, all_users = _load_fold_with_origin(name, cache_dir)
        if origin is None:
            raise RuntimeError(
                f"fold {name} cache lacks 'origin' column - rebuild with the patched "
                "_build_fold (include_origin=True)"
            )
        Xf = X.fillna(0.0)
        y_arr = np.asarray(y, dtype=np.float64)
        y_bin = (y_arr > 0).astype(np.int32)
        oids = origin.astype("datetime64[ns]").values

        oof = hybrid.oof_purchase_prob(Xf.values, y_bin, oids, seed)
        clf_full = hybrid.fit_purchase_prob(Xf.values, y_bin, seed)
        Xf["purchase_prob"] = oof

        Xv = infer.reindex(all_users).fillna(0.0)
        Xv["purchase_prob"] = np.asarray(clf_full.predict(Xv.values), dtype=np.float64)

        params = make_params(cfg, seed)
        model = lgb.train(
            params,
            lgb.Dataset(Xf, label=y_arr),
            num_boost_round=int(cfg["n_estimators"]),
        )
        pred = np.expm1(np.asarray(model.predict(Xv), dtype=np.float64))
        y_true = truth.reindex(all_users).fillna(0.0).values
        scores.append(rmsle(y_true, pred))

        gain = dict(zip(Xf.columns, model.feature_importance("gain"), strict=False))
        pp_gain = gain.get("purchase_prob", 0.0)
        pp_rank = sum(1 for v in gain.values() if v > pp_gain) + 1
        logger.info(
            "  hybrid fold %s: purchase_prob gain=%.4g rank=%d/%d",
            name,
            pp_gain,
            pp_rank,
            len(gain),
        )
    return float(np.mean(scores)), scores


def format_results_table(results: list[ConfigResult]) -> str:
    lines = [
        "| # | n_est | leaves | lr | ff | bag | min_child | RMSLE(mean±std) | folds | pred_mean | time_s |",
        "|---|-------|--------|-----|----|-----|-----------|-----------------|-------|-----------|--------|",
    ]
    for i, r in enumerate(results, 1):
        c = r.config
        lines.append(
            f"| {i} | {c['n_estimators']} | {c['num_leaves']} | {c['learning_rate']} | "
            f"{c['feature_fraction']} | {c['bagging_fraction']} | {c['min_child_samples']} | "
            f"{r.rmsle_mean:.4f}±{r.rmsle_std:.4f} | "
            f"{','.join(f'{x:.4f}' for x in r.rmsle_folds)} | "
            f"{r.pred_mean:.2f} | {r.fit_time_sec:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/train.parquet")
    ap.add_argument("--strategy", choices=["grid", "random"], default="grid")
    ap.add_argument("--max-configs", type=int, default=24)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fold-cache-dir", type=str, default=str(FOLD_CACHE_DIR))
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--rebuild-folds", action="store_true")
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="A/B: compare lgb/xgb/cat (+ blends) on the CV folds",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="lgb,xgb,cat",
        help="comma-separated model families for --ensemble",
    )
    ap.add_argument(
        "--two-stage", action="store_true", help="T3.8.8: P(buy) classifier x amount regressor A/B"
    )
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="T3.8.9: weighted single-stage (clip(1+log1p(gmv),1,6))",
    )
    ap.add_argument(
        "--calibrated", action="store_true", help="T3.8.6: single-stage + log-space bias correction"
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="T3.8.10: OOF P(Y>0) as a feature (leave-origin-out stacking)",
    )
    ap.add_argument(
        "--wf-step",
        type=int,
        default=None,
        help="T3.8.4: walk-forward step days override (ECUP26_WALK_FORWARD_STEP_DAYS)",
    )
    ap.add_argument(
        "--wf-origins",
        type=int,
        default=None,
        help="T3.8.4: walk-forward origin count override (ECUP26_WALK_FORWARD_ORIGINS)",
    )
    args = ap.parse_args()

    if args.wf_step is not None:
        os.environ["ECUP26_WALK_FORWARD_STEP_DAYS"] = str(args.wf_step)
    if args.wf_origins is not None:
        os.environ["ECUP26_WALK_FORWARD_ORIGINS"] = str(args.wf_origins)
    cache_dir = Path(args.fold_cache_dir)
    df = _load_clean(Path(args.data))
    max_date = df["event_date"].max()

    # folds: fold i has cutoff = max - 30*(n_folds - i); truth (cutoff, cutoff+30]
    fold_names: list[str] = []
    for i in range(args.folds):
        cutoff = max_date - pd.Timedelta(days=TARGET_DAYS * (args.folds - i))
        name = chr(ord("A") + i)
        fold_names.append(name)
        if (cache_dir / f"fold_{name}_train.parquet").exists() and not args.rebuild_folds:
            logger.info("Fold %s cached (%s)", name, cutoff.date())
            continue
        _build_fold(df, cutoff, name, cache_dir)
    del df

    if args.two_stage:
        logger.info("Two-stage A/B on config %s", BASE_CONFIG)
        res = run_two_stage(BASE_CONFIG, fold_names, cache_dir, SEED)
        print("=" * 60)
        print(f"two-stage raw      RMSLE {res['raw']:.4f}")
        print(f"two-stage raw_cf   RMSLE {res['raw_cf']:.4f}")
        print(f"two-stage log_cal  RMSLE {res['log_cal']:.4f}")
        print("=" * 60)
        return 0

    if args.weighted:
        logger.info("Weighted single-stage A/B (T3.8.9)")
        mean_rms, folds_rms = run_weighted(BASE_CONFIG, fold_names, cache_dir, SEED)
        print("=" * 60)
        print(f"weighted RMSLE {mean_rms:.4f} (folds {[f'{x:.4f}' for x in folds_rms]})")
        print("=" * 60)
        return 0

    if args.calibrated:
        logger.info("Calibrated single-stage A/B (T3.8.6)")
        mean_rms, folds_rms = run_calibrated(BASE_CONFIG, fold_names, cache_dir, SEED)
        print("=" * 60)
        print(f"calibrated RMSLE {mean_rms:.4f} (folds {[f'{x:.4f}' for x in folds_rms]})")
        print("=" * 60)
        return 0

    if args.hybrid:
        logger.info("Hybrid OOF stacking A/B (T3.8.10)")
        mean_rms, folds_rms = run_hybrid(BASE_CONFIG, fold_names, cache_dir, SEED)
        print("=" * 60)
        print(f"hybrid RMSLE {mean_rms:.4f} (folds {[f'{x:.4f}' for x in folds_rms]})")
        print("=" * 60)
        return 0

    if args.ensemble:
        models = tuple(m.strip() for m in args.models.split(","))
        logger.info("Ensemble A/B on config %s, models %s", BASE_CONFIG, models)
        scores = run_ensemble_ab(BASE_CONFIG, fold_names, cache_dir, SEED, models)
        print("=" * 60)
        for key, val in sorted(scores.items(), key=lambda kv: kv[1]):
            print(f"{key:18s} RMSLE {val:.4f}")
        print("=" * 60)
        return 0

    if args.strategy == "grid":
        configs = curated_grid()[: args.max_configs]
    else:
        configs = random_configs(args.max_configs)
    logger.info("Evaluating %d configs on folds %s", len(configs), fold_names)

    results: list[ConfigResult] = []
    for cfg in configs:
        scores, pmeans, pmedians, times = [], [], [], []
        for name in fold_names:
            rms, pm, pmed, ft = _eval_fold(cfg, name, SEED, cache_dir)
            scores.append(rms)
            pmeans.append(pm)
            pmedians.append(pmed)
            times.append(ft)
        results.append(
            ConfigResult(
                config=cfg,
                rmsle_mean=float(np.mean(scores)),
                rmsle_std=float(np.std(scores)) if len(scores) > 1 else 0.0,
                rmsle_folds=scores,
                pred_mean=float(np.mean(pmeans)),
                pred_median=float(np.median(pmedians)),
                fit_time_sec=float(np.mean(times)),
            )
        )
        logger.info(
            "cfg=%s -> %.4f±%.4f (folds %s)",
            cfg,
            results[-1].rmsle_mean,
            results[-1].rmsle_std,
            [f"{x:.4f}" for x in results[-1].rmsle_folds],
        )

    results.sort(key=lambda r: r.rmsle_mean)
    table = format_results_table(results)
    print("=" * 78)
    print(table)
    print("=" * 78)

    best = results[0]
    print(f"\nBEST: {best.config}  RMSLE {best.rmsle_mean:.4f}±{best.rmsle_std:.4f}")

    rows = [
        {
            **asdict(r)["config"],
            "rmsle_mean": r.rmsle_mean,
            "rmsle_std": r.rmsle_std,
            "rmsle_folds": r.rmsle_folds,
            "pred_mean": r.pred_mean,
            "pred_median": r.pred_median,
            "fit_time_sec": r.fit_time_sec,
        }
        for r in results
    ]
    csv_path = REPO_ROOT / "docs" / "results" / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("CSV saved to %s", csv_path)

    best_json = REPO_ROOT / "docs" / "results" / "benchmark_best.json"
    best_json.write_text(json.dumps(best.config, indent=2) + "\n")
    logger.info("Best config saved to %s", best_json)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# T3.8.1 benchmark ({args.strategy}, {args.folds} folds)\n\n{table}\n")
        logger.info("Report saved to %s", out)
    return 0


def _load_fold(name: str, cache_dir: Path):
    """Load one cached walk-forward fold (train/infer/truth/users)."""
    X = pd.read_parquet(cache_dir / f"fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id")
    infer = pd.read_parquet(cache_dir / f"fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(cache_dir / f"fold_{name}_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    return X, y, infer, truth, all_users


def _load_fold_with_origin(name: str, cache_dir: Path):
    """Load a cached fold keeping the per-row walk-forward origin (T3.8.10)."""
    X = pd.read_parquet(cache_dir / f"fold_{name}_train.parquet")
    y = X.pop("y_log").astype(float)
    origin = X.pop("origin") if "origin" in X.columns else None
    X = X.set_index("user_id")
    infer = pd.read_parquet(cache_dir / f"fold_{name}_infer.parquet").set_index("user_id")
    truth = pd.read_parquet(cache_dir / f"fold_{name}_truth.parquet")["y_true"]
    all_users = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    return X, y, origin, infer, truth, all_users


def _train_model(kind: str, cfg: dict, X: pd.DataFrame, y: pd.Series, seed: int):
    """Train a single model family on a fold's walk-forward train set."""
    if kind == "lgb":
        import lightgbm as lgb

        params = make_params(cfg, seed)
        return lgb.train(
            params,
            lgb.Dataset(X.fillna(0.0), label=np.asarray(y)),
            num_boost_round=int(cfg["n_estimators"]),
        )
    if kind == "xgb":
        import xgboost as xgb

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": float(cfg["learning_rate"]),
            "subsample": float(cfg["bagging_fraction"]),
            "colsample_bytree": float(cfg["feature_fraction"]),
            "min_child_weight": float(cfg["min_child_samples"]),
            "n_jobs": _default_num_threads(),
            "seed": int(seed),
            "verbosity": 0,
            "tree_method": "hist",
        }
        return xgb.train(
            params,
            xgb.DMatrix(X.fillna(0.0), label=np.asarray(y)),
            num_boost_round=int(cfg["n_estimators"]),
        )
    if kind == "cat":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=int(cfg["n_estimators"]),
            depth=6,
            learning_rate=float(cfg["learning_rate"]),
            l2_leaf_reg=float(cfg["min_child_samples"]),
            random_seed=int(seed),
            verbose=0,
            thread_count=_default_num_threads(),
            allow_writing_files=False,
        )
        model.fit(X.fillna(0.0), y)
        return model
    raise ValueError(f"unknown model kind: {kind}")


def _predict_model(kind: str, model, Xv: pd.DataFrame) -> np.ndarray:
    if kind == "lgb":
        return np.asarray(model.predict(Xv), dtype=np.float64)
    if kind == "xgb":
        import xgboost as xgb

        return np.asarray(model.predict(xgb.DMatrix(Xv)), dtype=np.float64)
    return np.asarray(model.predict(Xv), dtype=np.float64)


def run_ensemble_ab(
    cfg: dict,
    folds: list[str],
    cache_dir: Path,
    seed: int,
    models: tuple[str, ...] = ("lgb", "xgb", "cat"),
) -> dict[str, float]:
    """A/B: per-model + blended RMSLE (mean over folds, blend in log space)."""
    per_model: dict[str, list[float]] = {m: [] for m in models}
    blends: dict[tuple[str, ...], list[float]] = {}
    for name in folds:
        X, y, infer, truth, all_users = _load_fold(name, cache_dir)
        Xv = infer.reindex(all_users).fillna(0.0)
        y_true = truth.reindex(all_users).fillna(0.0).values
        raws: dict[str, np.ndarray] = {}
        for m in models:
            logger.info("  fold %s model %s ...", name, m)
            model = _train_model(m, cfg, X, y, seed)
            raws[m] = _predict_model(m, model, Xv)
            per_model[m].append(rmsle(y_true, np.expm1(raws[m])))
        for k in (2, 3):
            for combo in itertools.combinations(models, k):
                raw = np.mean([raws[m] for m in combo], axis=0)
                blends.setdefault(combo, []).append(rmsle(y_true, np.expm1(raw)))
    result: dict[str, float] = {f"{m}": float(np.mean(v)) for m, v in per_model.items()}
    for combo, vals in blends.items():
        result["+".join(combo)] = float(np.mean(vals))
    return result


if __name__ == "__main__":
    sys.exit(main())
