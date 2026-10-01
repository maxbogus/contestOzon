"""T3.9.13: blend tuned LGBM + XGB + CatBoost -> fold B, then production submit.

CatBoost tuned (9/0.03/10/1.0/700) beats LGBM on fold B (1.6849 vs 1.6857).
Blend weights are fit on fold B (log-space), then the same models are trained
on the full walk-forward (up to 2026-02-13) for the production submission.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import benchmark_t3 as bench  # noqa: E402


def _free_ram_gb() -> float:
    """Available RAM (GB) from /proc/meminfo MemAvailable (OOM guard, T3.10.18)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return 0.0

CACHE = "data/folds/w180"
LGB = {"n_estimators": 650, "num_leaves": 63, "learning_rate": 0.02,
       "feature_fraction": 0.6, "bagging_fraction": 0.8, "min_child_samples": 21}
XGB = {"n_estimators": 550, "learning_rate": 0.02, "max_depth": 6,
       "min_child_weight": 20, "colsample_bytree": 0.66}
CAT = {"iterations": 900, "learning_rate": 0.03, "depth": 9,
       "min_data_in_leaf": 10, "l2_leaf_reg": 1.0}


def _fits(Xf, y, weight=None):
    pl = lgb.train(bench.make_params({**bench.BASE_CONFIG, **LGB}, bench.SEED),
                   lgb.Dataset(Xf, label=np.asarray(y, dtype=np.float64),
                               weight=weight),
                   num_boost_round=650)
    px = xgb.XGBRegressor(n_estimators=550, learning_rate=0.02, max_depth=5,
                          min_child_weight=20, subsample=0.8, colsample_bytree=0.66,
                          reg_lambda=1.0, tree_method="hist", random_state=bench.SEED,
                          n_jobs=bench._default_num_threads())
    px.fit(Xf, y, sample_weight=weight)
    pc = CatBoostRegressor(iterations=900, learning_rate=0.03, depth=7,
                           min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
                           random_seed=bench.SEED, thread_count=bench._default_num_threads(),
                           verbose=0)
    pc.fit(Xf, y, sample_weight=weight)
    return pl, px, pc


def _fit_bgndb(txn: pd.DataFrame, ref_date):
    """Fit BG/NBD on transactions <= ref_date (lifetimes 0.11.3 API)."""
    from lifetimes import BetaGeoFitter
    from lifetimes.utils import ConvergenceError, summary_data_from_transaction_data

    txn = txn.copy()
    txn["event_date"] = pd.to_datetime(txn["event_date"])
    past = txn[txn["event_date"] <= pd.Timestamp(ref_date)]
    summary = summary_data_from_transaction_data(
        past[["user_id", "event_date", "price"]],
        customer_id_col="user_id", datetime_col="event_date",
        monetary_value_col="price",
    )
    last_err: Exception | None = None
    for penalizer in (0.0, 0.01, 0.1, 1.0):
        try:
            bgf = BetaGeoFitter(penalizer_coef=penalizer)
            bgf.fit(summary["frequency"], summary["recency"], summary["T"], maxiter=8000)
            return bgf
        except ConvergenceError as e:
            last_err = e
    raise last_err


def _add_btyd_exp(bgf, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add btyd_exp_30d (expected 30d purchases) from frequency/recency/T."""
    f = feature_df["frequency"].astype(float).clip(lower=0.0)
    r = feature_df["recency"].astype(float).clip(lower=0.0)
    t = feature_df["T"].astype(float).clip(lower=0.0)
    feature_df["btyd_exp_30d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t + 30.0, f, r, t)
    return feature_df


_POLICY = None


def _feature_policy() -> dict:
    """Load the T3.18 dedup/null-importance drop list + trend feature."""
    global _POLICY
    if _POLICY is None:
        pol = json.loads(Path("scripts/feature_policy.json").read_text())
        _POLICY = set(pol["drop"])
    return _POLICY


def _apply_feature_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop noise/duplicate columns and add the 90d-30d order trend."""
    drop = _feature_policy()
    drop = [c for c in drop if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    if {"to_ord_90d", "to_ord_30d"} <= set(df.columns):
        df["__trend_90_30"] = df["to_ord_90d"] - df["to_ord_30d"]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production", action="store_true")
    ap.add_argument("--cache-dir", type=str, default="data/folds/actor",
                    help="fold cache for the blend-weight fit")
    ap.add_argument("--btyd", action="store_true",
                    help="add btyd_exp_30d (T3.10.20; default OFF - it degraded on LB)")
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="log-space shrink multiplier on the blended prediction "
                         "(T3.10.13: c=0.95 fixes the zero over-prediction)")
    ap.add_argument("--task", type=str, default="",
                    help="ledger task code appended to the filename "
                         "(e.g. C1.A2-control / C1.A2-shrink / M1.A2-segshrink)")
    ap.add_argument("--gift-weight", type=float, default=1.0,
                    help="reweight the injected gift origin (2025-02-13) rows in "
                         "walk-forward training (T3.13: default 8 origins lack a "
                         "gift-season label -> model under-predicts the gift window)")
    args = ap.parse_args()

    # --- fold B blend ---
    X = pd.read_parquet(f"{args.cache_dir}/fold_B_train.parquet")
    y = X.pop("y_log").astype(float)
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id").fillna(0.0)
    X = _apply_feature_policy(X)
    infer = pd.read_parquet(f"{args.cache_dir}/fold_B_infer.parquet").set_index("user_id")
    infer = _apply_feature_policy(infer)
    truth = pd.read_parquet(f"{args.cache_dir}/fold_B_truth.parquet")["y_true"]
    au = np.sort(np.unique(np.concatenate([infer.index.values, truth.index.values])))
    Xv = infer.reindex(au).fillna(0.0)
    yt = np.log1p(truth.reindex(au).fillna(0.0).values)
    ml, mx, mc = _fits(X, y)
    pl, px, pc = (np.asarray(ml.predict(Xv)), np.asarray(mx.predict(Xv)),
                  np.asarray(mc.predict(Xv)))

    def rmsle(p):
        return float(np.sqrt(np.mean((p - yt) ** 2)))

    print(f"fold B: LGBM {rmsle(pl):.4f} | XGB {rmsle(px):.4f} | Cat {rmsle(pc):.4f}")
    best_w, best_r = None, 1e9
    for wc in np.arange(0.0, 1.01, 0.1):
        for wx in np.arange(0.0, 0.51, 0.1):
            wl = 1.0 - wc - wx
            if wl < 0:
                continue
            r = rmsle(wl * pl + wx * px + wc * pc)
            if r < best_r:
                best_r, best_w = r, (wl, wx, wc)
    print(f"best blend fold B: {best_r:.4f} (w_l,w_x,w_c)={[round(w,2) for w in best_w]}")

    if not args.production:
        return 0

    # --- production submission (full walk-forward up to 2026-02-13) ---
    from ecup26_user_value.run_pipeline import (  # noqa: E402
        _build_walk_forward_training,
        build_features,
    )

    df = bench._load_clean(Path("data/train.parquet"))
    ref = df["event_date"].max()
    Xf, yf = _build_walk_forward_training(df, ref, include_origin=True)
    origin_arr = Xf.pop("origin")
    Xf = _apply_feature_policy(Xf)
    gift_w = None
    if args.gift_weight != 1.0:
        gift_w = np.where(
            origin_arr == pd.Timestamp("2025-02-13"), args.gift_weight, 1.0
        )
        print(f"gift origin reweight: {float((gift_w > 1).mean()):.1%} rows x{args.gift_weight}")
    infer_f = build_features(df, ref_date=ref)
    infer_f = _apply_feature_policy(infer_f)
    if args.btyd:
        bgf = _fit_bgndb(df, ref)
        Xf = _add_btyd_exp(bgf, Xf)
        infer_f = _add_btyd_exp(bgf, infer_f)
    feats = [c for c in Xf.columns if c not in ("user_id", "origin")]
    free_gb = _free_ram_gb()
    if free_gb < 5.0:
        raise SystemExit(
            f"ABORT: only {free_gb:.1f} GB free RAM before production fit "
            f"(threads={bench._default_num_threads()}). Re-run with "
            f"ECUP26_NUM_THREADS=4 to lower peak memory."
        )
    print(f"production fit: free RAM {free_gb:.1f} GB, threads={bench._default_num_threads()}")

    ml, mx, mc = _fits(Xf[feats].fillna(0.0), yf, weight=gift_w)
    XvF = infer_f.set_index("user_id").reindex(
        pd.read_csv("data/sample_submit.csv")["user_id"]).fillna(0.0)
    pl, px, pc = (np.asarray(ml.predict(XvF)), np.asarray(mx.predict(XvF)),
                  np.asarray(mc.predict(XvF)))
    wl, wx, wc = best_w
    blend_log = wl * pl + wx * px + wc * pc
    if args.shrink != 1.0:
        blend_log = blend_log * args.shrink
    pred = np.clip(np.expm1(blend_log), 0.0, None)
    date_s = datetime.datetime.now().strftime("%Y-%m-%d")
    time_s = datetime.datetime.now().strftime("%H%M%S")
    tag = "btyd_retuned" if args.btyd else "actor_retuned"
    if args.shrink != 1.0:
        tag = f"{tag}_shrink{args.shrink}"
    task_s = f"_{args.task}" if args.task else ""
    out_name = f"submissions/submit_blend_{tag}_{date_s}_{time_s}{task_s}.csv"
    out = pd.DataFrame({"user_id": XvF.index, "predict": pred})
    out.to_csv(out_name, index=False)
    print(f"production blend: {out.shape}, mean {out.predict.mean():.2f}, out={out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
