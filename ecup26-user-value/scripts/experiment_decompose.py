#!/usr/bin/env python3
"""T3.13 / Г3#1: GMV decomposition = orders-count x AOV (+ gift-weight sweep).

Separate models: E[orders] (log1p 30d order count) and E[AOV | orders>0]
(log1p mean check per order). gmv_hat = expm1(orders_log) * expm1(aov_log).

Rationale (Q1+Q2 2026 press releases, public/unverified): GMV +36-37%,
orders +73-83%, frequency +53%, avg check -26%. The growth is in ORDER COUNT,
not AOV. A single GMV model mixes frequency and check; decomposing lets the
orders model carry the (larger) uplift without over-inflating zeros/checks.

Walk-forward finding (T3.13): the default 8 origins (2025-06..2026-01) contain
NO gift-season label, so the orders model under-predicts the gift window ~8x
(mean orders_hat 0.25 vs truth 2.37). Injecting the gift origin (2025-02-13,
label = fold-G window) is diluted (1/9 of rows). --gift-weights reweights the
gift-origin rows in LGBM training; the sweep finds the compromise between the
diluted model (0.25) and a gift-only overfit (9.0).

The walk-forward is cached once; the weight sweep re-fits cheaply.

Usage:
  ECUP26_ACTOR_FEATURES=1 ECUP26_CALENDAR_V2_FEATURES=1 ECUP26_EXTRA_ORIGINS=2025-02-13 \\
    uv run python scripts/experiment_decompose.py --production \\
      --cache-dir data/folds/actor3_calv2 [--task T3.13-xAOV]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.run_pipeline import (  # noqa: E402
    TARGET_DAYS,
    WALK_FORWARD_ORIGINS,
    WALK_FORWARD_STEP_DAYS,
    _available_mem_gb,
    _walk_forward_origins,
    build_features,
)
from scripts import benchmark_t3 as bench  # noqa: E402

LGB = {"n_estimators": 650, "num_leaves": 120, "learning_rate": 0.02,
       "feature_fraction": 0.6, "bagging_fraction": 0.8, "min_child_samples": 21}
GIFT_ORIGIN = "2025-02-13"
CACHE_WF = "data/cache/wf_orders_aov.parquet"


def _free_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return 0.0


def _fit(X, y, weight=None):
    return lgb.train(
        bench.make_params({**bench.BASE_CONFIG, **LGB}, bench.SEED),
        lgb.Dataset(X, label=np.asarray(y, dtype=np.float64), weight=weight),
        num_boost_round=650,
    )


def _build_walkforward(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Walk-forward X + y_orders + y_aov + origin, cached to disk."""
    if Path(CACHE_WF).exists():
        print(f"[cache hit] {CACHE_WF}")
        return pd.read_parquet(CACHE_WF)
    origins = _walk_forward_origins(ref_date, WALK_FORWARD_ORIGINS, WALK_FORWARD_STEP_DAYS)
    frames: list[pd.DataFrame] = []
    built: list[pd.Timestamp] = []
    for origin in origins:
        if _available_mem_gb() < 6.0:
            print(f"[skip origin {origin.date()}: low RAM]")
            continue
        feature_df = df[df["event_date"] <= origin]
        if feature_df.empty:
            continue
        feats = build_features(feature_df, ref_date=origin)
        if feats.empty:
            continue
        tgt = df[(df["event_date"] > origin) & (df["event_date"] <= origin + pd.Timedelta(days=TARGET_DAYS))]
        orders = tgt.groupby("user_id")["to_ord"].sum()
        gmv = tgt.groupby("user_id")["price"].sum()
        orders_s = orders.reindex(feats["user_id"]).fillna(0.0)
        feats["y_orders"] = np.log1p(orders_s).astype(float)
        aov = (gmv / orders.clip(lower=1)).replace([np.inf, -np.inf], 0.0)
        aov_s = aov.reindex(feats["user_id"]).fillna(0.0)
        feats["y_aov"] = np.log1p(aov_s).astype(float)
        frames.append(feats)
        built.append(origin)
        print(f"origin {origin.date()}: {len(feats)} users, {feats.shape[1]-2} feats")
    X = pd.concat(frames, ignore_index=True)
    X["origin"] = np.repeat(np.asarray(built, dtype="datetime64[ns]"), [len(f) for f in frames])
    Path(CACHE_WF).parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(CACHE_WF)
    print(f"[cache written] {CACHE_WF} ({X.shape})")
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production", action="store_true")
    ap.add_argument("--cache-dir", type=str, default="data/folds/actor3_calv2")
    ap.add_argument("--gift-weights", type=str, default="1,2,4,8",
                    help="comma list of gift-origin row weights to sweep")
    ap.add_argument("--task", type=str, default="T3.13-xAOV")
    args = ap.parse_args()

    X = pd.read_parquet(f"{args.cache_dir}/fold_B_train.parquet")
    X.pop("y_log")
    if "origin" in X.columns:
        X = X.drop(columns=["origin"])
    X = X.set_index("user_id").fillna(0.0)
    print(f"fold B sanity: train {X.shape}")

    if not args.production:
        return 0

    df = bench._load_clean(Path("data/train.parquet"))
    ref = df["event_date"].max()
    wf = _build_walkforward(df, ref)
    y_orders = wf.pop("y_orders").astype(float)
    y_aov = wf.pop("y_aov").astype(float)
    origin_arr = wf.pop("origin").values.astype("datetime64[ns]")
    feats = [c for c in wf.columns if c not in ("user_id", "origin")]
    if _free_ram_gb() < 5.0:
        raise SystemExit(f"ABORT: only {_free_ram_gb():.1f} GB free RAM")

    infer_f = build_features(df, ref_date=ref)
    XvF = infer_f.set_index("user_id").reindex(
        pd.read_csv("data/sample_submit.csv")["user_id"]).fillna(0.0)
    Xf_all = wf[feats].fillna(0.0)
    pos = y_aov > 0
    gift_mask = origin_arr == np.datetime64(GIFT_ORIGIN)
    print(f"wf rows={len(wf)}, gift rows={int(gift_mask.sum())} ({gift_mask.mean():.1%}), "
          f"aov positive={int(pos.sum())}")

    date_s = datetime.datetime.now().strftime("%Y-%m-%d")
    time_s = datetime.datetime.now().strftime("%H%M%S")
    for gw in [float(x) for x in args.gift_weights.split(",")]:
        w = np.where(gift_mask, gw, 1.0)
        m_orders = _fit(Xf_all, y_orders, w)
        m_aov = _fit(Xf_all[pos], y_aov[pos], w[pos])
        orders_log = np.asarray(m_orders.predict(XvF[feats]), dtype=np.float64)
        aov_log = np.asarray(m_aov.predict(XvF[feats]), dtype=np.float64)
        orders_hat = np.expm1(orders_log)
        aov_hat = np.expm1(aov_log)
        pred = np.clip(orders_hat * aov_hat, 0.0, None)
        out_name = f"submissions/submit_blend_retuned_{date_s}_{time_s}_{args.task}_gw{gw}.csv"
        out = pd.DataFrame({"user_id": XvF.index, "predict": pred})
        out.to_csv(out_name, index=False)
        print(f"gw={gw}: mean orders_hat={orders_hat.mean():.3f}, aov_hat={aov_hat.mean():.2f}, "
              f"gmv mean={pred.mean():.2f}, out={out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

