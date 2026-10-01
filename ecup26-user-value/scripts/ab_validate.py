#!/usr/bin/env python3
"""Repeated bootstrap A/B validation for E-CUP T3 (user-value).

New noise criterion (docs/EXPERIMENTS.md): the LB is compressed into ~0.0106
RMSLE (#1 1.6459 vs #78 1.6565), so single-fold ±0.001 deltas that used to be
"noise" are now the whole game. Decisions are made by SIGN CONSISTENCY across
repeated shuffled folds with replacement, not by a point delta.

Per seed k:
  1. bootstrap-resample the fold's training rows WITH replacement (seeded),
  2. train method A and B on the SAME resample,
  3. score both on the fold's full infer+truth set,
  4. record delta_k = rmsle(A) - rmsle(B)   (log-space RMSLE).

Report: mean±std of delta, P(A beats B), 95% bootstrap CI, and a decision.

Per .clinerules: R1 open-source, R6 seeded/reproducible, R8 dev-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import benchmark_t3 as bench  # noqa: E402

CACHE = Path("data/folds/w180")

#: Refined production config (T3.9.7): the reference baseline.
REFINED = {
    "n_estimators": 550,
    "num_leaves": 97,
    "learning_rate": 0.02,
    "feature_fraction": 0.66,
    "bagging_fraction": 0.8,
    "min_child_samples": 21,
}


def _train_lgb(X, y, cfg, seed, weight=None):
    """Train a LightGBM regression on log-space label (or weighted variant)."""
    import lightgbm as lgb

    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    ds = lgb.Dataset(X.fillna(0.0), label=np.asarray(y, dtype=np.float64), weight=w)
    return lgb.train(
        bench.make_params({**bench.BASE_CONFIG, **cfg}, seed),
        ds,
        num_boost_round=int(cfg["n_estimators"]),
    )


def _train_clf(X, y_bin, seed):
    """Binary P(y>0) classifier (scale_pos_weight = neg/pos * 1.1)."""
    import lightgbm as lgb

    n_pos = int(np.asarray(y_bin).sum())
    n_neg = len(y_bin) - n_pos
    params = {
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
        "scale_pos_weight": (n_neg / max(1, n_pos)) * 1.1,
        "boost_from_average": True,
        "verbose": -1,
        "random_state": int(seed),
        "num_threads": bench._default_num_threads(),
    }
    return lgb.train(params, lgb.Dataset(X.fillna(0.0), label=y_bin), num_boost_round=150)


# ---- methods: (X, y_log, Xv, seed, t) -> pred_log (log1p-space) ----


def m_baseline(X, y, Xv, seed, **_):
    """Single-stage LGBM on log1p(30d GMV) — current production."""
    model = _train_lgb(X, y, REFINED, seed)
    return np.asarray(model.predict(Xv), dtype=np.float64)


def m_hurdle(X, y, Xv, seed, **_):
    """Log-space hurdle: P(y>0) * E[log1p(y) | y>0].

    Unlike the rejected raw-space two-stage (p_buy * amount, 2.02), this
    multiplies in LOG space, which is the RMSLE-optimal combination.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    y_bin = (y_arr > 0).astype(np.int32)
    clf = _train_clf(X, y_bin, seed)
    pos = y_bin == 1
    reg = _train_lgb(X[pos], y_arr[pos], REFINED, seed)
    p1 = np.asarray(clf.predict(Xv), dtype=np.float64)
    mu = np.asarray(reg.predict(Xv), dtype=np.float64)
    return p1 * mu


def m_zero_threshold(X, y, Xv, seed, t=0.5, **_):
    """Baseline + hard-zero in LOG space where P(y>0) < t."""
    model = _train_lgb(X, y, REFINED, seed)
    y_bin = (np.asarray(y) > 0).astype(np.int32)
    clf = _train_clf(X, y_bin, seed)
    p1 = np.asarray(clf.predict(Xv), dtype=np.float64)
    pred_log = np.asarray(model.predict(Xv), dtype=np.float64)
    return np.where(p1 < t, 0.0, pred_log)



def m_cat_lgb(X, y, Xv, seed, **_):
    """T3.10.5 E1: CatBoost on log1p label (T3.9.13 tuned config)."""
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=700, learning_rate=0.03, depth=9,
        min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
        random_seed=int(seed), thread_count=bench._default_num_threads(),
        verbose=0, allow_writing_files=False,
    )
    m.fit(X.fillna(0.0), np.asarray(y, dtype=np.float64))
    return np.asarray(m.predict(Xv), dtype=np.float64)


def m_cat_tweedie(X, y, Xv, seed, **_):
    """T3.10.5 E2: CatBoost Tweedie on RAW 30d GMV (diagnostic).

    Expectation: WORSE than log1p L2 (raw-space zero-inflation loss optimizes
    the wrong thing — see docs/EXPERIMENTS.md section 5).
    """
    from catboost import CatBoostRegressor

    y_raw = np.expm1(np.asarray(y, dtype=np.float64))
    m = CatBoostRegressor(
        iterations=700, learning_rate=0.03, depth=9,
        min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
        loss_function="Tweedie:variance_power=1.5", random_seed=int(seed),
        thread_count=bench._default_num_threads(), verbose=0,
        allow_writing_files=False,
    )
    m.fit(X.fillna(0.0), y_raw)
    pred_raw = np.asarray(m.predict(Xv), dtype=np.float64)
    return np.log1p(np.maximum(pred_raw, 0.0))




def _train_quantile(X, y, seed, alpha, n_est=300):
    import lightgbm as lgb

    cfg = {**REFINED, "n_estimators": n_est}
    p = bench.make_params({**bench.BASE_CONFIG, **cfg}, seed)
    p["objective"] = "quantile"
    p["alpha"] = float(alpha)
    return lgb.train(p, lgb.Dataset(X.fillna(0.0), label=np.asarray(y, dtype=np.float64)),
                     num_boost_round=n_est)


def m_quantile(X, y, Xv, seed, alpha=0.75, **_):
    """T3.10.6 E1: single quantile regression in log-space."""
    m = _train_quantile(X, y, seed, alpha)
    return np.asarray(m.predict(Xv), dtype=np.float64)


def m_quantile_blend(X, y, Xv, seed, **_):
    """T3.10.6 E2: blend of 0.5/0.75/0.9 quantiles (mean in log-space)."""
    preds = []
    for a in (0.5, 0.75, 0.9):
        m = _train_quantile(X, y, seed, a)
        preds.append(np.asarray(m.predict(Xv), dtype=np.float64))
    return np.mean(preds, axis=0)


def _user_folds(user_ids, n_folds: int) -> np.ndarray:
    """Deterministic per-user OOF fold assignment (same user always same fold)."""
    import pandas as _pd

    hashed = _pd.util.hash_pandas_object(_pd.Series(np.asarray(user_ids)), index=False)
    return (hashed % int(n_folds)).values.astype(int)


def m_quantile_spread_feat(X, y, Xv, seed, n_est=300, n_folds=3, **_):
    """T3.10.22: quantile-spread (q0.9-q0.5, q0.75-q0.5) as 2 extra features.

    Quantile regression as a PREDICTOR was rejected (2.16, quantile != mean),
    but the SPREAD of quantiles is a volatility signal, valid as a feature:
    a user with a large spread is a volatile buyer. Train-side spreads use
    user-hash OOF (n_folds) so the main LGBM never sees in-sample quantile
    predictions; the infer side uses quantile models fit on all train rows.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    folds = _user_folds(X.index.values, n_folds)
    q: dict[float, np.ndarray] = {a: np.empty(len(X), dtype=np.float64) for a in (0.5, 0.75, 0.9)}
    for f in range(int(n_folds)):
        tr = folds != f
        va = folds == f
        for a in q:
            m = _train_quantile(X[tr], y_arr[tr], seed + f, a, n_est)
            q[a][va] = np.asarray(m.predict(X[va].fillna(0.0)), dtype=np.float64)
    qv: dict[float, np.ndarray] = {a: np.empty(len(Xv), dtype=np.float64) for a in (0.5, 0.75, 0.9)}
    for a in qv:
        m = _train_quantile(X, y_arr, seed, a, n_est)
        qv[a] = np.asarray(m.predict(Xv.fillna(0.0)), dtype=np.float64)

    Xa = X.copy()
    Xa["q_spread_90_50"] = q[0.9] - q[0.5]
    Xa["q_spread_75_50"] = q[0.75] - q[0.5]
    Xva = Xv.copy()
    Xva["q_spread_90_50"] = qv[0.9] - qv[0.5]
    Xva["q_spread_75_50"] = qv[0.75] - qv[0.5]
    model = _train_lgb(Xa, y, REFINED, seed)
    return np.asarray(model.predict(Xva), dtype=np.float64)


def m_knn_feat(X, y, Xv, seed, k=25, **_):
    """T3.10.7 (fast v3): kNN neighbor-mean of log1p(GMV) as an extra feature.

    v3 speedups (v1 brute-force cosine O(n^2 d) and v2 KDTree-on-1.9M were both
    too slow >15 min): PCA->10 dims, KDTree built on a 150k REFERENCE subsample,
    train-side feature computed on a 400k subset (others get the global mean),
    infer-side on all users. Neighbor means are a smoothed-label feature - a
    probe only; expected value is low given the feature plateau (all feature
    additions were TIE/noise).
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=np.float64)
    ref = rng.choice(len(X), size=min(len(X), 150_000), replace=False)
    pca = PCA(n_components=10, random_state=int(seed))
    pca.fit(np.asarray(X.iloc[ref].fillna(0.0), dtype=np.float32))
    Z = pca.transform(np.asarray(X.fillna(0.0), dtype=np.float32)).astype(np.float32)
    Zv = pca.transform(np.asarray(Xv.fillna(0.0), dtype=np.float32)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="kd_tree")
    nn.fit(Z[ref])

    idxv = nn.kneighbors(Zv, return_distance=False)
    nb_v = y_arr[ref][idxv].mean(axis=1)

    tsub = rng.choice(len(X), size=min(len(X), 400_000), replace=False)
    idx_t = nn.kneighbors(Z[tsub], return_distance=False)
    nb_t = y_arr[ref][idx_t].mean(axis=1)
    nb_mean = np.full(len(X), y_arr.mean())
    nb_mean[tsub] = nb_t

    Xa = X.copy()
    Xa["knn_y"] = nb_mean
    Xva = Xv.copy()
    Xva["knn_y"] = nb_v
    model = _train_lgb(Xa, y, REFINED, seed)
    return np.asarray(model.predict(Xva), dtype=np.float64)

def m_ordinal(X, y, Xv, seed, **_):
    """T3.10.8: ordinal bucket classifier + within-bucket regressors.

    Buckets in log-space: [0,10) [10,100) [100,1000) [1000,inf).
    pred_log = sum_b P(bucket=b) * mu_b(Xv) with mu_0 = 0.
    """
    import lightgbm as lgb

    y_arr = np.asarray(y, dtype=np.float64)
    edges = np.array([0.0, np.log1p(10.0), np.log1p(100.0)])
    bins = np.digitize(y_arr, edges)  # 0..3 -> 4 buckets: [0,10) [10,100) [100,1000) [1000,inf)
    p = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 4,
        "num_leaves": 19,
        "learning_rate": 0.05,
        "min_child_samples": 70,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": int(seed),
        "num_threads": bench._default_num_threads(),
    }
    clf = lgb.train(p, lgb.Dataset(X.fillna(0.0), label=bins), num_boost_round=150)
    prob = np.asarray(clf.predict(Xv), dtype=np.float64)  # (n, 4)
    mus = [np.zeros(len(Xv))]
    for b in range(1, 4):
        mb = _train_lgb(X[bins == b], y_arr[bins == b], REFINED, seed)
        mus.append(np.asarray(mb.predict(Xv), dtype=np.float64))
    return np.sum(prob * np.array(mus).T, axis=1)




def m_blend(X, y, Xv, seed, **_):
    """T3.9.13 production blend: 0.3*LGBM + 0.7*CatBoost (log-space)."""
    from catboost import CatBoostRegressor

    pl = np.asarray(_train_lgb(X, y, REFINED, seed).predict(Xv), dtype=np.float64)
    c = CatBoostRegressor(
        iterations=700, learning_rate=0.03, depth=9,
        min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
        random_seed=int(seed), thread_count=bench._default_num_threads(),
        verbose=0, allow_writing_files=False,
    )
    c.fit(X.fillna(0.0), np.asarray(y, dtype=np.float64))
    pc = np.asarray(c.predict(Xv), dtype=np.float64)
    return 0.3 * pl + 0.7 * pc




def m_shrink_blend(X, y, Xv, seed, c=0.95, **_):
    """T3.10.13: production blend + log-space shrink by c.

    Zeros (46%% of users) are over-predicted (pred_log~1.42, bias -1.42) and
    carry ~50%% of error^2; positives are near-calibrated. A multiplicative
    log-space shrink c<1 compresses toward 0, trading a little positive-side
    error for a lot of zero-side error. Point estimate on fold B: c=0.95 gives
    -0.0095. MUST pass sign-consistency + fold-A cross-check before use.
    """
    from catboost import CatBoostRegressor

    pl = np.asarray(_train_lgb(X, y, REFINED, seed).predict(Xv), dtype=np.float64)
    cc = CatBoostRegressor(
        iterations=700, learning_rate=0.03, depth=9,
        min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
        random_seed=int(seed), thread_count=bench._default_num_threads(),
        verbose=0, allow_writing_files=False,
    )
    cc.fit(X.fillna(0.0), np.asarray(y, dtype=np.float64))
    pc = np.asarray(cc.predict(Xv), dtype=np.float64)
    return float(c) * (0.3 * pl + 0.7 * pc)




def m_whale_boost(X, y, Xv, seed, beta=1.2, w_frac=0.05, **_):
    """C5.A2: post-hoc multiplicative boost of the whale segment (top w_frac by pred).

    Whales (top-5% by true GMV) are under-predicted (bias +1.74 on fold B,
    +2.34 on fold G, gap_analysis 2026-08-20). Unlike whale_copy (duplicates
    rows -> distorts calibration, REJECTED), this boosts ONLY the predicted-whale
    predictions post-hoc. No truth leak: the whale mask uses pred_log quantile.
    """
    pred_log = np.asarray(_train_lgb(X, y, REFINED, seed).predict(Xv), dtype=np.float64)
    thr = np.quantile(pred_log, 1.0 - w_frac)
    whale = pred_log >= thr
    pred_log = pred_log.copy()
    pred_log[whale] *= float(beta)
    return pred_log


def m_shrink_except_whales(X, y, Xv, seed, c=0.95, beta=1.0, w_frac=0.05, **_):
    """M1.A1: log-space shrink for NON-whales + whale keep/boost (segmental).

    C1.A3 decomposition on fold G (2026-08-20): global shrink fixes zeros
    (bias -1.736 -> -1.649) but hurts whales (+2.34 -> +2.55); the LB loss
    (1.6702) came from the whale damage on the target balance. This applies
    the shrink only to the non-whale segment and keeps (beta=1.0) or boosts
    (beta>1) the whale segment - the segmental correction.
    """
    pred_log = np.asarray(_train_lgb(X, y, REFINED, seed).predict(Xv), dtype=np.float64)
    thr = np.quantile(pred_log, 1.0 - w_frac)
    whale = pred_log >= thr
    pred_log = pred_log.copy()
    pred_log[~whale] *= float(c)
    pred_log[whale] *= float(beta)
    return pred_log


def m_whale_copy(X, y, Xv, seed, w_frac=0.05, n_copies=3, **_):
    """T3.10.19: oversample the whale rows (top w_frac users by label sum).

    Duplicates their training rows (n_copies-1 extra times) so the trees see
    whales more. Different from the REJECTED weight=1+log1p(gmv) (which
    reweighted ALL rows); this only amplifies the top ~5% segment, targeting
    the known whale under-prediction (bias +0.84 on fold B). The target is
    spike-like (per the shrink LB regression) - whales matter more there.
    """
    import pandas as _pd

    y_arr = np.asarray(y, dtype=np.float64)
    u = np.asarray(X.index.values)
    s = _pd.Series(y_arr).groupby(u).sum()
    thr = s.quantile(1.0 - float(w_frac))
    whales = set(s[s >= thr].index.tolist())
    mask = np.array([uid in whales for uid in u], dtype=bool)
    n_extra = int(n_copies) - 1
    if mask.sum() == 0 or mask.sum() == len(mask) or n_extra < 1:
        return np.asarray(_train_lgb(X, y, REFINED, seed).predict(Xv), dtype=np.float64)
    Xa = _pd.concat([X] + [X[mask]] * n_extra)
    ya = np.concatenate([y_arr] + [y_arr[mask]] * n_extra)
    model = _train_lgb(Xa, ya, REFINED, seed)
    return np.asarray(model.predict(Xv), dtype=np.float64)




def _oof_group_mean(y, groups, folds, global_mean=None) -> np.ndarray:
    """Leak-free OOF mean of y per group (fold-f rows use means from other folds)."""
    y_arr = np.asarray(y, dtype=np.float64)
    default = float(global_mean if global_mean is not None else y_arr.mean())
    out = np.full(len(y_arr), default)
    for f in np.unique(folds):
        tr = folds != f
        va = folds == f
        for c in np.unique(groups):
            sel = va & (groups == c)
            if sel.sum() == 0:
                continue
            src = tr & (groups == c)
            if src.sum() > 0:
                out[sel] = float(y_arr[src].mean())
    return out


def m_target_encode_segments(X, y, Xv, seed, k_clusters=8, n_folds=3, **_):
    """T3.10.21: OOF mean-y per k-means segment as a target-encoding feature.

    Smoothed-label segment feature (kNN-analog T3.10.7 and cluster-as-feature
    T3.10.14 were rejected; this groups by segment with leak-free user-hash OOF).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    km, Z, Zv = _kmeans_setup(X, Xv, seed, k_clusters)
    labs = km.predict(Z)
    labs_v = km.predict(Zv)
    folds = _user_folds(X.index.values, n_folds)
    oof = _oof_group_mean(y_arr, labs, folds)
    full_means = np.array([float(y_arr[labs == c].mean()) for c in range(int(k_clusters))])
    oof_v = full_means[labs_v]

    Xa = X.copy()
    Xa["seg_mean_y"] = oof
    Xva = Xv.copy()
    Xva["seg_mean_y"] = oof_v
    model = _train_lgb(Xa, y, REFINED, seed)
    return np.asarray(model.predict(Xva), dtype=np.float64)


def _kmeans_setup(X, Xv, seed, k_clusters, n_pca=10, n_sub=200_000):
    """Shared K-means setup: PCA-10 on a subsample, KMeans on the same subsample."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    Xf = np.asarray(X.fillna(0.0), dtype=np.float32)
    Xvf = np.asarray(Xv.fillna(0.0), dtype=np.float32)
    sub = rng.choice(len(Xf), size=min(len(Xf), n_sub), replace=False)
    pca = PCA(n_components=n_pca, random_state=int(seed))
    pca.fit(Xf[sub])
    Z = pca.transform(Xf).astype(np.float32)
    Zv = pca.transform(Xvf).astype(np.float32)
    km = KMeans(n_clusters=int(k_clusters), random_state=int(seed), n_init=5)
    km.fit(Z[sub])
    return km, Z, Zv


def m_kmeans_cluster(X, y, Xv, seed, k_clusters=4, **_):
    """T3.10.14 E1: K-means cluster id as a feature (segmentation-aware LGBM)."""
    km, Z, Zv = _kmeans_setup(X, Xv, seed, k_clusters)
    Xa = X.copy()
    Xa["cluster"] = km.predict(Z).astype(np.int32)
    Xva = Xv.copy()
    Xva["cluster"] = km.predict(Zv).astype(np.int32)
    model = _train_lgb(Xa, y, REFINED, seed)
    return np.asarray(model.predict(Xva), dtype=np.float64)


def m_kmeans_moe(X, y, Xv, seed, k_clusters=4, **_):
    """T3.10.14 E2: Mixture of Experts - per-cluster LGBM + soft weighting.

    Soft weights: exp(-dist_to_center / temp) normalized across clusters.
    """
    km, Z, Zv = _kmeans_setup(X, Xv, seed, k_clusters)
    labels = km.predict(Z)
    preds = np.zeros((len(Xv), int(k_clusters)))
    for c in range(int(k_clusters)):
        mask = labels == c
        if mask.sum() < 200:
            preds[:, c] = np.mean(y[mask]) if mask.sum() else 0.0
            continue
        m = _train_lgb(X[mask], np.asarray(y)[mask], REFINED, seed)
        preds[:, c] = np.asarray(m.predict(Xv), dtype=np.float64)
    # soft weights from distances (temperature 0.5)
    d = km.transform(Zv)
    d = d - d.min(axis=1, keepdims=True)  # softmax stability
    w = np.exp(-np.maximum(d, 0.0) / 0.5)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
    return np.sum(w * preds, axis=1)




def m_blend_retuned(X, y, Xv, seed, **_):
    """Retuned production blend (2026-08-19 overnight coordinate search):
    LGBM 650/120/0.02/0.6/0.8/21 + CatBoost 900/0.03/9/10/1.0, weights 0.45/0.55.
    """
    from catboost import CatBoostRegressor

    lgb_cfg = {"n_estimators": 650, "num_leaves": 120, "learning_rate": 0.02,
               "feature_fraction": 0.6, "bagging_fraction": 0.8, "min_child_samples": 21}
    pl = np.asarray(_train_lgb(X, y, lgb_cfg, seed).predict(Xv), dtype=np.float64)
    c = CatBoostRegressor(iterations=900, learning_rate=0.03, depth=9,
                          min_data_in_leaf=10, l2_leaf_reg=1.0, subsample=0.8,
                          random_seed=int(seed), thread_count=bench._default_num_threads(),
                          verbose=0, allow_writing_files=False)
    c.fit(X.fillna(0.0), np.asarray(y, dtype=np.float64))
    pc = np.asarray(c.predict(Xv), dtype=np.float64)
    return 0.45 * pl + 0.55 * pc


METHODS = {"baseline": m_baseline, "hurdle": m_hurdle, "zero_threshold": m_zero_threshold,
           "cat_lgb": m_cat_lgb, "cat_tweedie": m_cat_tweedie,
           "quantile": m_quantile, "quantile_blend": m_quantile_blend,
           "knn_feat": m_knn_feat, "ordinal": m_ordinal, "blend": m_blend,
           "shrink_blend": m_shrink_blend, "whale_copy": m_whale_copy,
           "whale_boost": m_whale_boost, "shrink_except_whales": m_shrink_except_whales,
           "kmeans_cluster": m_kmeans_cluster, "kmeans_moe": m_kmeans_moe,
           "quantile_spread_feat": m_quantile_spread_feat,
           "target_encode_segments": m_target_encode_segments,
           "blend_retuned": m_blend_retuned}


def _load(name, cache_dir=CACHE):
    X, y, infer, truth, all_users = bench._load_fold(name, cache_dir)
    Xv = infer.reindex(all_users).fillna(0.0)
    yt_log = np.log1p(truth.reindex(all_users).fillna(0.0).values)
    return X, np.asarray(y, dtype=np.float64), Xv, yt_log


def rmsle_log(pred_log, yt_log):
    return float(np.sqrt(np.mean((np.asarray(pred_log) - yt_log) ** 2)))


def _report(deltas, a_name, b_name):
    """Print mean/std/P(beats)/CI and the sign-consistency verdict."""
    p_a_beats = float(np.mean(deltas < 0))
    print("=" * 60)
    print(f"mean_delta(A-B) = {deltas.mean():+.6f}")
    print(f"std_delta       = {deltas.std():.6f}")
    print(f"P(A beats B)    = {p_a_beats:.2f}  ({int((deltas < 0).sum())}/{len(deltas)} draws)")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"95% CI          = [{lo:+.6f}, {hi:+.6f}]")
    if p_a_beats >= 0.90 and deltas.mean() < 0:
        verdict = "A WINS"
    elif p_a_beats <= 0.10 and deltas.mean() > 0:
        verdict = "B WINS"
    else:
        verdict = "TIE (more draws needed)"
    print(f"DECISION: {verdict}")


# ---- parallel train-bootstrap (fork pool; big frames shared via COW) ----

_WORKER: dict = {}


def _seed_worker(k: int):
    """Train A and B on bootstrap-resample #k of the fold's train rows."""
    Xa = _WORKER["Xa"]
    ya = _WORKER["ya"]
    Xva = _WORKER["Xva"]
    Xb = _WORKER["Xb"]
    yb = _WORKER["yb"]
    Xvb = _WORKER["Xvb"]
    yt = _WORKER["yt"]
    rng = np.random.default_rng(1000 + k)
    idx = rng.integers(0, len(Xa), size=len(Xa))
    pa = _WORKER["fn_a"](Xa.iloc[idx], ya[idx], Xva, seed=1000 + k,
                          t=_WORKER["t"], alpha=_WORKER["alpha"], c=_WORKER.get("c", 0.95), k=_WORKER.get("k", 50),
                      w_frac=_WORKER.get("w_frac", 0.05), n_copies=_WORKER.get("n_copies", 3),
                      k_clusters=_WORKER.get("k_clusters", 4))
    pb = _WORKER["fn_b"](Xb.iloc[idx], yb[idx], Xvb, seed=1000 + k,
                          t=_WORKER["t"], alpha=_WORKER["alpha"], c=_WORKER.get("c", 0.95), k=_WORKER.get("k", 50),
                      w_frac=_WORKER.get("w_frac", 0.05), n_copies=_WORKER.get("n_copies", 3),
                      k_clusters=_WORKER.get("k_clusters", 4))
    ra = rmsle_log(pa, yt)
    rb = rmsle_log(pb, yt)
    return k, ra, rb, ra - rb


def _parallel_seeds(args, Xa, ya, Xva, yt, Xb, yb, Xvb):
    """Run K bootstrap seeds in parallel (fork pool, threads per worker)."""
    import multiprocessing as mp

    jobs = args.jobs
    threads = max(1, (os.cpu_count() or 1) // jobs)
    _WORKER.update(Xa=Xa, ya=ya, Xva=Xva, yt=yt, Xb=Xb, yb=yb, Xvb=Xvb,
                   fn_a=METHODS[args.a], fn_b=METHODS[args.b],
                   t=args.t, alpha=args.alpha, c=args.c, k=args.k,
                      w_frac=args.w_frac, n_copies=args.n_copies, k_clusters=args.k_clusters)
    os.environ["ECUP26_NUM_THREADS"] = str(threads)
    print(f"parallel: {jobs} workers x {threads} threads (seeds={args.seeds})")
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=jobs) as pool:
        out = pool.map(_seed_worker, range(args.seeds))
    os.environ.pop("ECUP26_NUM_THREADS", None)
    out = sorted(out, key=lambda r: r[0])
    print("seed | rmsle(A) | rmsle(B) | delta(A-B) | winner")
    for k, ra, rb, d in out:
        w = "A" if d < 0 else ("B" if d > 0 else "=")
        print(f"{k:4d} | {ra:.6f} | {rb:.6f} | {d:+.6f} | {w}")
    return [r[3] for r in out]


def _free_ram_gb() -> float:
    """Available RAM from /proc/meminfo (GB)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")


def _acquire_lock(lock_path: Path = Path("/tmp/ab_validate.lock")) -> bool:
    """Single-instance lock: prevents accidental concurrent runs (OOM source)."""
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def _release_lock(lock_path: Path = Path("/tmp/ab_validate.lock")) -> None:
    with suppress(OSError):
        lock_path.unlink(missing_ok=True)


def _rss_gb() -> float:
    """Current process RSS in GB (for footprint visibility)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _cleanup(cache_a: Path, cache_b: Path, rm_cache: bool) -> None:
    """Free loaded frames + optionally delete the input fold caches."""
    import gc

    _WORKER.clear()
    gc.collect()
    if rm_cache:
        import shutil

        for c in (cache_a, cache_b):
            if c.exists() and c != CACHE:
                shutil.rmtree(c)
                print(f"removed cache: {c}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="baseline", choices=sorted(METHODS))
    ap.add_argument("--b", default="hurdle", choices=sorted(METHODS))
    ap.add_argument("--fold", default="B")
    ap.add_argument("--seeds", type=int, default=10, help="train-bootstrap resamples (mode=train)")
    ap.add_argument("--m", type=int, default=500, help="test-bootstrap draws (mode=test)")
    ap.add_argument("--mode", choices=["train", "test"], default="test")
    ap.add_argument("--t", type=float, default=0.5, help="zero_threshold cutoff")
    ap.add_argument("--alpha", type=float, default=0.75, help="quantile alpha")
    ap.add_argument("--c", type=float, default=0.95, help="shrink_blend log-space multiplier")
    ap.add_argument("--k", type=int, default=50, help="knn_feat n_neighbors")
    ap.add_argument("--w-frac", type=float, default=0.05, help="whale_copy top fraction")
    ap.add_argument("--beta", type=float, default=1.2,
                    help="whale_boost / shrink_except_whales whale multiplier")
    ap.add_argument("--n-copies", type=int, default=3, help="whale_copy duplication factor")
    ap.add_argument("--k-clusters", type=int, default=4, help="kmeans n_clusters")
    ap.add_argument("--n-est", type=int, default=None, help="override n_estimators (smoke)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="train-mode parallel workers (0 = auto: cpu//2)")
    ap.add_argument("--cache-a", type=str, default=str(CACHE), help="fold cache for method A")
    ap.add_argument("--cache-b", type=str, default=str(CACHE), help="fold cache for method B")
    ap.add_argument("--rm-cache", action="store_true",
                    help="delete cache-a/cache-b after the run")
    args = ap.parse_args()

    if args.n_est:
        REFINED["n_estimators"] = args.n_est
    if args.jobs <= 0:
        # MEMORY-GENTLE default: cpu//4 workers (6 workers was the OOM source:
        # each fork worker peaks ~4 GB for its LightGBM Dataset).
        args.jobs = max(2, (os.cpu_count() or 1) // 4)
    if not _acquire_lock():
        raise SystemExit("ERROR: /tmp/ab_validate.lock exists - another run is active.")
    try:
        return _main(args)
    finally:
        _release_lock()


def _main(args) -> int:
    free = _free_ram_gb()
    if free < 6.0:
        raise SystemExit(f"ABORT: only {free:.1f} GB free RAM - refusing to run.")
    if free < 12.0 and args.jobs > 2:
        print(f"MEM-GUARD: free RAM {free:.1f} GB -> capping jobs to 2")
        args.jobs = 2

    cache_a = Path(args.cache_a)
    cache_b = Path(args.cache_b)
    Xa, ya, Xva, yt_a = _load(args.fold, cache_a)
    Xb, yb, Xvb, yt_b = _load(args.fold, cache_b)
    if not np.allclose(yt_a, yt_b):
        raise SystemExit("ERROR: truth differs between cache-a and cache-b")
    yt_log = yt_a
    print(f"A={args.a}  B={args.b}  fold={args.fold}  mode={args.mode}  rows={len(Xa)}")

    if args.mode == "train":
        if not Xa.index.equals(Xb.index):
            raise SystemExit("ERROR: train rows misaligned between caches")
        deltas = _parallel_seeds(args, Xa, ya, Xva, yt_log, Xb, yb, Xvb)
        _report(np.array(deltas), args.a, args.b)
        _cleanup(cache_a, cache_b, args.rm_cache)
        print(f"peak RSS: {_rss_gb():.1f} GB")
        return 0

    # mode=test: train once per method, paired-bootstrap the TEST users (Efron).
    pa = METHODS[args.a](Xa, ya, Xva, seed=42, t=args.t, alpha=args.alpha, c=args.c, k=args.k,
                      w_frac=args.w_frac, n_copies=args.n_copies, k_clusters=args.k_clusters,
                      beta=args.beta)
    pb = METHODS[args.b](Xb, yb, Xvb, seed=42, t=args.t, alpha=args.alpha, c=args.c, k=args.k,
                      w_frac=args.w_frac, n_copies=args.n_copies, k_clusters=args.k_clusters,
                      beta=args.beta)
    ra = rmsle_log(pa, yt_log)
    rb = rmsle_log(pb, yt_log)
    print(f"single-fit: rmsle(A)={ra:.6f}  rmsle(B)={rb:.6f}  delta={ra-rb:+.6f}")
    n = len(yt_log)
    rng = np.random.default_rng(7)
    deltas = np.empty(args.m)
    for m in range(args.m):
        idx = rng.integers(0, n, size=n)
        deltas[m] = rmsle_log(pa[idx], yt_log[idx]) - rmsle_log(pb[idx], yt_log[idx])
    _report(deltas, args.a, args.b)
    _cleanup(cache_a, cache_b, args.rm_cache)
    print(f"peak RSS: {_rss_gb():.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
