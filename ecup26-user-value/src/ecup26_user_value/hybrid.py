"""T3.8.10: OOF P(Y>0) stacking feature, shared by dev and production.

The binary classifier's leave-origin-out probability becomes the
'purchase_prob' feature of the base LGBM. Leak-free: for each origin the
classifier trains only on OTHER origins, so no row's label (nor any label of
its origin period) enters its own OOF prediction.

Per R7/W3 the SAME code path is used in dev (benchmark_t3 --hybrid) and prod
(run_pipeline.predict_for_users). All randomness is seeded (R6).
"""

from __future__ import annotations

from typing import Any

import numpy as np

HYBRID_SCALE_POS = 2.5
HYBRID_NEG_CAP = 400_000
HYBRID_ROUNDS = 120


class _ConstantClf:
    """Fallback classifier predicting a constant probability (edge data)."""

    def __init__(self, p: float) -> None:
        self.p = float(p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.p, dtype=np.float64)


def clf_params(seed: int) -> dict[str, Any]:
    """Binary P(Y>0) classifier params for the stacking feature."""
    from ecup26_user_value.run_pipeline import _default_num_threads

    return {
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
        "scale_pos_weight": HYBRID_SCALE_POS,
        "boost_from_average": True,
        "verbose": -1,
        "random_state": int(seed),
        "num_threads": _default_num_threads(),
    }


def fast_rows_mask(y_bin: np.ndarray, seed: int) -> np.ndarray:
    """Boolean mask of classifier rows: all positives + capped negative sample.

    Returns a MASK (not indices) so callers can combine it with other masks
    (e.g. ``fast & ~val`` in leave-origin-out) without broadcast errors.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y_bin), dtype=bool)
    mask[y_bin == 1] = True
    neg = np.where(y_bin == 0)[0]
    cap = min(len(neg), HYBRID_NEG_CAP)
    keep_neg = neg[rng.choice(len(neg), size=cap, replace=False)]
    mask[keep_neg] = True
    return mask


def oof_purchase_prob(
    X: np.ndarray, y_bin: np.ndarray, origins: np.ndarray, seed: int
) -> np.ndarray:
    """Leave-origin-out P(Y>0) per row (leak-free stacking feature).

    For each origin a classifier is fit on ALL OTHER origins only; its
    prediction for the held-out origin becomes that origin's OOF probability.
    Degenerate origins (too few rows or a single class) get OOF = 0.0.
    """
    import lightgbm as lgb

    fast = fast_rows_mask(y_bin, seed)
    oof = np.zeros(len(X), dtype=np.float64)
    for o in np.unique(origins):
        val = origins == o
        tr = fast & ~val
        if tr.sum() < 50 or len(np.unique(y_bin[tr])) < 2:
            continue  # degenerate origin: OOF stays 0.0
        clf = lgb.train(
            clf_params(seed),
            lgb.Dataset(X[tr], label=y_bin[tr]),
            num_boost_round=HYBRID_ROUNDS,
        )
        oof[val] = np.asarray(clf.predict(X[val]), dtype=np.float64)
    return oof


def fit_purchase_prob(X: np.ndarray, y_bin: np.ndarray, seed: int) -> Any:
    """Full-data P(Y>0) classifier for inference rows (production equivalent)."""
    import lightgbm as lgb

    fast = fast_rows_mask(y_bin, seed)
    if fast.sum() < 50 or len(np.unique(y_bin[fast])) < 2:
        return _ConstantClf(float(y_bin[fast].mean()))
    return lgb.train(
        clf_params(seed),
        lgb.Dataset(X[fast], label=y_bin[fast]),
        num_boost_round=HYBRID_ROUNDS,
    )
