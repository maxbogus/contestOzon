"""Hyperparameter tuning CLI for T3.7.4 (Cycle #2).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (lightgbm, pandas, scipy, tqdm)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py

Grid search over 4 LightGBM hyperparameters (81 configs):
- n_estimators: [50, 100, 200]
- num_leaves: [7, 15, 31]
- min_child_samples: [10, 50, 100]
- learning_rate: [0.05, 0.1, 0.2]

Usage:
    uv run python scripts/tune_hyperparams.py \\
        --data data/train.parquet \\
        --holdout-days 30 \\
        --n-boot 100 \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecup26_user_value.eval.diagnostics import bootstrap_ci  # noqa: E402
from ecup26_user_value.eval.metrics import compute_all_metrics  # noqa: E402
from ecup26_user_value.features.build_features import build_features  # noqa: E402

#: Search space for LightGBM hyperparameters.
SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [50, 100, 200],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [10, 50, 100],
    "learning_rate": [0.05, 0.1, 0.2],
}

SEED = 42


def expand_grid(search_space: dict[str, list]) -> list[dict]:
    """Cartesian product over all param values."""
    import itertools

    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def load_synthetic_df(n_users: int = 100, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic transactions with full T3 schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-09-01", periods=60, freq="D")
    for uid in range(n_users):
        seg = "search" if uid % 2 == 0 else "cat"
        for _o in range(int(rng.integers(3, 10))):
            rows.append(
                {
                    "user_id": uid,
                    "event_date": dates[int(rng.integers(0, len(dates)))],
                    "type": seg,
                    "price": float(rng.uniform(5, 100)),
                }
            )
    return pd.DataFrame(rows)


def fit_predict_one(
    train_df: pd.DataFrame,
    val_users: np.ndarray,
    params: dict,
    n_boot: int = 100,
    seed: int = SEED,
) -> tuple[np.ndarray, float]:
    """Train LightGBM on (train_df -> features), predict for val_users.

    Returns (predictions, fit_time_sec).
    """
    import lightgbm as lgb

    # Build features once per fit
    cutoff = train_df["event_date"].max()
    features = build_features(train_df, ref_date=cutoff)
    features_indexed = features.set_index("user_id")

    # Target = sum of price per user in train (proxy for y_true alignment)
    y_train_series = train_df.groupby("user_id")["price"].sum()
    X_train = features_indexed.copy()
    y_train = y_train_series.reindex(X_train.index).fillna(0.0)
    X_train = X_train.fillna(0.0)

    t0 = time.time()
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        learning_rate=params["learning_rate"],
        verbose=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    X_val = X_train.reindex(val_users).fillna(0.0)
    pred = model.predict(X_val)
    pred = np.maximum(pred, 0.0)
    return pred, fit_time


def format_results_row(
    params: dict, metrics: dict, fit_time: float, ci_low: float, ci_high: float
) -> dict:
    """Format a single config result as a flat dict."""
    return {
        "params": params,
        "model_rmsle": metrics["model_rmsle"],
        "lift_vs_mean": metrics["lift_vs_mean"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "spearman": metrics["spearman"],
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "fit_time_sec": fit_time,
    }


def run_grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_boot: int = 100,
    top_k: int = 10,
    seed: int = SEED,
) -> list[dict]:
    """Run full grid search and return top-K configs by model_rmsle."""
    y_true_series = val_df.groupby("user_id")["price"].sum()
    val_users = y_true_series.index.values
    y_true = y_true_series.values

    combos = expand_grid(SEARCH_SPACE)
    results: list[dict] = []

    for params in tqdm(combos, desc="Grid search", total=len(combos)):
        pred, fit_time = fit_predict_one(train_df, val_users, params, n_boot=n_boot, seed=seed)
        metrics = compute_all_metrics(y_true, pred)
        ci_low, ci_high = bootstrap_ci(y_true, pred, n_boot=n_boot, seed=seed)
        results.append(format_results_row(params, metrics, fit_time, ci_low, ci_high))

    results.sort(key=lambda r: r["model_rmsle"])
    return results[:top_k]


def write_report(
    top_results: list[dict],
    output: Path,
    all_results_count: int,
    elapsed_sec: float,
) -> None:
    """Write top-K results as Markdown + CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    rows = []
    for r in top_results:
        row = {
            **r["params"],
            "model_rmsle": r["model_rmsle"],
            "lift_vs_mean": r["lift_vs_mean"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "spearman": r["spearman"],
            "bootstrap_ci_low": r["bootstrap_ci_low"],
            "bootstrap_ci_high": r["bootstrap_ci_high"],
            "fit_time_sec": r["fit_time_sec"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)

    # Markdown
    md = [f"# Hyperparameter tuning results — {pd.Timestamp.now().date()}\n"]
    md.append(f"- Total configs evaluated: {all_results_count}")
    md.append(f"- Top-K shown: {len(top_results)}")
    md.append(f"- Wall time: {elapsed_sec:.1f}s")
    md.append(
        f"- Bootstrap: {top_results[0]['bootstrap_ci_low']:.4f} <= RMSLE <= {top_results[0]['bootstrap_ci_high']:.4f} (best config)\n"
    )

    md.append(
        "| Rank | n_est | leaves | min_data | lr | RMSLE | lift | mae | rmse | spearman | fit_sec |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(top_results, 1):
        p = r["params"]
        md.append(
            f"| {i} | {p['n_estimators']} | {p['num_leaves']} | {p['min_child_samples']} | "
            f"{p['learning_rate']} | {r['model_rmsle']:.4f} | {r['lift_vs_mean']:.3f} | "
            f"{r['mae']:.2f} | {r['rmse']:.2f} | {r['spearman']:.3f} | {r['fit_time_sec']:.1f} |"
        )

    md.append("\n## Best config\n")
    best = top_results[0]
    md.append("```python\nLightGBMRegressor(")
    for k, v in best["params"].items():
        md.append(f"    {k}={v},")
    md.append(")\n```\n")

    md.append("## Compliance\n")
    md.append("- R1: open-source (lightgbm, pandas, tqdm)")
    md.append("- R6: reproducible (seeded)")
    md.append("- R8: dev-only, NOT imported by run.py")
    md.append("- R10: no submit.csv writes")

    output.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "train.parquet",
    )
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        help="If set, use only this fraction of train (subsample).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "tune_hyperparams_latest.md",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data (100 users) for quick test."
    )
    args = parser.parse_args()

    print("[tune] Loading data...")
    if args.synthetic:
        df = load_synthetic_df(n_users=100, seed=SEED)
    else:
        df = pd.read_parquet(args.data)
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["type"] = np.where(df["search"] > 0, "search", "cat")
        df["price"] = df["gmv"].astype(float)
        df = df[["user_id", "event_date", "type", "price"]]

    max_date = df["event_date"].max()
    cutoff = max_date - pd.Timedelta(days=args.holdout_days)
    train_df = df[df["event_date"] <= cutoff].copy()
    val_df = df[(df["event_date"] > cutoff) & (df["event_date"] <= max_date)].copy()
    print(
        f"[tune] train={len(train_df)}, val={len(val_df)}, "
        f"users_in_val={val_df['user_id'].nunique()}"
    )

    if args.train_frac is not None and args.train_frac < 1.0:
        rng = np.random.default_rng(SEED)
        sample_users = rng.choice(
            train_df["user_id"].unique(),
            size=int(len(train_df["user_id"].unique()) * args.train_frac),
            replace=False,
        )
        train_df = train_df[train_df["user_id"].isin(sample_users)]
        print(f"[tune] train subsampled to {len(train_df)} rows")

    n_configs = 1
    for v in SEARCH_SPACE.values():
        n_configs *= len(v)
    print(f"[tune] Grid: {n_configs} configs, {args.n_boot} bootstrap samples each")

    t0 = time.time()
    top_results = run_grid_search(
        train_df,
        val_df,
        n_boot=args.n_boot,
        top_k=args.top_k,
        seed=SEED,
    )
    elapsed = time.time() - t0

    write_report(top_results, args.output, n_configs, elapsed)
    print(f"\n[tune] DONE: {elapsed:.1f}s, top-1 RMSLE={top_results[0]['model_rmsle']:.4f}")
    print(f"[tune] Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
