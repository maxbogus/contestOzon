# Baseline vs Segmented BTYD comparison (2026-08-16)

## Setup

- Data: `/home/maxbogus/Repositories/contest/ecup26-user-value/data/train.parquet`
- Train: 27837039 rows (<= 2026-01-14)
- Val: 2793967 rows (2026-01-14 .. 2026-02-13, 30 days)
- Users with holdout activity: 250000
- Target: sum(price) per user in holdout window
- Models:
  - **btyd_global**: BTYDBaseline (single global model)
  - **btyd_segmented**: SegmentedBTYD (per-search-vs-cat + global fallback)

## Results

| Metric | btyd_global | btyd_segmented |
|---|---|---|
| `model_rmsle` | 5.5726 | 5.5611 |
| `naive_rmsle_mean` | 3.1750 | 3.1750 |
| `lift_vs_mean` | -0.7552 | -0.7515 |
| `mae` | 15267.3109 | 15098.2611 |
| `rmse` | 46405.2563 | 45969.2526 |
| `bias_mean` | 15264.2982 | 15095.2218 |
| `spearman` | 0.6200 | 0.6204 |
| `quantile_calibration` | 22389.2666 | 22136.6706 |
| `coverage_q10_q90` | 0.2272 | 0.2277 |
| `bootstrap_ci_low` | 5.5641 | 5.5525 |
| `bootstrap_ci_high` | 5.5813 | 5.5698 |
| `adversarial_max_rmsle` | 5.5726 | 5.5611 |

## Recommendation

**SWITCH to SegmentedBTYD: 0.2% RMSLE improvement.**

Winner: `btyd_segmented`

## Notes

- Holdout window simulates E-CUP private split (last 30 days).
- This is **not** the actual leaderboard - the real test set
  (2026-02-14 .. 2026-03-15) is held by organizers.
- Tier 1 metrics: model_rmsle, naive_rmsle, lift, mae, rmse,
  bias_mean, spearman, quantile_calibration.
- Tier 2/3 metrics: coverage, PIT, bootstrap CI, adversarial.

## Compliance

- R1: open-source (lifetimes Apache 2.0)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py
- R10: no submit.csv writes
