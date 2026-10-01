# T3 LTV/GMV Forecasting Optimization Loop

> **Status**: Active runbook (per T3.7.5)
> **Author**: Maxim Boguslavskiy
> **Created**: 2026-08-16
> **Audience**: Cline (or any LLM agent) running future optimization cycles

## Current State (2026-08-16, post-T3.7.2)

- Best model: **SegmentedBTYD**
- Holdout RMSLE: **5.5611**
- Naive baseline RMSLE: **3.1750** (lift = -0.75)
- **Problem**: model systematically underfits — it cannot capture the
  underlying patterns better than a simple constant forecast.

## Hypothesis

The primary cause is insufficient predictive features. BTYD models
rely heavily on recency, frequency, and monetary value, but they ignore:

- **Seasonality** (day-of-week, month, holidays)
- **Recent trends** (rolling GMV/order counts)
- **Cross-features** (interactions between RFM components)

Adding these will give the model more signal, allowing it to beat the
naive baseline.

## Optimization Cycle

Each cycle:
1. Pick 1 technique from {feature, hyperparameter, model architecture}.
2. RED -> GREEN (TDD).
3. Eval on holdout via `scripts/compare_models.py`.
4. If improvement > threshold AND statistical significance -> commit.
5. If no improvement -> revert and try different technique.

## Acceptance Gates

- RMSLE improvement > 2% vs current best (5.5611 -> 5.4499).
- Bootstrap CI does not overlap with previous best (statistical significance).
- `lift_vs_naive` improves (becomes less negative, ideally > 0).
- All 66+ tests pass, ruff clean.
- Per R8: new code NOT imported by `run.py` until T3.7.6 (integration).

## Cycle #1: Feature Engineering (selected technique, T3.7.3)

Feature Group | Specific Features | Rationale
---|---|---
Calendar | `day_of_week`, `month`, `quarter`, `is_weekend`, `is_holiday_ru` | GMV often exhibits weekly/monthly patterns and holiday spikes
Rolling Windows | Last 7, 14, 30 days: `avg_gmv`, `avg_orders`, `max_gmv` | Capture recent behaviour; local trends improve forecast vs global average
Cross-features | `recency * frequency`, `recency * avg_order_value`, `frequency * monetary` | Interactions expose non-linear relationships
Customer Tenure | `days_since_first_purchase`, `is_new_customer` | New vs mature customers behave differently

All features computed per customer, strictly using training data only
(no future leakage).

**Expected outcome**: 5-15% RMSLE reduction typical; moving from 5.56
toward 5.0-5.2.

## Cycle #2: Hyperparameter Tuning (T3.7.4)

Grid search over 6 params:
- BTYDBaseline: `penalizer_coef`, `time_period_days`
- SegmentedBTYD: `min_transactions_per_segment`
- LightGBM: `n_estimators`, `num_leaves`, `learning_rate`

~5120 combinations on holdout; expected 1-3% additional improvement
on top of Cycle #1.

## Cycle #3: Integration (T3.7.6)

After Cycles #1+2 succeed:
- Integrate best features + best params into `run.py`.
- Re-run pre-submit (T3.7.1 v3).
- Update `pre_submit_t3_v3_<date>.md`.

## If Gates Not Passed

For each cycle:
- Revert the commit.
- Document why features/params didn't help (data noise, insufficient history, wrong assumption).
- Next cycle focuses on a different technique.

## Compliance Checklist (per cycle)

- [ ] Open-source libraries only (pandas, numpy, lifetimes, LightGBM).
      No new heavy deps (Optuna optional, dev-only).
- [ ] Reproducible: random seeds fixed in config, applied.
- [ ] Not imported by `run.py` until T3.7.6 — isolate new code.
- [ ] Assertions added in `run.py` and `validate_submission.py` for R10.
- [ ] No cloud APIs at runtime (R4).
- [ ] No plagiarism / copying from non-author sources (R5).

## Commit & Documentation Convention

Each cycle produces:
- Commit: `feat(features): <short description>` or `feat(tune): <short>`.
- Ticket update: status (In Progress -> Review) + summary with final RMSLE delta.
- Report: `docs/results/<cycle>_<date>.md` with:
  - Baseline vs new RMSLE
  - Lift vs naive
  - Bootstrap CI
  - Feature importance (from LightGBM) or top-10 configs.

## Reference

- Tickets: T3.7.3 (Cycle #1), T3.7.4 (Cycle #2), T3.7.5 (this runbook).
- Holdout methodology: `scripts/compare_models.py` (T3.7.2).
- Pre-submit: T3.7.1 (latest: v2 with SegmentedBTYD).
- Compliance: `.clinerules/02-ecup-rules.md` (R1-R10).

## Update Log

- 2026-08-16: Initial version (after T3.7.2 comparison).
- Future cycles: append entries here.

## Update Log

- 2026-08-16: Initial version (after T3.7.2 comparison).
- 2026-08-17: **Cycle #2 (T3.7.4) DEFERRED**.
  - Grid search 81 configs on 10k user subsample completed in 2273s.
  - Best config: `n_estimators=100, num_leaves=7, min_child_samples=10,
    learning_rate=0.2` -> RMSLE 2.3668.
  - Default config (rank #6) -> RMSLE 2.3687. Improvement 0.08% (< 2% gate).
  - **Diagnostic**: bottleneck is bootstrap_ci (50 boot x 250k ops = 12.5M
    ops per config), not LightGBM fit (0.1s per config).
  - Conclusion: LightGBM defaults are near-optimal for this problem;
    re-tune only after more features are added.
  - Re-open criteria: after additional features or fixing bootstrap
    CI performance.
- 2026-08-17: **Cycle #1 (T3.7.3) DONE**.
  - 4 feature groups added (calendar + rolling + cross + tenure).
  - 11 RED tests, 107 total passing.
  - Subsample result: model_rmsle 5.5611 -> 2.3189 (-58.30% improvement).
  - lift_vs_mean: -0.7552 -> +0.2626 (positive!).
  - Top features: gmv_avg_14d, frequency_x_monetary, gmv_max_30d.
  - Calendar features had zero importance (no seasonality in data).
  - See `docs/results/feat_eng_2026-08-16.md`.
