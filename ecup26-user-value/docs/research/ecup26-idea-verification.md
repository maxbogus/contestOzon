# E-CUP T3 — User research vs empirical verification (2026-08-18)

> Source: Maxim's research notes (idea weights 1-10). Each idea is mapped to our
> measured results (tickets T3.10.x, ledger note 003).

## Data reality (our data, NOT the research's premise)

- Data is DAILY (30.6M rows, 250k users, 2025-01-01..2026-02-13), NOT weekly.
- 45.9% zero users (fold B truth); 4-5% "whales" (mean 84 vs median 7.9, p90 216).
- XGBoost was tried and is slightly WORSE than LGBM (fold B 1.6867 vs 1.6857);
  the ensemble is LGBM 0.3 + CatBoost 0.7 (log-space).

## Idea verification table

| # | Idea (weight) | Verdict | Evidence |
|---|---|---|---|
| 1 | Two-stage clf+reg (10) | REJECTED (raw-space) / sign-flip (log-space) | two-stage 2.02; log-space hurdle B +0.00043 / A -0.00178. BUT the insight led to the shrink (T3.10.13) which IS the zero-inflation fix done right |
| 2 | Mixture of Experts / K-means (9) | NOT TESTED -> T3.10.14 | only BTYD-segments + bucket calibration (LB 1.6672 degrade). K-means + per-cluster experts is open |
| 3 | Hierarchical features (9) | DONE, plateau | 143 features; all additions (v3/holiday/gift/absence/behavioral) TIE/noise |
| 4 | Holiday flags is_holiday/pre/post (8) | PARTIAL -> T3.10.15 | have is_holiday_ru, days_to_nearest_holiday, seasonal windows; MISSING pre/post flags |
| 5 | ZIP/ZINB zero-inflation (8) | TESTED, raw-space fails | neural ZILN 2.58, LGBM tweedie 2.53, CatBoost tweedie 2.47, hard-zero falsified; working equivalent = log-space shrink (T3.10.13, fold B +0.00946) |
| 6 | Regularization alpha/lambda (7) | PARTIAL | l1/l2 in classifiers; regressors use LGBM defaults |
| 7 | SHAP importance (7) | NOT TESTED -> T3.10.16 | only gain-importance (T3.9.1 pruning REJECTED) |
| 8 | Heuristics/ETS (6) | NOT TESTED | only mean/median naive baseline |
| 9 | Normalization Z (5) | NOT NEEDED (trees) | confirmed |
| 10 | Features >25-30 (4) | CONFIRMED plateau | we run 143 features; trees absorb but gains are exhausted |

## Conclusions from the research, re-mapped

1. "Two-stage as baseline" -> replaced by the log-space shrink (T3.10.13): the
   zero/positive split is handled multiplicatively in log-space, validated on
   fold A and B at P=0.00 (M=500).
2. The highest-weight NOT-TESTED idea is K-means segmentation (T3.10.14) -> the
   next experiment if we want to push past the shrink.
3. The 'whale' segment (top 4-5%) is under-served: shrink fixes zeros, but the
   whales (under-predicted +0.84 bias on fold B) remain. A whale-specific expert
   is the natural T3.10.14 use case.
4. Pre/post-holiday flags (T3.10.15) are cheap; the target window has two gift
   holidays, and our flags only mark the holiday DAY (is_holiday_ru).

## Open tickets created from this research

- T3.10.14 K-means segmentation / Mixture of Experts (weight 9)
- T3.10.15 is_pre_holiday / is_post_holiday flags (weight 8)
- T3.10.16 SHAP feature-importance diagnostic (weight 7)
