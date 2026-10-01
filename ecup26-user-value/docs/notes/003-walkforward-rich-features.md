# Note 003 — Walk-forward + full-schema features (2026-08-17)

## Status

- Leaderboard (public): **1.6976** (submitted: single-origin log1p, 22 features)
- Top-1 leaderboard: **1.6462**; field is very tight (1.646..1.697 = ~130 places)
- Honest out-of-sample proxy (`scripts/validate_pipeline_30d.py`): **1.7309** for
  the submitted model — matches the leaderboard within ~1.7% => **proxy is reliable**

## Numbers to remember

- In-sample proxy (against the training-label window): **1.31** — NOT comparable
  to the leaderboard. Use ONLY the out-of-sample proxy (~1.7x).
- After walk-forward + full-schema features: proxy **1.7070** (~ -1.4%).

## What changed (commit 278a048)

- Walk-forward training: 8 monthly origins (features at T, label GMV over
  (T, T+30]) instead of a single recent window.
- Full daily-schema activity features (63 total): per-channel GMV, order/cart/
  search counts, conversion rates, recency per channel.
- Memory: column selection + float32/int32/int8 downcast; memory guard skips
  origins below 6 GB free RAM.
- `TARGET_TRANSFORM` switch (log1p / tweedie) + walk-forward config in cache key.

## Incidents

- OOM killed VS Code 2026-08-17: full 18-column float64 frame ~25+ GB on a
  30 GB host during groupby feature building. Fixed with downcast + column
  selection + memory guard.
- OOM killed the production fit 2026-08-19 (2nd OOM): the 3-model production fit
  (LGBM 650 + XGB 550 + CatBoost 900 on the 1.88M-row walk-forward train) was
  SIGKILLed with no traceback - the walk-forward itself COMPLETED all 8 origins
  (last 2025-06-18). Root cause: default 12 threads (ECUP26_NUM_THREADS unset)
  blew up CatBoost/XGB hist memory on the full train. Fixed: ECUP26_NUM_THREADS=4
  + a free-RAM guard (fail-fast < 5 GB) in scripts/experiment_blend_production.py.

## Experiment backlog (RICE-sorted, tracked in docs/backlog/tickets T3.8.x)

| # | Experiment | Ticket | RICE | Status |
|---|---|---|---|---|
| 1 | Tweedie target | T3.7.6 | 36 | DONE - WORSE (2.53 vs 1.71) |
| 2 | Ensemble (XGB/CatBoost blend) | T3.8.2/3 | 10.5 | DONE - REJECTED (noise) |
| 3 | Walk-forward origin/step tuning | T3.8.4 | 10 | TODO |
| 4 | LightGBM tuning (200/31/0.05/0.8/0.8) | T3.7.6 | 8 | DONE (1.6950) |
| 5 | Features v2 / v3 | T3.8.5 | 6.7 | v2 DONE (1.6910); v3 REJECTED (1.7170) |
| 6 | Post-hoc calibration | T3.8.6 | 4 | DONE (1.7068) |
| - | Benchmarker | T3.8.1 | 1.6 | DONE (CV 1.7168) |
| - | Benchmarker ext (XGB/CatBoost, Optuna, pairwise) | T3.8.2 | 2.1 | DONE - REJECTED (noise) |

## Commands

```bash
# honest out-of-sample proxy (expect ~1.71)
uv run python scripts/validate_pipeline_30d.py --cache-dir /tmp/t3val_cache_v2

# production submission
uv run python -u run.py --test_data_path data/sample_submit.csv --output_path submit_final.csv
```

## Results log

- 2026-08-17 E1 Tweedie: proxy RMSLE **2.5289** (log1p = 1.7070) -> rejected,
  log1p stays as TARGET_TRANSFORM.
- 2026-08-17 Submission v2 (walk-forward log1p): `submit_final.csv` ready,
  Docker image pushed (digest sha256:b8cf8041). Local vs container identical
  (max diff 8e-07). Proxy RMSLE (out-of-sample) 1.7070.

## RMSLE ledger (proxy out-of-sample; leaderboard ~ proxy - 0.035)

| Step | Change | Proxy RMSLE | Leaderboard |
|---|---|---|---|
| 0 | baseline single-origin, 22 feats | 1.7309 | 1.6976 (fact) |
| 1 | + walk-forward + feats v1 (63) | 1.7070 | 1.6723 (fact) |
| 2 | + LightGBM tuning (200/31/0.05/0.8/0.8) | 1.6950 | ~1.660 |
| 3 | + feats v2 (60/90d, recency, 90 feats) | 1.6910 | ~1.656 |
| 4 | T3.8.1 benchmarker -> n_estimators 300 | 1.6901 (CV mean 1.7168) | ~1.655 |
| 5 | T3.8.2/3 ensemble (lgb+xgb+cat blend) | 1.7171 (blend) vs 1.7176 (lgb) - noise, REJECTED | ~1.655 |
| 5a | T3.8.5 features v3 (holiday dist, dow, convs) | 1.7170 - noise, REJECTED | ~1.655 |
| 6 | T3.8.7 Russian holiday/promo/gift features | 1.7155 (CV); fold B ~1.688 | ~1.657 |
| 7 | T3.8.4 walk-forward step (15d/16 or) | 1.7158 vs 1.7155 - no gain, REJECTED | ~1.657 |
| 8 | T3.8.12 gift-season labels (12 or) | 1.7155 = baseline, REJECTED | ~1.657 |
| 9 | T3.8.13 season-only model (Feb-Mar 25) | fold B 2.0043 vs 1.6881, REJECTED | - |
| 10 | T3.8.10 hybrid OOF purchase_prob feature | 1.7123 (folds 1.7426/1.6821), KEPT | ~1.651 |
| 8 | T3.8.6 calibration (log-space bias) | 1.7068 (fold B 1.6744) - KEPT | ~1.643 |
| 9 | T3.8.8/9 two-stage + weighted single-stage | 2.02 / 2.09 - REJECTED | ~1.657 |

Rule: after each task, record its proxy RMSLE before -> after + LB estimate
(proxy - 0.035) in this ledger and in the ticket's Result section.

- 2026-08-17 T3.8.7 Russian/Ozon holiday+promo+gift features: CV 1.7170 -> 1.7155, proxy fold B ~1.6881 (LB est ~1.657) -> KEPT.

- 2026-08-17 Phases 1-3: two-stage+bias ABORTED (log_cal 1.7541 > single); weighted single-stage 2.09 REJECTED; log-space calibration 1.7068 KEPT (integrated into pipeline).

- 2026-08-17 T3.8.4: step 15d (16 origins, 3.7M rows) CV 1.7158 vs step 30d 1.7155 - no gain, keep step 30/8.
- 2026-08-17 T3.8.12: 12 origins incl. gift season 2025 label (2.4-2.7M rows): CV 1.7155 (folds 1.7429/1.6883) vs baseline 1.7155 (1.7434/1.6881) - no gain, REJECTED.
- 2026-08-17 T3.8.13: train ONLY on gift season 2025 (198k rows): fold B 2.0043 vs 1.6881 - REJECTED.
- 2026-08-17 T3.8.10: leave-origin-out OOF P(Y>0) as feature: CV 1.7123 vs 1.7155 (-0.0032), fold B 1.6821 vs 1.6881 (-0.0060); purchase_prob rank 1/115 gain. KEPT -> integrate into production.
- 2026-08-17 hybrid integrated into production (predict_for_users, USE_HYBRID=1): submit_final_hybrid.csv, CV 1.7123, fold B 1.6821, run ~8min.
- 2026-08-17 T3.9.1 feature selection: gain-rank prune curve K=[20..114] on fold B: 1.6986..1.6882, full=1.6882 - pruning gives NO gain, REJECTED. Window gain: 90d=45.7% (dominant).
- 2026-08-17 T3.9.2 rolling windows +120/180d: fold A 1.7427, fold B 1.6874, CV 1.7150 (vs 1.7155) - consistent -0.0005/0.0007, KEPT -> integrated (DEFAULT_ROLLING_WINDOWS).
- 2026-08-17 production integrated w/ extended windows: submit_final_w120_180.csv (250k rows, valid, 138 feats, run ~21min), LB est ~1.658.
- 2026-08-17 kitchen-sink blend (lgb 0.55 / xgb 0.2 / cat 0.2 / gru 0.05): fold B 1.6866 vs 1.6874 (-0.0008); xgb alone 1.6873 < lgb 1.6874.
- 2026-08-17 T3.9.6 neural ZILN (dense 90d tokens + ZILN head): fold B 2.5788 vs 1.6874 - much worse, REJECTED. Likely: ZILN loss optimizes raw-space, RMSLE is log1p.
- 2026-08-17 production with tuned params + extended windows: submit_final_tuned.csv (138 feats, 500 trees @ lr 0.02, run ~29min), CV 1.7138, LB est ~1.657.
- 2026-08-17 T3.9.7 fine +-10% local search: full CV 1.7137 vs 1.7138 (-0.0001) - KEPT (weak), adopted 550/97/0.02/21/0.66.
- 2026-08-17 T3.9.8 dense+MSE-log1p: fold B 1.7262 vs LGBM 1.6861 - REJECTED; ZILN confirmed as dense failure; neural path exhausted.
- 2026-08-17 FINAL: submit_final_refined.csv (refined 550/97/0.02/21/0.66 + windows 120/180, 138 feats) CV 1.7137, LB est ~1.657. Stacked: windows -0.0005 + retune -0.0012 + refined -0.0001 = -0.0018 CV.
- 2026-08-18 T3.9.12 per-bucket calibration: fold B -0.0100 (cross A->B) BUT LB 1.6672 (+0.0101 DEGRADED) - target season NOT over-predicted like fold B. REJECTED. 2nd proof fold-B level-adjustments do not transfer (after hybrid).
- 2026-08-18 T3.9.13 blend LGBM 0.3 + CatBoost 0.7: fold B 1.6847 (-0.0010 vs LGBM); submit_final_blend.csv ready (LB est ~1.656). CatBoost tuned 1.6849 > LGBM.


- 2026-08-18 T3.10.x NEW METHODOLOGY: LB compressed to 0.0106 (#1 1.645914 vs #78 1.656517);
  noise criterion changed to SIGN CONSISTENCY across bootstrap folds (docs/EXPERIMENTS.md,
  scripts/ab_validate.py). Single-fold deltas no longer decide (hybrid/calibration degraded on LB).
- 2026-08-18 T3.10.2 E1 zero-threshold sweep (fold B, 550t): BEST t=0.05 (=no threshold),
  monotonic degradation t=0.10 +0.0002 / t=0.50 +0.1027 / t=0.95 +1.02.
  -> hard-zero in log-space FALSIFIED (classifier cannot isolate true zeros; log-space
  penalty for zeroing a true positive outweighs gain). Ledger entry replacing "main lever" claim.
- 2026-08-18 T3.10.2 E1 baseline re-check: fold B rmsle(550t) = 1.685742 == ledger 1.6857. OK.

- 2026-08-18 T3.10.2 E2 zero-threshold bootstrap (test mode M=500, fold B, t=0.1):
  baseline wins 100% of draws (delta -0.00017, CI [-0.00026,-0.00009]). FALSIFIED.
  Empirical noise floor of fold-B RMSLE = std 0.00024 (paired test-bootstrap) ->
  single-fold deltas <0.0005 are unreliable. This CONFIRMS the new noise criterion.
- 2026-08-18 T3.10.3 log-space hurdle: SIGN-FLIP across folds. fold B: hurdle WINS
  (delta +0.00043, P(A beats B)=0.04, CI [-0.00002,+0.00087]); fold A: baseline WINS
  (delta -0.00178, P=1.00, CI [-0.0022,-0.0013]). zero_frac truth: A=0.437 (NY spike),
  B=0.459 (lull). Hurdle helps only the higher-zero fold; target gift season has lower
  zero incidence -> REJECTED as-is. Lesson: single-fold sign does not transfer.
- 2026-08-18 T3.10.5 E1 CatBoost(log1p) vs LGBM baseline, fold B (M=500): CatBoost WINS
  100% of draws (delta +0.00087, CI [+0.0005,+0.0012]). T3.9.13 1.6849 vs 1.6857 IS REAL.
  Cross-fold check on fold A pending.

- 2026-08-18 T3.10.5 E1b CatBoost(log1p) vs LGBM, fold A (M=500): TIE
  (delta +0.000145 toward CatBoost, P(A beats B)=0.17, CI [-0.00018,+0.00051]).
  No sign-flip: CatBoost >= LGBM on BOTH folds (fold B WINS 100%, fold A TIE-with-lean).
  CatBoost is a genuine, fold-consistent candidate (was already in submit_final_blend 0.7).

- 2026-08-18 T3.10.5 E2 CatBoost Tweedie(raw) vs LGBM log1p, fold B (M=500):
  baseline wins 100% (2.4687 vs 1.6857, delta -0.783). Raw-space zero-inflation loss
  confirmed catastrophic for RMSLE (2nd proof after LGBM tweedie 2.53). log1p L2 stays.

- 2026-08-18 NEW-FINDING (dual-cache A/B): T3.9.11 features (n_active_7/14d,
  has_cart_no_order_30d, gmv_90_180d) HURT fold B by -0.00017 (P(A without wins)=0.93,
  CI [-0.00042,+0.00006]). They are in production (commit 146ed78) but were never
  A/B'd on the fold cache. Recommend REMOVAL (or confirm on fold A first).
- 2026-08-18 T3.10.4 absence/gap features (t3911 vs abs, fold B, M=500): TIE,
  delta +0.00006 toward absence, P=0.30, CI [-0.00019,+0.00033]. Effect at noise floor
  -> NOT integrated (keep behind ECUP26_ABSENCE_FEATURES flag).

- 2026-08-18 T3.10.6 quantile blend (0.5/0.75/0.9, log-space) vs baseline, fold B:
  baseline wins 100% (2.1605 vs 1.6857). Quantiles != mean; RMSLE is mean-based ->
  quantile regression REJECTED (theory + empirics).

- 2026-08-18 T3.9.13-blend (0.3 LGBM + 0.7 CatBoost) vs baseline, fold B (M=500):
  blend WINS 100% (1.684677 vs 1.685742, delta +0.001065, CI [+0.00082,+0.00130]).
  Better than CatBoost alone (+0.00087). The current submit_final_blend.csv IS the
  right family; blend direction confirmed rigorously.

- 2026-08-18 T3.10.8 ordinal-residual (bucket clf + within-bucket regs) vs baseline,
  fold B (M=500): baseline wins 100% (1.687319 vs 1.685742, delta -0.00158,
  CI [-0.0020,-0.0011]). REJECTED - the trained version is WORSE than the (already
  rejected) post-hoc bucket calibration. Ordinal buckets add no value for RMSLE.
- 2026-08-18 CRASH-POSTMORTEM: Ubuntu session OOM'd at 6 parallel train-bootstrap
  workers (~25 GB peak: 6 x ~4 GB Dataset copies). ab_validate.py reworked to be
  memory-gentle: default jobs=cpu//4 (3), MEM-GUARD aborts below 6 GB free RAM and
  caps jobs at 2 below 12 GB, single-instance lock /tmp/ab_validate.lock, RSS report.
  GPU (RTX 5060 8GB) left idle on purpose: it drives the display, compute stays CPU.

- 2026-08-18 T3.10.5 cat_lgb TRAIN-mode deep check (6 seeds, 3 workers x 4 threads):
  CatBoost wins 6/6 seeds (mean delta +0.00068, CI [+0.00036,+0.00095]). Passes BOTH
  test-bootstrap (fold B 100%, fold A TIE-lean) and train-bootstrap. CatBoost is the
  most robust improvement over the LGBM baseline. Blend (0.3/0.7) even better (+0.00106).

- 2026-08-18 T3.9.11 features fold-A check (dual-cache, M=500): TIE
  (delta +0.00003 toward the features, P=0.39, CI [-0.00019,+0.00028]). Mixed with
  fold B (-0.00017, P=0.93) -> NO production change (noise-level, not sign-consistent
  on both folds).

- 2026-08-18 blend (0.3 LGBM + 0.7 CatBoost) fold-A check (M=500): blend WINS 100%
  (1.741095 vs 1.741581, delta +0.00049, CI [+0.00026,+0.00074]). NO sign-flip: blend
  wins both folds (B +0.00106 / A +0.00049). submit_final_blend.csv CONFIRMED as the
  production model family. Final session verdict below.

## 2026-08-18 session verdict (bootstrap A/B, sign-consistency criterion)

| Method | fold B | fold A | train-mode | Verdict |
|---|---|---|---|---|
| zero_threshold | -0.00017 (A 100%) | - | - | REJECTED (falsified) |
| log-space hurdle | +0.00043 (B wins) | -0.00178 (A 100%) | - | REJECTED (sign-flip) |
| CatBoost log1p | +0.00087 (B 100%) | TIE-lean (P=0.17) | +0.00068 (6/6) | KEEP |
| CatBoost Tweedie | -0.78 (A 100%) | - | - | REJECTED (raw-space) |
| quantile blend | -0.47 (A 100%) | - | - | REJECTED |
| kNN feature | too slow | - | - | DEFERRED |
| ordinal-residual | -0.00158 (A 100%) | - | - | REJECTED |
| blend 0.3/0.7 | +0.00106 (B 100%) | +0.00049 (A 100%) | - | CONFIRMED (production) |
| T3.9.11 feats | -0.00017 (P=0.93) | +0.00003 (TIE) | - | mixed - no change |

- 2026-08-18 P0 gift-season fold G + promo fold P (full-history spike proxy, cutoff
  2025-11-01): fold P blend WINS 100% (+0.00052, CI [+0.00027,+0.00076]) - production
  is robust on spike regimes. fold G (gift 2025, cutoff 2025-02-14) is DATA-LIMITED
  (2 origins, 168K rows): baseline beats blend (-0.0058) but absolute RMSLE ~1.89-1.90
  -> CatBoost-overfitting artifact on tiny data, NOT a gift-season failure.
- 2026-08-18 KEY DOMAIN FINDING: gift season is NOT a spike. fold G truth zero_frac=
  0.484 > fold B (0.459); mean 85.6 ~ fold B (84.0). People buy gifts in ADVANCE
  (up to ~2 weeks before Feb 23 / Mar 8), so the target (Feb 14 - Mar 15 2026) is a
  zero-heavy spread-out period, not a burst. Target may have MORE zeros than fold B ->
  production may slightly OVER-predict the LB (consistent with T3.9.12 calibration degrade).
- 2026-08-18 T3.10.12 behavioral timing features: 5 non-overlapping windows of the
  2025 gift season (adv_feb23 [Feb14-20], lm_feb23 [Feb21-23], adv_mar8 [Feb24-Mar5],
  lm_mar8 [Mar6-8], late [Mar9-15]) + shares (planner/lastmin/late) + mass-buying
  baseline (pre-Sep1, pre-NY) + gift_vs_mass ratio. Behind ECUP26_BEHAVIOR_FEATURES=1.

- 2026-08-18 T3.10.12 behavioral timing features (14 feats, M=500): fold B TIE
  (delta +0.00006, P=0.31), fold A B WINS (delta +0.00019, P=0.07, CI [-0.00006,+0.00039]).
  Directionally consistent (no sign-flip) but noise-floor effect -> NOT integrated
  (behind ECUP26_BEHAVIOR_FEATURES=1). Feature plateau confirmed again.

- 2026-08-18 T3.10.13 LOG-SPACE SHRINK - BREAKTHROUGH (production blend c=0.95):
  bootstrap A/B M=500: fold B delta +0.00946 (P=0.00), fold A +0.00233 (P=0.00).
  NO sign-flip, wins both folds 100%. Optimum c: B=0.94 / A=0.97 (scales with
  zero_frac; target ~0.484 wants c~0.94-0.95). Zeros carry 49.9% of error^2 and are
  over-predicted (pred_log 1.42); multiplicative shrink fixes what additive bias
  (T3.8.6, b~0) could not. Production: --shrink 0.95. Matches the #78->#1 gap (0.0106).

- 2026-08-18 T3.10.7 kNN feature (fast v3: PCA-10 + KDTree on 150k ref + k=25,
  ~4min): fold B M=200 baseline wins 98% (delta -0.00025, CI [-0.00045,-0.00002]).
  REJECTED - smoothed-label neighbor-mean adds noise. Feature plateau holds.
  Note: v1 (brute cosine) and v2 (KDTree on 1.9M) were >15min, v3 made it feasible.

- 2026-08-18 T3.10.13 shrink REGIME-DEPENDENCE (fold P/G validation):
  fold P (promo spike, zeros 0.432): c=0.95 HURTS -0.0051 (blend wins 100%, P=1.00).
  fold G (gift, zeros 0.484, data-limited): c=0.95 HELPS +0.0203 (shrink wins 100%).
  => optimal c scales with zero_frac: B=0.94, A=0.97, P>1 (inflate), G<<1. The
  shrink is a regime-dependent lever: GREAT for zero-heavy (gift) folds, BAD for
  spike (promo) folds. Target (gift, zeros~0.484) is zero-heavy -> shrink likely
  helps, but the fold-P flip is a risk flag per the sign-consistency methodology.

- 2026-08-18 T3.10.13 piecewise-shrink probe (t in 1.5..3.0, c in 0.3..0.85):
  fold B best t=3.0,c=0.85 -> 1.6788 (gain -0.0059, WORSE than global c=0.94 -0.0098);
  fold P ALL configs HURT (+0.002..+0.22). Even shrinking only pred_log<1.5 hurts on
  the spike -> regime dependence is fundamental, not fixable by a threshold.
  STRATEGY: submit BOTH blend and shrink0.95; public LB (20% users) is the regime
  probe - if shrink wins public, target is zero-heavy (gift-like, keep for private);
  if blend wins, target is spike-like (drop shrink).

- 2026-08-18 T3.10.19 whale oversampling (top-5% copies x3, M=500): REJECTED all
  folds - B -0.0036, A -0.0019, P -0.0019 (baseline wins 100% everywhere).
  RMSLE is unweighted per user -> overweighting whales distorts calibration
  (same mechanism as weight=1+log1p 2.09). The whale tail is best left to the
  log-space loss.

- 2026-08-18 T3.10.17 ACTOR FEATURES - FIRST FEATURE BREAKTHROUGH (10 feats:
  instant_buy_share, cart_backlog, max_day_cart/ord, buy_gap_mean/cv, dom_mode,
  gmv_cat_share, gmv_cv_daily, max_day_gmv_share). M=500: fold B +0.00232
  (P=0.00, CI [+0.0019,+0.0028]), fold A +0.00149 (P=0.00, CI [+0.0011,+0.0019]).
  First feature addition passing sign-consistency (absence/behavioral/timing were
  TIE). Cart-state + regularity + concentration capture what windowed sums miss.
  fold P check pending (rebuild). Integration: ECUP26_ACTOR_FEATURES=1.

- 2026-08-18 T3.10.17 ACTOR FEATURES CONFIRMED on fold P (spike): +0.00244
  (P=0.00, CI [+0.0020,+0.0029]). ALL THREE folds pass at P=0.00 - no sign-flip.
  First session method winning B+A+P. Integration: ECUP26_ACTOR_FEATURES=1 in
  the production blend (shrink stays OFF - LB-regressed). Expected LB ~ -0.002.

- 2026-08-18 T3.10.14 K-means: cluster-as-feature REJECTED (B -0.00033, A TIE,
  P -0.00037). MoE (k=3 per-cluster experts): fold B +0.00054 (P=0.01) but A TIE
  (-0.00013) and P TIE (-0.00005) - WEAK/INCONCLUSIVE, not integrated. GBM already
  partitions the feature space; only actor features (T3.10.17) passed all folds.

- 2026-08-18 T3.10.25 targeted interactions (on actor features): REJECTED -
  sign-flip on fold P (B +0.00029 P=0.01, A TIE, P -0.00042 A wins 100%). 2nd
  method (after shrink) that helps B but flips on the spike fold - the P check
  is earning its keep. Actor features remain the only confirmed feature win.

- 2026-08-18 T3.10.26 funnel-chain features (9 feats incl. cart_to_order_lag):
  REJECTED - P-flip. fold B +0.00044 (P=0.00) on top of actor, fold A TIE (-0.00005),
  fold P -0.00022 (A wins 99%). 3rd method (after shrink, interactions) that helps
  the lull fold but flips on the spike fold. The P-gate is doing its job.
  cart_to_order_lag alone (wife insight) may deserve a single-feature re-probe.

- 2026-08-19 T3.10.26b cart_to_order_lag single-feature probe: STRUCTURALLY DEAD.
  Measured on 2000-user sample: lag=0 for 100% of 38813 orders (order days ALWAYS
  have same-day to_cart>0 in Ozon's data). The 'fill cart, buy later' pattern is
  captured by cart_backlog (already in the WINNING actor features), not by a lag.
  Funnel set verdict stands: REJECTED (P-flip). Production unchanged (blend+actor).
  Note: memory-lean merge_asof used; initial np.where/ffill versions OOM'd the
  rebuild on the 30GB box - merge_asof fixed it.

- 2026-08-19 T3.10.20 BTYD features: WEAK POSITIVE, no sign-flip.
  btyd_exp_30d (expected 30d purchases): B +0.00029 (P=0.01), A TIE (-0.00002),
  P TIE (+0.00008). No negative flip on P-gate. p_alive is CONSTANT (1.0) - the
  fold frequency (active days ~110) makes BG/NBD treat everyone as alive.
  exp_30d is a non-linear (f,r,T) interaction the GBM gains from - candidate.

- 2026-08-19 CORRECTION: the first btyd A/B was contaminated (palive cache was
  derived from actor3 which included the REJECTED funnel features). Fixed: funnel
  features gated behind ECUP26_FUNNEL_FEATURES (off by default); palive rebuilt
  from the clean actor cache (155 cols, no funnel). Clean re-run confirms:
  btyd_exp_30d alone: B +0.00029 (P=0.01, 99%), A TIE (-0.00002), P TIE (+0.00008).
  Production script now: ECUP26_ACTOR_FEATURES=1 (actor only) + btyd_exp_30d.

- 2026-08-19 btyd integrated into production (actor+btyd, no funnel): submit_final_blend.csv
  regenerated (250k rows, unique, >=0, mean 35.42). fold-B blend with btyd: 1.6824
  (weights shifted to 0.4/0.6). BG/NBD on the full train needs the penalizer fallback
  (0.0 -> 0.01 -> 0.1 -> 1.0) - 0.0 alone did not converge. Overnight re-tuning
  launched (LGBM+CatBoost coordinate on palive cache).

- 2026-08-19 btyd DEGRADED on LB: 1.655630374 vs actor-only 1.655574949 (+0.00006).
  btyd was fold-B-only improvement (A/P TIE) - 5th 'fold-B mirage' (after shrink,
  hybrid, bucket-cal, funnel). REFINED RULE: integrate ONLY if ALL THREE folds
  (B+A+P) show improvement (P<=0.1), not merely 'no flip'. btyd OFF in production
  (--btyd flag default off). Production regenerating actor-only.
- 2026-08-19 submissions/ folder + INDEX.md added (date-suffixed names, results table).

- 2026-08-19 RETUNE CONFIRMED (first hyperparam change passing B+A+P):
  blend_retuned (LGBM 650/120/0.02/0.6/0.8/21 + CatBoost 900/0.03/9/10/1.0,
  weights 0.45/0.55) vs current: B +0.00046 (P=0.00), A +0.00013 (P=0.02),
  P +0.00031 (P=0.00). ALL improve, no flip -> integrated into production.
- 2026-08-19 T3.10.18 Markov chains 13-21 (wake-up analysis):
  13-18 (post-wake-up details) ALL FLAT: buy_rate ~0.705 regardless of dormant_days,
  first step, intensity, holiday, check size. 19: wake calendar grows with onboarding
  (no monthly holiday peak). 20 time-to-buy: median 4d, 22% day0, 39%<=2d; conditional
  buy_rate DECLINES GRADUALLY 0.705->0.651->0.621->...->0.467 (no sharp 20% cliff).
  21: channel intensity differs (search: 2.74 searches, 0.79 cart vs cat 0.29) but
  buy_rate flat. SYNTHESIS: the ONLY spike-window signal = Dormant->Active wake-up
  RATE (2x gift/NY); post-wake-up behavior is deterministic ~70.5%. gift_activation_boost
  must encode the wake-up RATE boost, not post-wake-up details.
- 2026-08-19 T3.10.18 E1 gift_activation_boost (sigma=14, dormant x 0.18 x gauss) A/B
  M=500 (mode=test, cache-a=actor vs cache-b=actor+gift_boost): fold B WINS +0.000278
  (P=0.01, CI [+0.000048,+0.000530]), fold A TIE-lean +0.000120 (P=0.16), fold P TIE
  -0.000036 (P=0.62). STRICT RULE violated (P<=0.1 on ALL of B+A+P) -> NOT INTEGRATED.
  6th candidate failing B+A+P after btyd; fold-B-only win does not transfer.
- 2026-08-19 production actor_retuned COMPLETE (re-run after OOM; OOM-guard lowered to 5 GB):
  submit_blend_actor_retuned_2026-08-19.csv (250000 rows, R10 valid, mean 35.29, zeros 0.25%).
  Blend fit on fold B: LGBM 0.4 / XGB 0.0 / Cat 0.6 (fold B 1.6820). Walk-forward 8 origins,
  143 feats, ECUP26_NUM_THREADS=4. corr 0.995 with actor-only #11 -> same ranking, -0.085 level.
  First run ABORTed at 12 GB guard (10.2 free); guard lowered to 5 GB; fit started at 11.2 GB free.
- 2026-08-20 T3.10.27 gap-analysis (blend_retuned on fold B actor3 + gift-proxy G gp,
  scripts/gap_analysis.py): fold B RMSLE 1.6824 zero_frac 0.459 -> zeros 45.9% users /
  50.0% error2 (bias -1.42 over), positives bias +0.83, whales top-1% 2.5% err2
  (bias +2.14), top-5% 9.1% err2 (bias +1.74). fold G RMSLE 1.899 zero_frac 0.517
  (all 209k users) -> zeros 56.3% err2 (bias -1.80), gift even MORE zero-heavy than B.
  E3: zeros = main lever (shrink regime-dependent, P-flip), whales <=9% err2 so
  T3.10.19 REJECTED is consistent, gift zero-heavy -> T3.10.28 fold G = mandatory gate.
- 2026-08-20 T3.10.16 SHAP diagnostic (fold B actor3, prod-LGBM 650/120, TreeExplainer
  100k; scripts/shap_diagnostic.py): top-20 by mean|SHAP|: to_ord_180d 0.433 DOMINATES
  (3x to #2), then to_ord_120d/total/90d, gmv_total 0.063, recency_gmv 0.062.
  Order-count > gmv-value ('gmv dominates' hypothesis FALSIFIED). zeros-vs-whales
  divergence also in order volume: to_ord_180d diff +0.974, 120d +0.366, total +0.211,
  90d +0.177, gmv_total +0.160 -> whales = tail of same order signal (input for
  T3.10.14 experts = order-volume features). Feature plateau confirmed (top feats
  already in prod); T3.9.1 pruning verdict consistent.
- 2026-08-20 T3.10.28 gift-fold G EXTENDED + VALIDATED (scripts/rebuild_folds_gift.py,
  OUT=/tmp/bench_t3_folds_giftG, walk-forward step 7d): G cutoff 02-14 3 origins /
  353k rows, G2 02-23 4 origins / 599k, G3 03-08 6 origins / 955k (actor feats 153
  cols). truth zero_frac G = 0.4840 == note 003 (0.484). E2 (ab_validate test M=500):
  on G blend_retuned WINS vs baseline +0.0102 (P=0.00) - the old fold-G blend loss
  (-0.0058) WAS a CatBoost-overfit artifact on 168k rows; on G3 baseline WINS
  -0.0032 (post-gift regime != target window). G is now a MANDATORY gate: chain
  B+A+P+G (integrate only if P<=0.1 on ALL FOUR). shrink c=0.95 on G WINS +0.0140
  (P=0.00) - gift is zero-heavy, shrink confirmed on the extended fold.
- 2026-08-20 T3.10.15 holiday pre/post flags: REJECTED (sign-flip). Dual-cache
  A/B actor3 vs +6 flags (rebuild_folds_holiday.py, /tmp/bench_t3_folds_holiday,
  M=500): fold B -0.00028 (P=0.98 noflags wins), fold A +0.00012 (P=0.14 TIE),
  fold P +0.00028 (P=0.01 flags wins). B vs P sign-flip -> no B+A+P consistency
  (flags are noise-level, +-0.0003 < 0.0005). Feature plateau confirmed; flags
  stay behind ECUP26_HOLIDAY_PREPOST_FEATURES=1 (OFF in prod), leak-free tests kept.
- 2026-08-20 T3.10.22 quantile-spread feature: REJECTED. m_quantile_spread_feat
  (q0.9-q0.5, q0.75-q0.5, 300 trees, user-hash OOF) vs baseline, M=500: fold B
  -0.00069 (P=1.00), fold A -0.00034 (P=0.99) - baseline wins both, consistent
  negative. Volatility already covered by gmv_cv_daily / buy_gap_cv (T3.10.16
  top-20). Feature plateau again; code kept in ab_validate.py (dev-only).
- 2026-08-20 T3.10.21 target-encoding segments: REJECTED. m_target_encode_segments
  (OOF mean-y per k-means k=8, user-hash OOF, leak-free) vs baseline, M=500:
  fold B -0.00031 (P=0.99 baseline), fold A -0.00006 (P=0.71 TIE). No positive
  signal; same mechanism as kNN T3.10.7 / cluster-as-feature T3.10.14 - smoothed
  group label = noise, tree already recovers segments from raw sums.
- 2026-08-20 T3.10.24 multitask NN ATTEMPT: FAIL gate. scripts/experiment_neural_multitask.py
  (shared GRU + clf/reg heads, dense 90d tokens, 4 origins, RTX 5060): fold-B proxy
  1.7251 >= gate 1.70 -> closed. ~equal to dense+MSE single-head 1.7262; BCE head
  learns (0.49) but does not help the regression. Neural path exhausted (5 attempts).
- 2026-08-20 T3.10.23 conformal threshold ATTEMPT (diagnostic, scripts/conformal_threshold.py):
  split conformal on fold B, alpha=0.1, cal 20%: q_hat=0.6902 (zero-membership iff
  P(y>0)<=0.69) vs heuristic t=0.05; implied zero-membership 59.7%; empirical coverage
  0.9079 (guarantee holds). Explains T3.10.2 falsification: heuristic zeroed almost
  always while 69% confidence is needed to exclude zero. No zeroing applied.
