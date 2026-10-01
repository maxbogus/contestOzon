# E-CUP T3 — Retry-кандидаты: полная история вариантов (2026-08-14..20)

> Собран из: git log, docs/notes/003, EXPERIMENTS.md, submissions/INDEX.md,
> docs/research/*, docs/backlog/tickets (contest-root + ecup26-user-value).
> Машиночитаемая версия: docs/research/retry_candidates.jsonl

| id | variant | category | status | date | context.features | context.model | context.methodology | context.knowledge | evidence | reject_reason | regime_dependent | new_since | retry | retry_with |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| btyd-segmented-baseline | BTYD segmented baseline | segmentation | TRIED-KEPT | 2026-08-16 | ~22 feats (calendar+rolling) | SegmentedBTYD | single-origin, in-sample | нет | baseline scaffold |  | no | walk-forward, actor | NO | — |
| walkforward-fullschema | walk-forward + full-schema (63 feats) | features | TRIED-KEPT | 2026-08-17 | 63 feats | LGBM single-origin | CV 1.7x | нет | proxy 1.7070 (база 1.7309) |  | no | actor, fold G | NO | — |
| tweedie-target | tweedie target transform | target-transform | TRIED-REJECTED | 2026-08-17 | 63 feats | LGBM tweedie | CV 2-fold | нет | 2.53 vs 1.71 | noise | no | — | NO | — |
| ensemble-lgb-xgb-cat | ensemble lgb+xgb+cat | ensemble | TRIED-REJECTED | 2026-08-17 | 90 feats | LGBM+XGB+Cat | CV 1.7x | нет | 1.7171 vs 1.7176 noise | noise | no | retune | MAYBE | retuned blend weights |
| features-v2-windows | feats v2 (60/90d+recency) | features | TRIED-KEPT | 2026-08-17 | 90 feats | LGBM 200/31/0.05 | CV | нет | 1.6910 |  | no | actor | NO | — |
| lgbm-tuning | LGBM tuning 200/31/0.05/0.8/0.8 | ensemble | TRIED-KEPT | 2026-08-17 | 90 feats | LGBM | CV | нет | 1.6950 |  | no | retune | NO | — |
| features-v3 | feats v3 (holiday dist, dow, convs) | features | TRIED-REJECTED | 2026-08-17 | 90 feats | LGBM | CV | нет | 1.7170 noise | noise | no | — | NO | — |
| holiday-promo-gift | T3.8.7 holiday/promo/gift features | features | TRIED-KEPT | 2026-08-17 | 138 feats | LGBM | CV | нет | 1.7155 (fold B 1.688) |  | no | actor | NO | — |
| walkforward-step15 | walk-forward step 15d (16 origins) | other | TRIED-REJECTED | 2026-08-17 | 138 feats | LGBM | CV | нет | 1.7158 vs 1.7155 | noise | no | fold G | NO | — |
| gift-season-labels-12or | 12 origins incl gift 2025 label | other | TRIED-REJECTED | 2026-08-17 | 138 feats | LGBM | CV | нет | 1.7155 = baseline | noise | no | fold G | NO | — |
| season-only-model | season-only Feb-Mar 2025 | segmentation | TRIED-REJECTED | 2026-08-17 | 138 feats (short history) | LGBM | fold B | нет | 2.0043 vs 1.6881 | noise | no | fold G | NO | — |
| two-stage-raw | two-stage p_buy x amount (raw) | zero-inflation | TRIED-REJECTED | 2026-08-17 | 138 feats | LGBM clf+reg | fold B | нет | 2.02 | noise | no | log-space insight | NO | — |
| weighted-single-stage | weight=clip(1+log1p(gmv),1,6) | zero-inflation | TRIED-REJECTED | 2026-08-17 | 138 feats | LGBM weighted | fold B | нет | 2.09 | noise | no | whale bias +1.74 | NO | — |
| log-calibration-bias | additive log-space bias (b~0) | calibration | TRIED-KEPT | 2026-08-17 | 138 feats | LGBM | fold B | нет | 1.7068, b≈0 (нейтральна) |  | no | shrink | NO | — |
| hybrid-oof-purchase-prob | hybrid OOF purchase_prob feature | features | TRIED-DEGRADED-LB | 2026-08-17 | 138 feats | LGBM + OOF P(y>0) | CV + LB | нет | CV 1.7123 (-0.0032), LB 1.6672 (+0.0101) | LB-degraded | yes | actor-фичи, fold G | MAYBE | OOF P(y>0) на actor-фичах + гейт G |
| benchmarker-nest300 | benchmarker n_estimators 300 | ensemble | TRIED-KEPT | 2026-08-17 | 138 feats | LGBM | CV | нет | 1.6901 (CV 1.7168) |  | no | retune | NO | — |
| feature-selection-prune | T3.9.1 gain-rank pruning | features | TRIED-REJECTED | 2026-08-18 | 114 feats | LGBM | fold B | gain 90d=45.7% | K=20..114 -> 1.6986..1.6882, full=1.6882 | noise | no | — | NO | — |
| rolling-windows-120-180 | +120/180d windows | features | TRIED-KEPT | 2026-08-17 | 138 feats | LGBM | fold A+B | нет | CV 1.7150 (-0.0005) |  | no | actor | NO | — |
| gp-synthesis | gplearn feature synthesis | features | TRIED-REJECTED | 2026-08-17 | 114 feats | LGBM | fold B | нет | rejected (см T3.10.25) | noise | no | — | NO | — |
| tree-leaf-encoding | leaf-membership one-hot features | features | NOT-TESTED | 2026-08-17 | 114 feats | LGBM | — | hybrid LB-degraded | ? | — | no | actor, fold G | MAYBE | leaf-encoding на actor-фичах + G |
| neural-ziln | neural ZILN head | NN | TRIED-REJECTED | 2026-08-17 | 90d dense tokens | GRU+ZILN | fold B | нет | 2.58 | noise | no | — | NO | — |
| neural-dense-mse | dense+MSE GRU | NN | TRIED-REJECTED | 2026-08-17 | 90d dense tokens | GRU single-head | fold B | нет | 1.7262 vs LGBM 1.6861 | noise | no | multitask insight | NO | — |
| neural-transformer | transformer over tokens | NN | TRIED-REJECTED | 2026-08-17 | 90d dense tokens | Transformer | fold B | нет | хуже LGBM | noise | no | — | NO | — |
| features-t3911 | T3.9.11 3 feature gaps | features | TRIED-REJECTED | 2026-08-18 | 138 feats | LGBM | fold B | нет | hurt B -0.00017 (P=0.93) | noise | no | — | NO | — |
| bucket-calibration | per-bucket post-hoc calibration | calibration | TRIED-DEGRADED-LB | 2026-08-18 | 138 feats | LGBM + bucket bias | fold B cross A->B + LB | не знали gift regime | fold B -0.0100, LB 1.6672 (+0.0101) | LB-degraded | yes | fold G (0.484), optuna-joint | NO | — |
| blend-lgbm-catboost | blend LGBM 0.3 + CatBoost 0.7 | ensemble | TRIED-KEPT | 2026-08-18 | 138 feats | blend log-space | fold B M=500 | CatBoost>=LGBM | B +0.001065 (P=0.00) |  | no | retune | NO | — |
| zero-threshold-hard | hard-zero threshold P(y>0) | zero-inflation | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM + clf | fold B sweep+bootstrap | zero_frac 0.459 | best t=0.05 (=нет порога), t=0.5 -> +0.10 | noise | no | conformal q=0.69 | NO | — |
| log-hurdle | log-space hurdle P(y>0) x E[log1p\|y>0] | zero-inflation | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM clf+reg | fold B+A M=500 | zero_frac A=0.437 B=0.459 | B +0.00043 (P=0.04), A -0.00178 (P=1.00) sign-flip | sign-flip | yes | fold G (0.484), OOF-P(y>0) метод | MAYBE | hurdle на G, сегментный OOF P(y>0) |
| absence-features | absence/gap features | features | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM | fold B M=500 | нет | +0.00006 TIE (P=0.30) | noise | no | — | NO | — |
| catboost-log1p | CatBoost log1p candidate | ensemble | TRIED-KEPT | 2026-08-18 | 143 feats | CatBoost | fold B+A M=500 | нет | B +0.00087 (P=0.00), A TIE-lean |  | no | retune | NO | — |
| catboost-tweedie | CatBoost tweedie (raw) | target-transform | TRIED-REJECTED | 2026-08-18 | 143 feats | CatBoost tweedie | fold B M=500 | нет | 2.4687 vs 1.6857 | noise | no | — | NO | — |
| quantile-blend | quantile regression blend | target-transform | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM quantile | fold B M=500 | нет | 2.1605 vs 1.6857 | noise | no | — | NO | — |
| knn-feature | kNN neighbor-mean feature | features | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM + knn_y | fold B M=200 | нет | 1.685992 vs 1.685742 (-0.00025) | noise | no | — | NO | — |
| ordinal-residual | ordinal bucket + residual reg | calibration | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM multiclass+reg | fold B M=500 | нет | 1.687319 vs 1.685742 (-0.00158) | noise | no | — | NO | — |
| behavioral-timing | T3.10.12 behavioral timing | features | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM | fold B+A M=500 | нет | B +0.00006 TIE, A +0.00019 | noise | no | — | NO | — |
| shrink-c095 | log-space shrink c=0.95 | calibration | TRIED-DEGRADED-LB | 2026-08-18 | 143 feats incl actor | blend LGBM 0.3+Cat 0.7 (log) | fold B+A bootstrap M=500, без G | zero_frac 0.46, whale +0.84; НЕ знали gift 0.484 | B +0.00946 (P=0.00), A +0.00233; LB 1.6702 (+0.0137) | LB-degraded | yes | fold G (0.484; shrink +0.0140 подтверждён), whale bias +1.74 | YES | сегментный shrink (нули вниз, киты вверх) + гейт G |
| kmeans-cluster | kmeans cluster-as-feature | segmentation | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM + cluster | fold B+A+P | whale +0.84 | B -0.00033, A TIE, P -0.00037 | noise | no | actor-фичи | NO | — |
| kmeans-moe | kmeans Mixture of Experts | segmentation | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM per-cluster | fold B+A+P | whale +0.84 | B +0.00054 (P=0.01), A/P TIE | inconclusive | yes | actor-feature space, whale bias +1.74, SHAP order-vol | MAYBE | whale-expert на actor-фичах + гейт G |
| actor-features | actor behavioral features | features | TRIED-KEPT | 2026-08-18 | 143 feats (incl actor) | blend + actor | fold B+A+P M=500 | target spike-like | B +0.00232, A +0.00149, P +0.00244 (все P=0.00) |  | no | fold G | NO | — |
| whale-oversampling | whale row copies | zero-inflation | TRIED-REJECTED | 2026-08-18 | 143 feats | LGBM + whale copies | fold B+A+P M=500 | whale bias +0.84 | B -0.0036, A -0.0019, P -0.0019 | noise | no | whale <=9% err2 (gap) | NO | — |
| targeted-interactions | domain feature interactions | features | TRIED-REJECTED | 2026-08-18 | 143 feats (actor) | LGBM | fold B+A+P M=500 | нет | B +0.00029 (P=0.01), A TIE, P -0.00042 FLIP | sign-flip | yes | fold G (gift regime) | MAYBE | пере-тест на G (помогали lull≈gift) |
| funnel-chains | funnel-chain coefficients | features | TRIED-REJECTED | 2026-08-18 | 143 feats (actor) | LGBM | fold B+A+P M=500 | нет | B +0.00044 (P=0.00), A TIE, P -0.00022 FLIP | sign-flip | yes | fold G (gift regime) | MAYBE | пере-тест на G |
| cart-to-order-lag | cart->order lag probe | features | TRIED-REJECTED | 2026-08-19 | 143 feats | EDA (2000-user sample) | sample | нет | lag=0 для 100% заказов (структурно мёртв) | noise | no | — | NO | — |
| markov-gift-boost | markov gift_activation_boost | features | TRIED-REJECTED | 2026-08-19 | 143 feats (actor) | LGBM | fold B+A+P M=500 | Dormant->Active 1.72x | B +0.000278 (P=0.01), A TIE (P=0.16), P TIE (P=0.62) | inconclusive | yes | fold G (gift regime) | MAYBE | пере-тест на G (gift-specific) |
| btyd-exp30d | btyd_exp_30d feature | features | TRIED-DEGRADED-LB | 2026-08-19 | 143 feats (actor) | blend + btyd | fold B+A+P + LB | p_alive константа | B +0.00029 (P=0.01), A/P TIE; LB +0.00006 | LB-degraded | yes | fold G | MAYBE | btyd на G |
| retune-blend | retuned blend LGBM 650 + Cat 900 | ensemble | TRIED-KEPT | 2026-08-19 | 143 feats (actor) | LGBM 650/120 + Cat 900, 0.45/0.55 | fold B+A+P M=500 | 5 fold-B миражей | B +0.00046, A +0.00013, P +0.00031 (все P<=0.02) |  | no | fold G | NO | — |
| gap-analysis | gap-анализ до top-1 | other | TRIED-KEPT | 2026-08-20 | 155 feats (actor3) | blend_retuned | fold B + gift-G | нули 50% err2 | B: zeros 50% err2 (bias -1.42), whales top-5% 9.1% (bias +1.74) |  | no | — | NO | — |
| shap-diagnostic | SHAP-диагностика | other | TRIED-KEPT | 2026-08-20 | 161 feats (actor) | prod-LGBM 650/120 | fold B 100k sample | gain 90d доминирует | to_ord_180d 0.433 (3x #2); gmv НЕ доминирует |  | no | — | NO | — |
| gift-fold-G | расширенный fold G (gift-гейт) | other | TRIED-KEPT | 2026-08-20 | 153 feats (actor) | blend_retuned | B+A+P+G | gift zero_frac 0.484 | G: 353k строк, blend +0.0102 (P=0.00), shrink +0.0140 (P=0.00) |  | no | — | NO | — |
| holiday-prepost-flags | T3.10.15 holiday pre/post flags | features | TRIED-REJECTED | 2026-08-20 | 159 feats | baseline LGBM | fold B+A+P M=500 | gift zero-heavy | B -0.00028 (P=0.98), A TIE, P +0.00028 (P=0.01) sign-flip | sign-flip | yes | — | NO | — |
| quantile-spread-feature | T3.10.22 quantile-spread feature | features | TRIED-REJECTED | 2026-08-20 | 161 feats | baseline LGBM | fold B+A M=500 | нет | B -0.00069 (P=1.00), A -0.00034 (P=0.99) | noise | no | — | NO | — |
| target-encoding-segments | T3.10.21 OOF mean-y per cluster | features | TRIED-REJECTED | 2026-08-20 | 161 feats | baseline LGBM | fold B+A M=500 | нет | B -0.00031 (P=0.99), A TIE | noise | no | — | NO | — |
| multitask-nn | multitask GRU (2 heads) | NN | ATTEMPT-FAIL-GATE | 2026-08-20 | 90d dense tokens | GRU+BCE/MSE | fold B proxy | neural path exhausted | 1.7251 >= gate 1.70 (≈dense 1.7262) | gate-fail | no | — | NO | — |
| conformal-threshold | conformal zero threshold | calibration | ATTEMPT-FAIL-GATE | 2026-08-20 | 161 feats | LGBM clf | fold B split-conformal | hard-zero falsified | q_hat=0.6902 vs эвристика 0.05; cov 0.9079 | gate-fail | no | — | NO | — |
| unsupervised-seq-pretrain | T3.10.9 MAE/SimCLR embeddings | NN | NOT-TESTED | 2026-08-18 | ? | GBDT + embeddings | — | neural exhausted | ? | — | no | multitask result | NO | — |
| cnn-attention-nbeats | T3.10.10 CNN+attention/N-BEATS | NN | NOT-TESTED | 2026-08-18 | 90d dense tokens | CNN+attn | — | sparse-GRU 1.7153 | ? | — | no | multitask result | NO | — |
| optuna-joint-calibration | T3.10.11 joint Optuna (thr+bias+weights) | calibration | NOT-TESTED | 2026-08-18 | 143 feats | blend | — | bucket-cal LB-degraded lesson | ? | — | no | fold G (мульти-фолд гейт) | MAYBE | joint на B+A+P+G |
| whale-expert-segment | T3.10.14 E2 whale expert | segmentation | NOT-TESTED | 2026-08-18 | 143 feats | LGBM expert | — | whale bias +0.84 | ? | — | no | whale bias +1.74, SHAP order-vol, fold G | MAYBE | whale-expert + segment shrink |
| funnel-v2-chains | T3.10.26 v2 chains 6/7/9/11 | features | NOT-TESTED | 2026-08-19 | 143 feats | LGBM | — | funnel v1 P-flip | ? | — | no | fold G | MAYBE | изолированный тест на G |
| markov-sweep | T3.10.18 E2 sigma/peak sweep | features | NOT-TESTED | 2026-08-19 | 143 feats | LGBM | — | Dormant->Active 1.72x | ? | — | no | fold G | MAYBE | sweep + G |
| multi-horizon-heads | GRePO multi-horizon aux heads (7d/14d/30d) | NN | NOT-TESTED | 2026-08-16 | 143 feats | LGBM + aux heads | — | нет | ? | — | no | — | MAYBE | aux heads (дешёво) |
| hawkes-point-process | Hawkes point process | other | NOT-TESTED | 2026-08-18 | — | — | — | sparse supervision | ? | — | no | — | NO | — |
| hier-bayes | full hierarchical Bayes | segmentation | NOT-TESTED | 2026-08-18 | — | — | — | BTYD partial | ? | — | no | — | NO | — |
