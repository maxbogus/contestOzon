# Ответ на «ignored_conclusions» — cross-reference с фактической историей (2026-08-19)

> Источник: JSON-анализ беседы (ignored_conclusions / unheard_user_points /
> overlooked_issues / methods_not_discussed / what_to_try_next). Сверено с тикетами
> T3.10.x, леджером docs/notes/003, docs/EXPERIMENTS.md и research-доками.

## 1. ignored_conclusions

| # | Тезис | Вердикт |
|---|---|---|
| 1 | «Главный рычаг — zero-inflation threshold, а не фичи» | ⚠️ Тезис верен, механизм нет. Нули ≈ 50% error² (ВЕРНО), но hard-zero порог FALSIFIED (T3.10.2: лучший t=0.05; t=0.5 → +0.10). Рабочий эквивалент = log-space shrink c=0.95 (T3.10.13, fold B +0.00946). Фичи-добавки (YoY/active_days/has_cart_no_order) — TIE/noise (T3.10.4 и др.). |
| 2 | «Two-stage мёртв» | ✅ ВЕРНО. raw two-stage 2.02 (T3.8.8/9); log-space hurdle sign-flip (T3.10.3); гибрид OOF purchase_prob KEPT на CV, LB DEGRADED 1.6672 (T3.8.10). |
| 3 | «Веса 1+log1p(gmv) = 2.09 провал» | ✅ ВЕРНО (T3.8.8/9). Нюанс: whale-oversampling T3.10.19 — другая схема (копии top-5%, не веса), TBD. |
| 4 | «Трансформер слаб» | ✅ ВЕРНО. neural ZILN 2.58, dense+MSE 1.7262, transformer — REJECTED. Единственный неопробованный NN — multitask-энкодер (T3.10.24, conf 0.15). |
| 5 | «Нет лимитов — T3 это CSV, не Docker» | ✅ По формату верно (task3: «создать .csv с двумя столбцами»). НО тяжёлые методы отброшены по эмпирике (провал), не из-за лимитов; реальное ограничение — 30 GB RAM + CPU (OOM). |

## 2. unheard_user_points

| # | Тезис | Вердикт |
|---|---|---|
| 1 | «Продолжали предлагать отвергнутое» | ✅ Справедливо: 6 «fold-B миражей» (shrink LB-откат, hybrid, bucket-cal, btyd, gift_boost, funnel). Методология B+A+P введена поздно (T3.10.x). |
| 2 | «YoY есть, 365/364-сдвиги как P0» | ⚠️ 365/364-day сдвиги не делали; «P0» — переоценка (все фичи — noise). |
| 3 | «Почему нет публичных разборов» | ✅ Решения публикуются после конца соревнования; в гонке — только статьи по похожим задачам (agent-ltv, grepo-ltv, growth-gr). |
| 4 | «82 место — нет gap-анализа» | ⚠️ Частично: gap 0.0106 (#78 1.6565 vs #1 1.6459) есть, но shrink НЕ перенёсся на LB → разрыв реально не закрыт → тикет T3.10.27. |

## 3. overlooked_issues

| # | Тезис | Вердикт |
|---|---|---|
| 1 | «log1p-RMSE: ~7.7 нулям, занижает китов» | ✅ Ядро T3.10.13 (shrink) — решено для нулей. Киты (bias +0.84) — ОТКРЫТО (T3.10.19). |
| 2 | «Фолды не покрывают таргет 14.02–15.03» | 🔴 ГЛАВНЫЙ ПРОБЕЛ — корень 6 миражей → тикет T3.10.28. |
| 3 | «Данные — только активные дни» | ⚠️ Тестировано: T3.10.4 absence-фичи → TIE. Ограничение данных реально, фичи не помогли. |
| 4 | «Строки с разным числом дней смещают» | ❌ НЕВЕРНО для нашей схемы: обучение на user-уровне (1 строка на user×origin), не на daily. |
| 5 | «Калибровка на одном фолде» | ✅ Подтверждено: bucket-cal LB 1.6672 DEGRADED (T3.9.12); bias b≈0; zero-threshold falsified. |

## 4. methods_not_discussed → фактический статус

| Метод | Статус |
|---|---|
| ZILN loss | REJECTED: neural ZILN 2.58 (T3.6.7/T3.9.6) |
| Hurdle (правильная связка) | REJECTED: sign-flip (T3.10.3) |
| Ordinal-residual (CC-OR-Net) | REJECTED (P1-ensemble) |
| Quantile regression | как предиктор 2.16 REJECTED; разброс-фича T3.10.22 TBD |
| Survival / time-to-purchase | T3.10.20: B +0.00029, A TIE, P TIE |
| Multitask NN | T3.10.24 TBD (conf 0.15) |
| Unsupervised seq pretrain (MAE/SimCLR) | НЕ тестировалось → REJECTED (нейро-путь исчерпан) |
| GNN | неприменимо — нет item_id (grepo-ltv §5) |
| GMM / кластеры + эксперты | T3.10.14 REJECTED |
| Target encoding | T3.10.21 TBD (conf 0.2) |
| kNN-фичи | REJECTED (T3.10.7) |
| Hawkes / point processes | НЕ тестировалось → REJECTED (sparse supervision) |
| HMM скрытые состояния | T3.10.18 NOT INTEGRATED |
| Conformal | T3.10.23 TBD (RICE 0.5, диагностика) |
| Bayesian hierarchical | частично (BTYD/BG-NBD); полный hier-Bayes нет |
| CatBoost Tweedie/custom | Tweedie 2.47 REJECTED (T3.10.5 E2) |
| LGBM custom ZI-objective | НЕ тестировалось → REJECTED (ручная версия = hurdle, отвергнута) |

## 5. what_to_try_next → фактический статус

| Идея | Статус |
|---|---|
| ZILN/hurdle loss | уже мёртв (2.58 / sign-flip) |
| CC-OR-Net | уже REJECTED |
| Absence-фичи | уже TIE (T3.10.4) |
| Quantile blend + bucket cal | уже REJECTED (2.16 / LB 1.6672) |
| Unsupervised embeddings | открыто, низкий EV → REJECTED |
| Граф похожести + mean-GMV соседей | уже kNN REJECTED |
| CatBoost custom/Tweedie | Tweedie 2.47; log1p KEEP (вес 0.6) |
| Optuna joint (threshold+bias+blend) | частично; joint = P2 TBD |
| Стабильность калибровки на фолдах | 🔴 реальный пробел → T3.10.28 |
| CNN+attention / N-BEATS | открыто, низкий EV → REJECTED |

## 6. REJECTED-направления (не возвращаться)

- raw-space zero-inflation: ZILN 2.58, LGBM tweedie 2.53, CatBoost tweedie 2.47,
  hard-zero FALSIFIED, hurdle sign-flip.
- two-stage raw 2.02 / weighted 2.09 / quantile-as-predictor 2.16.
- transformer / dense / NN (neural path exhausted: 2.58 / 1.7262).
- kNN-фичи (T3.10.7), ordinal-residual (P1), графы (нет item_id).
- Hawkes / unsupervised seq embeddings / LGBM custom ZI-objective — документированный
  отказ, НЕ тикеты (низкий EV, путь исчерпан).

## 7. Действия по результатам

- **T3.10.16 SHAP-диагностика — «делаем»** (RICE 26.7): нули vs киты профили влияния,
  input для T3.10.19 (киты) и T3.10.27 (gap-анализ).
- Созданы: T3.10.27 (gap-анализ, RICE 90.0), T3.10.28 (gift-валидация, RICE 18.9).

## 8. Оставшиеся тикеты (RICE-очередь)

| # | Тикет | Пункт | RICE | Тип |
|---|---|---|---|---|
| 1 | T3.10.27 | Gap-анализ до топ-1 | 90.0 | сделан |
| 2 | T3.10.16 | SHAP-диагностика | 26.7 | сделан |
| 3 | T3.10.19 | Киты (whale oversampling) | 20.0 | сделан (REJECTED) |
| 4 | T3.10.28 | Таргет-сезон (gift) валидация | 18.9 | сделан (fold G внедрён) |
| 5 | T3.10.15 | Holiday пре/пост-флаги | 13.3 | сделан (REJECTED) |
| 6 | T3.10.22 | Quantile-разброс фича | 4.5 | сделан (REJECTED) |
| 7 | T3.10.21 | Target encoding сегментов | 3.0 | сделан (REJECTED) |
| 8 | T3.10.24 | Multitask NN | 0.6 | попытка (сделана, FAIL gate) |
| 9 | T3.10.23 | Conformal-порог | 0.5 | попытка (сделана, диагностика) |
