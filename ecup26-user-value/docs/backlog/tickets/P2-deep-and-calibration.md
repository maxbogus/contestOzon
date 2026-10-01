# P2 — Глубже и калибровка (T3.10.9–T3.10.11)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Критерий: бутстреп sign-consistency (docs/EXPERIMENTS.md §1).

---

## T3.10.9 — Unsupervised sequence pretraining → эмбеддинги в GBDT

- **Hypothesis**: эмбеддинг поведения (GRU/Transformer autoencoder) даёт фичи,
  которые GBDT сам не вытащит. Ожидание УМЕРЕННОЕ (neural path exhausted, note 003).
- **Expectations**: эмбеддинг-фичи не ухудшают и дают ≥0.0005 при sign-consistency;
  иначе — закрыть тикет, не продолжать.
- **Series**: E1: автоэнкодер (masked) → embedding 64/128 → фичи в LGBM;
  E2: contrastive (SimCLR) вариант, если E1 дал сигнал.
- **Sweeps**: dim, encoder (GRU/Transformer), доля mask.
- **Compliance**: R1 (torch dev-only), R6, R8; R4 (без интернета).
- **Forbidden**: не тратить >1 дня; не переносить torch в runtime без выигрыша.
- **Result**: TBD.

---

## T3.10.10 — CNN+attention / N-BEATS гибрид

- **Hypothesis**: чистый Transformer слаб на 409-дневных рядах; CNN (локальные
  паттерны) + attention (длинные зависимости) может быть лучше.
- **Expectations**: справочная точка — sparse-GRU 1.7153 (хуже LGBM 1.6857).
  Прогресс = < 1.70 на fold-B прокси; иначе закрыть.
- **Series**: E1: CNN+attention на dense-токенах (взять код experiment_neural_transformer);
  E2: N-BEATS-подобная.
- **Sweeps**: kernel 3/7, layers 2/4, heads 4/8.
- **Compliance**: R1/R6/R8.
- **Forbidden**: не ждать чуда — у модели жёсткий gate: proxy < 1.70.
- **Result**: TBD.

---

## T3.10.11 — Optuna совместная калибровка (порог + bias + веса бленда)

- **Hypothesis**: совместный подбор (zero-порог + bucket-bias + веса бленда) на
  НЕСКОЛЬКИХ фолдах даст устойчивый выигрыш — в отличие от T3.9.12 (один фолд,
  LB-degrade 1.6672).
- **Expectations**: выигрыш sign-consistent на A И B; НЕ деградация на LB.
- **Series**: E1: Optuna на fold A+B (цель = mean RMSLE); E2: проверить лучший
  набор на бутстрепе; E3: production-бленд.
- **Sweeps**: t∈[0,0.7], bias по бакетам, веса {lgb,cat,xgb}.
- **Compliance**: R1 (optuna open-source), R6, R8.
- **Forbidden**: подбирать калибровку на одном фолде (T3.9.12 — урок).
- **Result**: TBD.
