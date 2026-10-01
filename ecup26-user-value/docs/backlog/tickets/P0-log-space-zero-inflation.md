# P0 — Zero-inflation в log-спейсе (T3.10.2–T3.10.5)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Протокол: `docs/EXPERIMENTS.md`. Критерий: знаковая согласованность на
> бутстреп-фолдах (K=10), НЕ точечная дельта одного фолда.
> Reference (fold B): refined LGBM **1.6857** (note 003). LB #78 = 1.656517,
> #1 = 1.645914 → разрыв 0.0106.

---

## T3.10.2 — Zero-inflation threshold (hard-zero в log-спейсе)

- **Hypothesis**: 45.9% нулевых юзеров дают ~50% error^2; baseline предсказывает
  им ~7.7 GMV (условное среднее в log-спейсе). Для squared-error оптимальный
  предиктор — HARD ZERO. Порог применяем к P(y>0) в log-спейсе
  (`pred_log = 0` там, где P < t) — не в сыром, как в двухстадийнике.
- **Expectations**: минимум RMSLE при t≈0.4–0.6; выигрыш 0.001–0.005 на fold B;
  бутстреп: zero_threshold побеждает baseline на ≥90% сидов (P(A beats B) ≥ 0.90).
- **Series**:
  - E1: свип t ∈ {0.1,…,0.9, шаг 0.1} на fold B → таблица rmsle(t).
  - E2: бутстреп A/B (K=10) лучшего t против baseline.
  - E3: (если E2 выиграл) свип конфига классификатора.
- **What to check**: rmsle(t) минимум; P(A beats B); 95% CI не пересекает 0.
- **How to verify**: `uv run python scripts/ab_validate.py --a baseline --b zero_threshold --t <best> --seeds 10`.
- **Sweeps**: t-грид; классификатор (num_leaves 19/31, scale_pos_weight 1.0/1.1, rounds 150/300).
- **Compliance**: R1 (lightgbm), R6 (seeded), R8 (dev-only).
- **Forbidden**: не применять порог в сыром спейсе (двухстадийник 2.02 — мёртв);
  не подгонять t на одном фолде — только свип + бутстреп.
- **Result** (2026-08-18): REJECTED — FALSIFIED.
  E1 sweep: best t=0.05 (=no threshold); monotonic degradation (t=0.10 +0.0002,
  t=0.50 +0.10). E2 bootstrap (t=0.1, M=500): baseline wins 100% of draws
  (delta -0.00017, CI [-0.00026,-0.00009]). Hard-zero in log-space does not work.


---

## T3.10.3 — Log-space hurdle: pred_log = P(y>0) · E[log1p(y)|y>0]

- **Hypothesis**: raw two-stage (p_buy × amount) провалился (2.02), потому что
  умножал в СЫРОМ спейсе. В log-спейсе `E[log1p(y)] = P1 · E[log1p(y)|y>0]` —
  это RMSLE-оптимальная связка (см. EXPERIMENTS.md §5).
- **Expectations**: hurdle ≤ baseline на fold B; бутстреп: hurdle побеждает на
  ≥90% сидов ИЛИ TIE с K>10; gain ~0.0005–0.003.
- **Series**:
  - E1: A/B hurdle vs baseline (K=10, fold B).
  - E2: тюнинг позитивного регрессора (L2 vs Huber, num_leaves 41/97).
  - E3: (опция) OOF-версия P1 (leave-origin-out) для снижения переобучения множителя.
- **What to check**: sign-consistency; CI; первичная дельта на fold B.
- **How to verify**: `uv run python scripts/ab_validate.py --a baseline --b hurdle --seeds 10`.
- **Sweeps**: loss позитивного регрессора (L2/Huber α=0.9), num_leaves, раунды clf.
- **Compliance**: R1, R6, R8.
- **Forbidden**: НЕ возвращаться к raw-space перемножению; не копировать
  двухстадийный код benchmark_t3.run_two_stage.
- **Result** (2026-08-18): REJECTED — SIGN-FLIP across folds.
  fold B: hurdle WINS (delta +0.00043, P(A beats B)=0.04, CI [-0.00002,+0.00087]);
  fold A: baseline WINS (delta -0.00178, P=1.00). zero_frac truth: A=0.437 (NY),
  B=0.459 (lull). Hurdle helps only the higher-zero fold -> would likely HURT on LB.
  Log-space formula retained as reference; not a universal win.


---

## T3.10.4 — Absence-фичи (нулевые дни)

- **Hypothesis**: данные — прореженные ряды (30.6M строк из ~102M полной сетки,
  task3: «прореженные ряды… 100 млн»). Модель не отличает «не заходил» от
  «не купил». Фичи «отсутствия» закроют это.
- **Expectations**: gain на fold B, НО главный критерий — sign-consistency
  (feats v3/holiday были noise и REJECTED). Ожидание: −0.0005..−0.002 при P≥0.90.
- **Series**:
  - E1: `n_zero_days_7/14/30`, `days_since_last_{search,cart,order}`, recency каналов → A/B.
  - E2: комбинации окон {7,14,30} × {count, share}.
- **What to check**: sign-consistency на K=10; НЕ обманываться единственной дельтой.
- **How to verify**: реализовать в build_features (за флагом), прогнать бутстреп A/B.
- **Sweeps**: окна 7/14/30; определение «нулевого дня» (нет строки vs gmv=0).
- **Compliance**: R1, R6, R8.
- **Forbidden**: не добавлять праздничные/сезонные фичи (T3.8.7/12/13 — noise);
  не менять DEFAULT_ROLLING_WINDOWS без бутстрепа.
- **Result** (2026-08-18): TIE — NOT integrated.
  t3911 vs abs (fold B, M=500): delta +0.00006 toward absence features, P(A beats B)=0.30,
  CI [-0.00019,+0.00033]. Effect at the noise floor (std 0.00014). Features stay behind
  ECUP26_ABSENCE_FEATURES=1 (no production change).
  BONUS: the T3.9.11 features (n_active_7/14d, has_cart_no_order_30d, gmv_90_180d)
  HURT fold B by -0.00017 (P=0.93) - they were never fold-A/B'd; removal candidate.

---

## T3.10.5 — CatBoost: log1p A/B + Tweedie-диагностика

- **Hypothesis**: CatBoost tuned 1.6849 < LGBM 1.6857 на fold B (T3.9.13). Нужно
  проверить, реально ли это (sign-consistency). Отдельно: CatBoost Tweedie(raw)
  должен быть ХУЖЕ — это подтвердит тезис «log-спейс оптимален» (LGBM tweedie 2.53).
- **Expectations**: E1 — CatBoost log1p ≥ LGBM по sign-consistency; E2 — Tweedie
  хуже на 0.3+ (ожидание на основе LGBM tweedie 2.53).
- **Series**:
  - E1: A/B CatBoost(log1p) vs baseline LGBM (K=10).
  - E2: CatBoost Tweedie(raw) vs CatBoost log1p — диагностика.
  - E3: (если E1 выиграл) бленд LGBM+CatBoost весами {0.3/0.7, 0.5/0.5}.
- **What to check**: P(A beats B); CI; fold-B дельты обеих моделей.
- **How to verify**: расширить ab_validate методами `cat_lgb`/`cat_tweedie`.
- **Sweeps**: depth 6/9, lr 0.02/0.03, iterations 550/700.
- **Compliance**: R1 (catboost Apache-2.0), R6, R8. CatBoost — dev-группа, не runtime.
- **Forbidden**: не переносить catboost в runtime-зависимости без необходимости;
  не считать одиночную дельту 1.6849 vs 1.6857 «выигрышем».
- **Result** (2026-08-18): E1 CatBoost(log1p) KEEPS as candidate: fold B WINS 100%
  (delta +0.00087, CI [+0.0005,+0.0012]), fold A TIE-with-lean (P=0.17, 83% toward
  CatBoost), NO sign-flip -> CatBoost >= LGBM on both folds. E2 CatBoost Tweedie REJECTED
  (2.4687 vs 1.6857, baseline wins 100%) — raw-space thesis confirmed.
  Next: train-mode deep check + blend-weight bootstrap before production.

