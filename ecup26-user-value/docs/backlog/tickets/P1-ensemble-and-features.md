# P1 — Ансамбль и фичи (T3.10.6–T3.10.8)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Критерий: бутстреп sign-consistency (docs/EXPERIMENTS.md §1).

---

## T3.10.6 — Quantile regression (log-спейс) + бленд

- **Hypothesis**: квантильные предсказания (0.5/0.75/0.9) несут информацию о
  хвосте, которой нет в условном среднем; бленд квантилей калибрует RMSLE.
- **Expectations**: бленд ≥ baseline по sign-consistency; gain 0.0005–0.002.
- **Series**: E1: LGBM quantile для {0.5,0.75,0.9} → бленд средним; E2: 4 квантиля;
  E3: бленд с медианой.
- **Sweeps**: набор квантилей; способ бленда (mean/median/взвешенный).
- **How to verify**: метод `quantile_blend` в ab_validate.
- **Compliance**: R1/R6/R8.
- **Forbidden**: не блендить с сыро-спейсовыми моделями.
- **Result** (2026-08-18): REJECTED. Quantile blend (0.5/0.75/0.9, log-space) vs
  baseline, fold B (M=500): baseline wins 100% (2.1605 vs 1.6857). RMSLE is
  mean-based; quantiles systematically underpredict the conditional mean -> the
  blend is much worse. Theory + empirics agree.


---

## T3.10.7 — kNN-фичи + target encoding

- **Hypothesis**: средний GMV похожих юзеров (по 138 фичам) — новый сигнал,
  никогда не пробовался; kNN-фичи компенсируют хвост.
- **Expectations**: kNN-фича добавляет gain на fold B; sign-consistency на K=10.
- **Series**: E1: kNN (cosine, k=20/50/100) → neighbor_mean_log_gmv как фича;
  E2: target encoding по BTYD-сегментам; E3: комбо.
- **Sweeps**: k, метрика (cosine/euclid), число сегментов.
- **How to verify**: ab_validate с методом `knn_feat`.
- **Compliance**: R1/R6/R8.
- **Forbidden**: kNN по тестовым юзерам (leak) — только по train.
- **Result** (2026-08-18): REJECTED - now MEASURED (fast v3: PCA-10, KDTree on 150k
  reference subsample, k=25, ~4 min/run). fold B M=200: knn_feat 1.685992 vs baseline
  1.685742 (delta -0.00025), baseline wins 98%% (P=0.98, CI [-0.00045,-0.00002]).
  Smoothed-label neighbor-mean adds noise, not signal - consistent with the feature
  plateau (all feature additions TIE/noise). Larger k = more smoothing = not better.

---

## T3.10.8 — Ordinal-residual (CC-OR-Net стиль)

- **Hypothesis**: бакет-классификатор (0, 1–10, 10–100, 100–1000, >1000) +
  внутрибакетовая регрессия остатков — это «обучаемая» версия отвергнутой
  пост-калибровки (T3.9.12 LB-degrade): коррекция внутри обучения, а не пост-хок.
- **Expectations**: обученная версия должна быть sign-consistent (в отличие от
  пост-хок калибровки); gain 0.0005–0.002.
- **Series**: E1: LGBM multiclass бакет + внутрибакетовый LGBM; E2: 5 бакетов vs 4;
  E3: объединение с T3.10.2 (порог нуля).
- **Sweeps**: EDGES бакетов; число бакетов; внутрибакетовый loss.
- **How to verify**: ab_validate метод `ordinal_residual`.
- **Compliance**: R1/R6/R8.
- **Forbidden**: НЕ повторять пост-хок калибровку (LB 1.6672 degrade).
- **Result** (2026-08-18): REJECTED. Ordinal-residual vs baseline, fold B (M=500):
  baseline wins 100% (1.687319 vs 1.685742, delta -0.00158, CI [-0.0020,-0.0011]).
  Trained version worse than the already-rejected post-hoc bucket calibration.
  Buckets add no value for RMSLE.
