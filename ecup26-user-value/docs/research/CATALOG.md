# CATALOG — единый каталог исследования E-CUP 2026 Task 3 (GMV-прогноз)

> **Назначение**: единая точка входа в исследовательский материал T3 после финала
> конкурса (2026-08-31). Заменяет разрозненные чтения HANDOFF / reviews / tickets —
> здесь один индекс, все выводы и ссылки на источники.
>
> **Порядок чтения**: этот файл → `decision_ledger_v2.jsonl` (вердикты) →
> `submissions/INDEX.md` (LB-история) → `T3_neural_findings.md` (нейро) →
> `EXPERIMENTS.md` (протокол) → `reviews/` (хронология).

## 1. Задача, цель, итог

- **Задача**: предсказать суммарный GMV за 30 дней для 250 000 пользователей Ozon
  из daily-агрегатной истории (без item_id). Метрика **RMSLE** (log1p-MSE), ниже = лучше.
  Сабмит — CSV `user_id,predict`.
- **Цель конкурса**: private top-1 (RMSLE < 1.6457). Топ-1 по факту = **1.645914**
  (Daniil Stepanov).
- **Наш итог**: private **1.6655516098 (ранг 59)**, public **1.6503662713 (ранг 70)**.
  Цель «топ-50» (1.649604) НЕ достигнута.
- **Главный урок**: разрыв public→private **+0.0152** — всё, подогнанное под публичный
  LB (w=0.3, c=1.04, lr=0.0005), на private (80% клиентов) не перенеслось.
  LB ранжирует, private — истина.

## 2. Факт-лист (числа + источники)

| # | Факт | Число | Источник |
|---|------|-------|----------|
| 1 | Итог private / public | 1.6655516 (59) / 1.6503663 (70) | HANDOFF.md, reviews/2026-08-31.md |
| 2 | Топ-1 / топ-50 | 1.645914 / 1.649604 | reviews/2026-08-31.md |
| 3 | Табличный потолок (10 дней) | 1.651833 (calv2 c=1.04) | T3_neural_findings.md |
| 4 | Чистый GRU (w=1.0) — ХУЖЕ таблицы | 1.679 | ledger T3.19.A6 |
| 5 | Гибрид нейро×таблица (w=0.25→0.3) | 1.650409 → 1.650366 | T3_neural_findings.md |
| 6 | U-кривая веса w | 1.0→1.6793, 0.5→1.6558, 0.1→1.6510, 0.25→1.6504 | ledger T3.19.A6 |
| 7 | Окно L (90→409 дней = вся история) | 1.7095→1.6745 (−0.035) | T3_neural_findings.md |
| 8 | lr 0.0003→0.0005 | −0.0011 fold-B, перенос на LB | ledger T3.19.A2 |
| 9 | proxy-дрейф fold-B vs LB | 1.68 vs 1.65, offset −0.02..0.03 | ledger T3.19.A7 |
| 10 | mean таблица / нейро / бленд | 38.3 / 45–49 / ~40 | T3_neural_findings.md |
| 11 | BTYD → LightGBM+фичи (10k сабсемпл) | 5.5611 → 2.3189 (−58.3%) | feat_eng_2026-08-16.md |
| 12 | Нулевые пользователи | 45.9% (нули ~50% err²) | note 003, EXPERIMENTS.md |
| 13 | GRU / xLSTM / RWKV / Mamba (fold-B) | 1.6702 / 1.6736 / 2.2888 / 1.9982 | T3_neural_findings.md §4 |
| 14 | Леджер | 60 проверок (см. §3) | decision_ledger_v2.jsonl |

## 3. Леджер (decision_ledger_v2.jsonl)

`uv run python scripts/ledger_report.py` → 60 атомарных проверок:

| Статус | Кол-во |
|--------|--------|
| CONFIRMED | 18 |
| FALSIFIED | 14 |
| OBSOLETE | 19 |
| REJECT | 3 |
| PARTIAL | 1 |
| TODO (открыто) | 5 |

Открытые TODO (после финала не влияют на сабмит):
- G6.A1 — Markov sigma/peak sweep (не доделан).
- H3.A1 — сезонная нормализация фич на zero_frac-тренд (не доделана).
- H5.A1 — эмпирический калибратор proxy→LB (не доделан).
- C3.A2 (P3) — 5–10 поведенческих фич (плато не опровергнуто).
- C6.A2 (P6) — процессная политика «accept/reject только по LB».

## 4. Индекс тикетов → вердикт → статус

### T3.10.x — фичи и техники (все исполнены, вердикты в Result)

| Тикет | Вердикт | Статус |
|-------|---------|--------|
| T3.10.12 behavioral-timing | WEAK POSITIVE, не интегрировано | ✅ done |
| T3.10.13 log-space-shrink | c=0.95 win на фолдах, но FALSIFIED на LB (c<1 неверно) | ✅ done |
| T3.10.14 kmeans-segmentation | REJECTED | ✅ done |
| T3.10.15 pre-post-holiday | REJECTED (sign-flip) | ✅ done |
| T3.10.16 shap-diagnostic | диагностика выполнена | ✅ done |
| T3.10.17 actor-behavioral | **CONFIRMED — первая победа фич** | ✅ done |
| T3.10.18 markov-user-states | TBD (после E0) | ⏳ open |
| T3.10.19 whale-oversampling | REJECTED | ✅ done |
| T3.10.20 survival-pzero | WEAK POSITIVE, не интегрировано | ✅ done |
| T3.10.21 target-encoding | REJECTED | ✅ done |
| T3.10.22 quantile-dispersion | REJECTED | ✅ done |
| T3.10.23 conformal-threshold | диагностика выполнена | ✅ done |
| T3.10.24 multitask-nn | FAIL gate (нейро-путь исчерпан) | ✅ done |
| T3.10.25 targeted-interactions | REJECTED (P-flip) | ✅ done |
| T3.10.26 funnel-chains | REJECTED (P-flip) | ✅ done |
| T3.10.27 gap-analysis-top1 | выполнено | ✅ done |
| T3.10.28 gift-season-validation | выполнено (fold G внедрён) | ✅ done |

### T3.19 — нейро-направление

| Тикет | Вердикт | Статус |
|-------|---------|--------|
| T3.19 neural-sequence | закрыто: гибрид gru409 w0.3 = потолок 1.650366 | ✅ done |

### P — парадоксы (RCA-ветка)

| Тикет | Вердикт | Статус |
|-------|---------|--------|
| P0-root-cause-analysis | RCA 8 звеньев → локальный минимум 1.6555 | ✅ done |
| P0-log-space-zero-inflation | закрыто леджером (ZILN/tweedie REJECTED) | ⏸️ closed |
| P1-ensemble-and-features | закрыто леджером (ensemble=production) | ⏸️ closed |
| P1-shrink-g-paradox | resolved: C1.A2 FALSIFIED (shrink мёртв) | ⏸️ closed |
| P2-deep-and-calibration | закрыто леджером | ⏸️ closed |
| P2-hard-zero-paradox | resolved: hard-zero FALSIFIED | ⏸️ closed |
| P3-feature-plateau-paradox | открыто: C3.A2 (TODO) | ⏳ open |
| P4-ensemble-paradox | resolved: ensemble = продакшн (08-18) | ⏸️ closed |
| P5-whale-paradox | resolved: whale REJECTED | ⏸️ closed |
| P6-proxy-lb-drift | открыто: C6.A2 (TODO, процессное) | ⏳ open |

### G / H — гэпы и смелые гипотезы (superseded финалом)

| Тикет | Вердикт | Статус |
|-------|---------|--------|
| G1 segmental-correction-master | superseded (M1 FALSIFIED на LB) | ⏸️ closed |
| G2 regime-flip-retests | superseded | ⏸️ closed |
| G3 optuna-joint-calibration | superseded | ⏸️ closed |
| G4 seasonal-shrink | superseded | ⏸️ closed |
| G5 pzero-actor-combo | superseded | ⏸️ closed |
| G6 markov-e2-sweep | открыто: G6.A1 (TODO) | ⏳ open |
| H1-H5 bold-hypotheses | частично REJECT/не доделано в леджере | ⏳ open |

## 5. SOTA — литература (3 обзора, все closed)

| Обзор | Источник | Применили / отвергли |
|-------|----------|----------------------|
| AgentLTV (KDD 2026 ADS) | agent-ltv.md, arXiv:2602.21634 | взяли только bucket-diagnostics; LLM-AutoML отвергнут (R7) |
| GRePO-LTV (KDD 2025 ADS) | grepo-ltv.md, arXiv:2506.11037 | multi-horizon Pareto — применим частично; граф отвергнут (нет item_id) |
| GrowthGR (Taobao) | growth-gr.md, arXiv:2605.17994 | декомпозиция GMV — теория для сегментов; ранжирование не применимо |

## 6. Методологические выводы (5 главных)

1. **RNN одна не бьёт таблицу, гибрид — бьёт.** Чистый GRU = 1.679 (хуже таблицы 1.6518),
   но на малом весе w=0.25 в log-бленде даёт 1.6504 — «разнообразие слабой модели»
   ценнее одиночки. Ключевой научно-интересный результат (U-кривая w).
2. **Прокси врёт.** fold-B (1.68) ≠ LB (1.65), offset −0.02..0.03; переносятся только
   ранжирующие рычаги (окно, lr, w), а не «клоны». Вывод: LB — единственный ground truth.
3. **public ≠ private.** Разрыв +0.0152 — всё, подогнанное под публичный LB, не перенеслось
   на private (80%). Табличный хедж был бы надёжнее нейро-гибрида.
4. **Преждевременные «путь исчерпан» = ложные стены.** «neural exhausted» (после 5),
   «feature plateau» (после 24) — потом опровергнуты. Вердикты условны, а не вечны
   (ledger v2 + revisit_trigger).
5. **Человек vs LLM-агент.** Агент быстро генерит фичи (5.56→2.32) и переборы, но
   «туннелится» (6 fold-B миражей); человек ре-открывает мёртвые пути и опрокидывает
   собственные фолды (инфляция c>1.0 → 1.6518). Тандем работает в разделении ролей.

---

*Сгенерировано из decision_ledger_v2.jsonl + submissions/INDEX.md + tickets. См. также HANDOFF.md.*
