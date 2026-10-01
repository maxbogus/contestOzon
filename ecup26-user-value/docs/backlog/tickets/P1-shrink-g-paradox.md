# P1 — Парадокс shrink: WIN на fold G (+0.0140), LOSE на LB (+0.0137)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Ledger: C1.A1–C1.A4. Мастер-проверка: M1 (сегментная коррекция).

## Hypothesis

Shrink c=0.95 подтверждён на fold G (P=0.00), но упал на LB (1.6702 vs 1.6565).
Три кандидата-причины: (a) fold G не совпадает с 2026-таргетом; (b) LB-сабмит
строился на другой базе, чем G-тест; (c) глобальный shrink режет китов (+1.74),
которых на реальном таргете больше, чем на G.

## Root cause link

P0-звенья 3 (нет G-гейта) + 4 (глобальная коррекция) + 5 (reject-by-default).

## Atomic checks

| ID | Проверка | Гейт | Стоимость |
|---|---|---|---|
| C1.A1 | Дифф баз: shrink-LB-сабмит vs G-тестируемая база | max_abs_diff(predict)=0 → парадокс реален | локально |
| C1.A2 | LB-переносимость: blend_retuned ± shrink c=0.95 | shrink < базы на LB → G НЕ переносится | **2 LB-слота** |
| C1.A3 | Декомпозиция shrink на G: delta(zeros)/delta(whales)/delta(middle) | delta(zeros)>>0, delta(whales)<0 → сегментная гипотеза | локально |
| C1.A4 | zero_frac/whale_frac: B vs G truth | диагностика | локально |

## Forbidden

- Не делать вывод «shrink мёртв» без C1.A2 (LB-факт обязателен).
- Не тюнить c на одном фолде.
- Не применять shrink глобально в production до M1 (сегментная версия).

## Result

(заполнить после прогона: observed + verdict per check)

## Compliance

R1/R6/R8. Не используем 2026-таргет для фич (R10-дух).
