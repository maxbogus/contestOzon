# G6 — Markov E2 sweep (T3.10.18, Open)
> **Статус: ⏳ open** — TODO (леджер/не завершено), см. `docs/research/CATALOG.md` §3-4.

> Ledger: G6.A1.

## Hypothesis

Markov-фичи (chains 13–21) дали Dormant→Active 1.72x, но E1 не доведён
(sigma/peak sweep открыт). С fold G (gift) sweep может дать WIN на 4 фолдах.

## Root cause link

Бэклог-недоделка (T3.10.18 E2 Open) + retry-кандидат markov-sweep.

## Atomic checks

| ID | Проверка | Гейт | Стоимость |
|---|---|---|---|
| G6.A1 | sigma/peak sweep (из T3.10.18) на B+A+P+G | WIN P≤0.1 на 4 фолдах → LB-слот | локально + LB |

## Forbidden

- Не оставлять тикет Open после прогона (закрыть: WIN или LB-confirmed reject).

## Result

(заполнить после прогона)

## Compliance

R1/R6/R8.
