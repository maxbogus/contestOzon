# G2 — Re-test regime-flip фич на fold G (дешёвый батч)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Ledger: G2.A1–G2.A5. 5 фич, отвергнутых «sign-flip на P», но таргет = gift-режим.

## Hypothesis

Все 5 фич отвергнуты до появления fold G (20-го). Fold G имитирует gift-таргет
(zero_frac 0.484). Пере-тест на G = правильная проверка для режимно-зависимых фич.

## Atomic checks

| ID | Фича (переоткрытие) | Гейт (все на fold G, M=500) |
|---|---|---|
| G2.A1 | holiday pre/post флаги (T3.10.15) | WIN P≤0.1 |
| G2.A2 | targeted-interactions (T3.10.25) | WIN P≤0.1 |
| G2.A3 | funnel-v2-chains (T3.10.26) | WIN P≤0.1 |
| G2.A4 | markov-gift-boost (T3.10.18) | WIN P≤0.1 |
| G2.A5 | hybrid-OOF P(y>0) на actor-фичах | WIN P≤0.1 |

Победители → LB-слоты (1–2/день).

## Root cause link

P0-звено 3 (нет G-гейта) + 5 (reject-by-default по P-флипу).

## Forbidden

- Не отвергать по старому вердикту без G-прогона.
- Не комбинировать победителей до их изолированных WIN.

## Result

(заполнить после прогона)

## Compliance

R1/R6/R8.
