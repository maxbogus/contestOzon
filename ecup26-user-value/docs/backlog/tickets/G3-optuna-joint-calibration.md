# G3 — Optuna совместная калибровка (T3.10.11, TBD)
> **Статус: ⏸️ closed** — гипотеза закрыта леджером/финалом, см. `docs/research/CATALOG.md` §4.

> Ledger: G3.A1–G3.A2.

## Hypothesis

Bucket-calibration упала на LB (1.6672), потому что подбиралась на ОДНОМ фолде.
Совместный Optuna-подбор (zero-порог + bucket-bias + веса бленда) на НЕСКОЛЬКИХ
фолдах (A+B) даст sign-consistent WIN.

## Root cause link

P0-звено 1 (подбор на одном фолде) — прямой ответ.

## Atomic checks

| ID | Проверка | Гейт | Стоимость |
|---|---|---|---|
| G3.A1 | Optuna (thr+bias+weights) на fold A+B, цель mean RMSLE, trials 200 | sign-consistent на A И B | локально |
| G3.A2 | Победитель → бутстреп B+A+P+G → production → LB | WIN на 4 фолдах → LB | 1 LB-слот |

## Forbidden

- Не подбирать калибровку на одном фолде (урок T3.9.12).
- Не использовать optuna на тестовых данных.

## Result

(заполнить после прогона)

## Compliance

R1 (optuna open-source), R6, R8.
