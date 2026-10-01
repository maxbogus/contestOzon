# Задача 1: Матчинг товаров

> **Источник**: [`docs/tasks/task1_matching.md`](../docs/tasks/task1_matching.md),
> [`docs/tasks/task1_matching_data.md`](../docs/tasks/task1_matching_data.md)
> **Дата**: 2026-08-14

## Суть задачи

Классификация пар товаров: один ли это и тот же товар (дубли карточек) или разные.
Binary classification: пара одинаковая или нет.

**pipeline**: Retrieval → пары кандидатов подаются на вход → наш классификатор.

## Данные

### Файлы (в `ecup26-matching/`)

| Файл | Размер | Описание |
|---|---|---|
| `items.parquet` | 4.1 GB | Полный каталог ~13M товаров, колонки: `id, name, attributes, category` |
| `items_human.parquet` | 205 MB | Подмножество товаров из ручных пар (содержится в `items.parquet`) |
| `matches.parquet` | 4 MB | Ручная разметка ~365K пар, `target` ∈ {0, 1} |
| `matches_llm.parquet` | 100 MB | LLM-разметка ~11M пар, `target` ∈ [0, 1] |
| `matching-baseline-submit.zip` | 1.2 GB | Baseline (cross-encoder MiniLM-L12-v2, 18.5 мин / лимит 20 мин) |
| `matching-baseline-lightweight.zip` | 20 KB | Baseline без модели (для отладки) |

### Глоссарий

```
matches.parquet / matches_llm.parquet:
  - id1: int — первый товар в паре
  - id2: int — второй товар в паре
  - target: float — 0/1 (ручная) или 0..1 (LLM)

items.parquet / items_human.parquet:
  - id: int
  - name: str — название товара
  - attributes: str — свёрнутый JSON атрибутов
  - category: str — категория
```

### Важные детали

- 20 категорий товаров
- На тесте — новые товары (не из train). Нужны фичи, работающие на unseen items
- Тестовый объём ≈ `matches.parquet + items_human.parquet`
- Можно доразмечать данные open-source LLM (проприетарные запрещены)

## Метрика

**Macro Averaged PR-AUC** (площадь под PR-кривой, усреднённая по 20 категориям)

```python
from sklearn.metrics import average_precision_score
```

> ⚠️ Использовать `auc()` для PR-кривой — **нельзя** (завышает score).
> Только `average_precision_score`.

## Ограничения

| Параметр | Значение |
|---|---|
| Время Check | 1 мин |
| Время Public | 6 мин |
| Время Private | 13 мин |
| RAM | 200 GB |
| GPU | Tesla H100 80 GB |
| CPU | 20 ядер |
| Архив решения | 5 GB |
| Docker-образ | 15 GB |

## Интерфейс решения

```bash
python run.py --items_path <path> --matches_path <path> --output-path <path>
```

### Формат submit.csv

```csv
id1,id2,predict
1001,2002,0.87
```

- `id1, id2` — идентификаторы пар
- `predict` — сырая вероятность совпадения (num)

## Baseline-подход

1. Для каждой пары: `name1 + " " + attributes1` vs `name2 + " " + attributes2`
2. bi-encoder или cross-encoder на эмбеддингах
3. CatBoost на фичах из эмбеддингов + текстовых метрик

## Разделение train/test

- **Public** ≈ 115K пар (~30%)
- **Private** ≈ 275K пар (~70%)

## TODO в нашем репо

- [ ] `T1.1.1` — Baseline run (matching-baseline-lightweight)
- [ ] `T1.2.1` — Фичи: текстовые метрики (Jaccard, BM25, edit distance)
- [ ] `T1.2.2` — Фичи: эмбеддинги (bi-encoder)
- [ ] `T1.2.3` — Фичи: категорийные
- [ ] `T1.3.1` — CatBoost per-category classifier
- [ ] `T1.4.1` — Оценка PR-AUC локально
- [ ] `T1.5.1` — Submission pipeline
- [ ] `T1.6.1` — Docker-контейнер
