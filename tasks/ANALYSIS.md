# E-CUP 2026 — Анализ проекта и стратегия

> **Дата**: 2026-08-14
> **Статус**: Данные собраны, структура репо создана, baseline не запущен

## Контекст

E-CUP 2026 — соревнование по ML от Ozon. 3 независимые задачи:

| # | Название | Тип | Метрика | Deadline pressure |
|---|---|---|---|---|
| 1 | Матчинг товаров | Binary classification | Macro PR-AUC | Высокое (20 категорий, время критично) |
| 2 | VLM Quality Control | Classification + explanation | Macro F1 | Среднее (VLM долго) |
| 3 | User GMV Prediction | Tabular regression | RMSLE | Низкое (проще всего) |

## Архитектура решения

```
contest/                          ← monorepo (contest-level docs, tasks)
├── docs/                         ← исходная документация ODS (извлечённая)
│   └── tasks/                   ← чистые markdown из HTML
├── tasks/                        ← наши аналитические документы
│   ├── TASK_1_matching.md
│   ├── TASK_2_quality.md
│   ├── TASK_3_user_value.md
│   ├── COMMISSION_FLOW.md
│   └── ANALYSIS.md
├── ecup26-matching/              ← задача 1
├── ecup26-quality/               ← задача 2
├── ecup26-user-value/            ← задача 3
└── (опционально) ecup26-mcp/    ← commission MCP-сервер
```

## Стратегия по задачам

### Задача 1 — Matching (приоритет: высокий)

**Почему первая**: PR-AUC чувствительна к качеству фичей, долгий цикл итераций.
Baseline требует 18.5 мин → нужно начинать рано.

**План**:
1. Запустить lightweight baseline (без модели) — проверить pipeline
2. Построить фичи: текстовые метрики (Jaccard, BM25, edit distance)
3. Добавить эмбеддинги (bi-encoder: `sentence-transformers/paraphrase-multilingual` или специализированный)
4. CatBoost per-category или одна модель с категорией как фича
5. Оптимизировать inference time (13 мин hard limit)

**Риски**:
- Время исполнения может превысить лимит → нужно профилировать рано
- Новые товары на test → фичи должны generalise

### Задача 2 — Quality (приоритет: средний)

**Почему средняя**: VLM дорогие, но deadline длиннее.

**План**:
1. Запустить baseline с Qwen3-VL
2. Построить классификатор: текст + эмбеддинги картинок
3. Fine-tune LoRA на объяснениях (опционально)
4. Валидация формата (comment 50-300 символов)

**Риски**:
- VLM медленные на CPU → нужен GPU для отладки
- Генерация объяснений может не влезать в лимит времени

### Задача 3 — User Value (приоритет: низкий, но самая простая)

**Почему низкий**: tabular проще всего, baseline хороший.

**План**:
1. Запустить baseline notebook
2. Добавить фичи: RFM, временные окна
3. CatBoost/LightGBM
4. RMSLE-aware оптимизация (обучение на log1p(target))

**Риски**:
- Прореженные ряды (30M записей → 100M если дозаполнить нулями)
- Простая модель может быть достаточно

## Ресурсы и ограничения

| Ресурс | Задача 1 | Задача 2 | Задача 3 |
|---|---|---|---|
| CPU | 20 ядер | 20 ядер | 20 ядер |
| RAM | 200 GB | 100 GB | — |
| GPU | H100 80GB | H100 80GB | — |
| Время (private) | 13 мин | 40 мин | — |
| Размер архива | 5 GB | 5 GB | CSV only |
| Docker | Да | Да | Нет |

## Быстрые победы (low-hanging fruit)

1. **Task 3**: Аггрегация по пользователю + CatBoost → быстрый baseline
2. **Task 1**: Text similarity фичи (Jaccard, token overlap) → улучшение без модели
3. **Task 2**: Анализ data.csv → понять категории и распределение
4. **Все задачи**: Унифицировать API (run.py с --validate и --submit)

## Что уже сделано (2026-08-14)

- [x] Данные скачаны и проверены (items.parquet, images.zip, train.parquet)
- [x] HTML → MD конвертация docs/
- [x] tasks/TASK_*.md созданы
- [x] COMMISSION_FLOW.md создан
- [ ] Git init во всех репо
- [ ] Baseline запуск во всех репо
- [ ] MCP-сервер (commission mirror)

## Что нужно сделать (Sprint 1)

### Docs (TD.0.1-5) — ✅ ЗАВЕРШЕНО
- [x] TASK_1_matching.md
- [x] TASK_2_quality.md
- [x] TASK_3_user_value.md
- [x] COMMISSION_FLOW.md
- [x] ANALYSIS.md

### Repo Init (TM.0.1, T1.0.1, T2.0.1, T3.0.1)
- [ ] `ecup26-matching/`: pyproject.toml, run.py, .gitignore, data/ folder
- [ ] `ecup26-quality/`: pyproject.toml, run.py, .gitignore, data/ folder
- [ ] `ecup26-user-value/`: pyproject.toml, run.py, .gitignore, data/ folder
- [ ] `ecup26-mcp/` (commission mirror): commission_specs/, validate_submission.py

### Sprint 1 Baseline
- [ ] T1.1.1: matching-baseline-lightweight → run.py
- [ ] T2.1.1: quality-baseline-submit → run.py
- [ ] T3.1.1: baseline-notebook → run.py

## Команды для работы

```bash
# Matching
cd ~/Repositories/contest/ecup26-matching
python run.py --validate

# Quality
cd ~/Repositories/contest/ecup26-quality
python run.py --validate

# User Value
cd ~/Repositories/contest/ecup26-user-value
python run.py --validate

# MCP Commission
cd ~/Repositories/contest/ecup26-mcp
python validate_submission.py --task matching --commit HEAD
```

## Метрики и мониторинг

| Задача | Метрика | Функция |
|---|---|---|
| Matching | PR-AUC | `sklearn.metrics.average_precision_score` (per-category) |
| Quality | F1 | `sklearn.metrics.f1_score` (per-category) |
| User Value | RMSLE | `np.sqrt(np.mean((log1p(y) - log1p(y_hat))**2))` |

## Glossary

| Термин | Значение |
|---|---|
| PR-AUC | Area Under Precision-Recall Curve |
| GMV | Gross Merchandise Value (сумма покупок) |
| RMSLE | Root Mean Squared Logarithmic Error |
| VLM | Vision-Language Model |
| LTV | Lifetime Value |
| RFM | Recency, Frequency, Monetary |
| RAG | Retrieval-Augmented Generation |
| Bi-encoder | Модель эмбеддингов: sentence → vector |
| Cross-encoder | Модель эмбеддингов: (sentence1, sentence2) → score |
| LoRA | Low-Rank Adaptation (fine-tuning) |
