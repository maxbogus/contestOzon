# E-CUP 2026 — Ozon Tech Competition

Three ML tasks for Ozon marketplace: product matching, quality control (VLM), and GMV prediction.

## Tasks

| # | Name | Type | Metric | Deadline |
|---|---|---|---|---|
| 1 | Product Matching | Binary classification | Macro PR-AUC | Container (20 min) |
| 2 | Quality Control | Classification + explanation | Macro F1 | Container (40 min) |
| 3 | User Value | Tabular regression | RMSLE | CSV only |

## Structure

```
contest/
├── docs/                      # ODS documentation (extracted from HTML)
│   └── tasks/                # Clean markdown from ODS pages
├── tasks/                     # Our analysis & task specs
│   ├── TASK_1_matching.md
│   ├── TASK_2_quality.md
│   ├── TASK_3_user_value.md
│   ├── COMMISSION_FLOW.md
│   └── ANALYSIS.md
├── ecup26-matching/           # Task 1 repo
├── ecup26-quality/            # Task 2 repo
└── ecup26-user-value/        # Task 3 repo
```

## Quick Start

### Task 1 — Matching

```bash
cd ecup26-matching
python run.py --validate
```

### Task 2 — Quality

```bash
cd ecup26-quality
# Extract images.zip first (9.2 GB)
unzip data/images.zip -d data/
python run.py --validate
```

### Task 3 — User Value

```bash
cd ecup26-user-value
python run.py --validate
```

## Data

All data files are in the respective `data/` directories.

| Task | Key Files |
|---|---|
| 1 Matching | `items.parquet` (4.1 GB), `matches.parquet`, `matches_llm.parquet` |
| 2 Quality | `data.csv` (37 MB), `images.zip` (9.2 GB) |
| 3 User Value | `train.parquet` (180 MB), `sample_submit.csv` |

## Resources

- ODS Competition: https://ods.ai/competitions/e-cup-2026
- Rules: `docs/Правила E-CUP_23.07.26.pdf`
