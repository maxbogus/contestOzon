# E-CUP 2026 — Task 3: User Value Prediction (GMV Forecasting)

Predict 30-day GMV for 250K users based on historical activity.
Metric: **RMSLE** (Root Mean Squared Logarithmic Error).

This is a **tabular-only** competition (no Docker, just CSV submission).

## Data

Files in `data/` (symlinked from root or copied):

| File | Description |
|---|---|
| `train.parquet` (180 MB) | ~30M rows, 250K users, 18 columns |
| `sample_submit.csv` (4.6 MB) | Template: `user_id,predict` |

## Run

```bash
# Validation (local)
python run.py --validate

# Submit to ODS
python run.py --submit
```

## Baseline

From `baseline-seacrh-ltv.ipynb`:
1. Feature engineering with Polars (window aggregates over 30/60/90 day windows)
2. Time-based CV (4 folds, 14-day stride)
3. CatBoostRegressor

## Key Features (from baseline)

```python
# Window definitions
windows = [
    ("7d", -7, 0),
    ("30d", -30, 0),
    ("60d", -60, 0),
    ("90d", -90, 0),
]

# Aggregations per window
aggs = ["sum", "max", "std", "mean"]

# Value columns
value_cols = ["gmv", "searches"]
```

## API

```bash
python run.py --output-path submit.csv
```

Output format:

```csv
user_id,predict
1001,487.97
1002,31.04
```

## Files

```
.
├── run.py                       # Entry point
├── src/
│   ├── features.py              # Feature engineering (Polars)
│   ├── model.py                # CatBoost training + inference
│   └── metrics.py              # RMSLE metric
├── baseline-seacrh-ltv.ipynb  # Original baseline notebook
├── data/                       # Data files
├── tests/                       # Unit tests
└── pyproject.toml              # Dependencies
```

## Metrics

```python
import numpy as np

def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
```

Negative predictions are clipped to 0.
