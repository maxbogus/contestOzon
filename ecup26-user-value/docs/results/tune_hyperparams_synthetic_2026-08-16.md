# Hyperparameter tuning results — 2026-08-17

- Total configs evaluated: 81
- Top-K shown: 5
- Wall time: 15.2s
- Bootstrap: 0.5650 <= RMSLE <= 0.7730 (best config)

| Rank | n_est | leaves | min_data | lr | RMSLE | lift | mae | rmse | spearman | fit_sec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 7 | 50 | 0.05 | 0.6885 | 0.029 | 72.36 | 92.55 | 0.000 | 0.0 |
| 2 | 50 | 7 | 50 | 0.1 | 0.6885 | 0.029 | 72.36 | 92.55 | 0.000 | 0.0 |
| 3 | 50 | 7 | 50 | 0.2 | 0.6885 | 0.029 | 72.36 | 92.55 | 0.000 | 0.0 |
| 4 | 50 | 7 | 100 | 0.05 | 0.6885 | 0.029 | 72.36 | 92.55 | 0.000 | 0.0 |
| 5 | 50 | 7 | 100 | 0.1 | 0.6885 | 0.029 | 72.36 | 92.55 | 0.000 | 0.0 |

## Best config

```python
LightGBMRegressor(
    n_estimators=50,
    num_leaves=7,
    min_child_samples=50,
    learning_rate=0.05,
)
```

## Compliance

- R1: open-source (lightgbm, pandas, tqdm)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py
- R10: no submit.csv writes