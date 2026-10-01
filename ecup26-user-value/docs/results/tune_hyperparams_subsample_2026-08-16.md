# Hyperparameter tuning results — 2026-08-17

- Total configs evaluated: 81
- Top-K shown: 10
- Wall time: 2273.4s
- Bootstrap: 2.3634 <= RMSLE <= 2.3703 (best config)

| Rank | n_est | leaves | min_data | lr | RMSLE | lift | mae | rmse | spearman | fit_sec |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 7 | 10 | 0.2 | 2.3668 | 0.255 | 115.52 | 498.46 | 0.029 | 0.1 |
| 2 | 100 | 7 | 50 | 0.1 | 2.3671 | 0.254 | 115.49 | 480.54 | 0.028 | 0.1 |
| 3 | 100 | 31 | 10 | 0.05 | 2.3678 | 0.254 | 115.47 | 487.41 | 0.027 | 0.2 |
| 4 | 200 | 7 | 50 | 0.05 | 2.3679 | 0.254 | 115.48 | 480.44 | 0.027 | 0.1 |
| 5 | 100 | 15 | 50 | 0.05 | 2.3683 | 0.254 | 115.47 | 475.83 | 0.027 | 0.1 |
| 6 | 50 | 15 | 50 | 0.1 | 2.3687 | 0.254 | 115.47 | 476.30 | 0.027 | 0.1 |
| 7 | 50 | 31 | 10 | 0.1 | 2.3690 | 0.254 | 115.48 | 487.90 | 0.027 | 0.1 |
| 8 | 200 | 7 | 10 | 0.1 | 2.3708 | 0.253 | 115.52 | 498.15 | 0.026 | 0.1 |
| 9 | 100 | 15 | 100 | 0.05 | 2.3715 | 0.253 | 115.47 | 468.37 | 0.006 | 0.1 |
| 10 | 50 | 15 | 100 | 0.1 | 2.3727 | 0.253 | 115.48 | 468.76 | 0.027 | 0.0 |

## Best config

```python
LightGBMRegressor(
    n_estimators=100,
    num_leaves=7,
    min_child_samples=10,
    learning_rate=0.2,
)
```

## Compliance

- R1: open-source (lightgbm, pandas, tqdm)
- R6: reproducible (seeded)
- R8: dev-only, NOT imported by run.py
- R10: no submit.csv writes