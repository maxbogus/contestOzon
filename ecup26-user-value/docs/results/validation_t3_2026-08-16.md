# T3 validation report

- Predictions: `/tmp/submit.csv`
- Target: `placeholder=data/sample_submit.csv['predict']`
- Rows: 250000

| Metric | Value | Pass |
|---|---|---|
| `model_rmsle` | 0.0000 | ✓ |
| `naive_rmsle_mean` | 4.1801 | ✓ |
| `naive_rmsle_median` | 3.4417 | ✓ |
| `lift_vs_mean` | 1.0000 | ✓ |
| `mae` | 0.0000 | ✗ |
| `rmse` | 0.0000 | ✗ |
| `bias_mean` | 0.0000 | ✗ |
| `spearman` | 1.0000 | ✓ |
| `quantile_calibration` | 0.0000 | ✗ |

## Naive baseline RMSLE (mean): 4.1801

## Pass criteria

- `lift_vs_mean > 0`: model must beat the mean constant predictor
- `mae < mean(true)`: must be better than mean
- `rmse < 2 * mae`: scale check
- `bias_mean abs < mae`: model not systematically biased
- `spearman > 0`: must have positive correlation
- `quantile_calibration < mae`: distribution close to ground truth

## Compliance

- R1: open-source (numpy, pandas, scipy)
- R6: reproducible
- R8: dev-only, does not write submit.csv
- R10: predictions file asserted unique IDs

## Forbidden actions taken

None. Read-only on `submit.csv`; writes only this report.
