# T3 validation report

- Predictions: `/tmp/submit_v2.csv`
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
| `spearman_global` | n/a | ✓ |
| `percentiles_q10_pred` | 0.0000 | ✓ |
| `percentiles_q50_pred` | 3082.8509 | ✓ |
| `percentiles_q90_pred` | 42275.1905 | ✓ |
| `percentiles_q10_true` | 0.0000 | ✓ |
| `percentiles_q50_true` | 3082.8509 | ✓ |
| `percentiles_q90_true` | 42275.1905 | ✓ |
| `coverage_q10_q90` | 0.9000 | ✓ |
| `pit_total_observations` | 250000.0000 | ✓ |
| `pit_max_bin_count` | 49999.0000 | ✓ |
| `pit_min_bin_count` | 0.0000 | ✓ |
| `bootstrap_ci_low` | 0.0000 | ✓ |
| `bootstrap_ci_high` | 0.0000 | ✓ |
| `adversarial_n_slices` | 1.0000 | ✓ |
| `adversarial_max_rmsle` | 0.0000 | ✓ |

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

## Tier

Tier 1 + Tier 2/3 (enable Tier 2/3 with --tier=2; default = Tier 1).

## Forbidden actions taken

None. Read-only on `submit.csv`; writes only this report.
