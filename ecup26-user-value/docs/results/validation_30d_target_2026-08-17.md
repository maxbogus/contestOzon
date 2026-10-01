# T3 validation — 30-day-target fix (2026-08-17)

## Critical bug found & fixed

The production pipeline (`run_pipeline.predict_for_users`) trained LightGBM on
**lifetime cumulative GMV** (`train_df.groupby("user_id")["price"].sum()` over
the whole 13-month train), while the E-CUP metric scores GMV over the **next 30
days**. Consequences (measured on `data/train.parquet`):

| Signal | Value |
|---|---|
| `mean predict` (buggy model) | 1088.41 = exactly the lifetime GMV mean |
| true 30-day GMV mean (last 30d window) | 84.03 |
| true 30-day median / p90 | 7.89 / 215.86 |
| users with zero 30-day GMV | 114 835 (46%) |
| zeros in buggy predictions | 0 |

The buggy submission scored **proxy RMSLE 3.96 — worse than the naive
constant-mean baseline (3.18)**.

## Fix (committed in `run_pipeline.py`)

1. **30-day target, no leakage**: label = sum(price) over the most recent
   `TARGET_DAYS` (30) window; features built strictly before that window
   (`feature_cutoff = ref_date - 30d`).
2. **log1p label**: `y = log1p(30d GMV)` with L2 loss → directly minimizes the
   RMSLE numerator; predictions are `expm1`-ed back to GMV and clipped ≥ 0.
3. **Two feature matrices**: `train` features at feature_cutoff (aligned with
   the label) and `infer` features at ref_date (freshest signal for scoring).
4. Cache schema bumped to v2 (`features_train_*`, `features_infer_*`, `model_*`).

## Results

### Honest out-of-sample proxy (`scripts/validate_pipeline_30d.py`)

Trains exactly like production on data up to `proxy_now` and scores the next
30 days (the window the model did NOT train on):

```
model RMSLE (30d-target pipeline)  = 1.7309
naive RMSLE (constant mean)        = 3.1750
lift vs naive                      = +45.48%
pred : mean=40.53 median=9.12
truth: mean=84.03 median=7.89
rmsle < naive (PASS): True
```

### Production submission (`run.py` on `data/sample_submit.csv`)

```
[INFO] Sanity: mean predict=32.77, naive-RMSLE-against-zero=2.7279
proxy RMSLE vs last-30d truth = 1.3069   (naive 3.1750)
pred mean / median / p90     = 32.77 / 6.90 / 83.47
wall time (cold, cache miss) = 29s   |  warm (cache hit) = 12s
```

## Before vs after

| Model | Proxy RMSLE | vs naive (3.18) |
|---|---|---|
| BTYD (old baseline) | 5.56 | -75% |
| Buggy production (lifetime target) | 3.96 | -25% |
| **Fixed production (30d + log1p)** | **1.31** | **+59%** |

The subsample "RMSLE 2.37" previously reported in `tune_hyperparams_*` is an
artifact of the lifetime-target bug + subsample distribution and is superseded.

## Compliance

- R1: open-source (lightgbm, pandas, numpy)
- R6: reproducible (seeded, deterministic pipeline)
- R8: dev-only validation; `run.py` entry point unchanged
- R10: submit.csv asserted unique user_id
- R4: no internet at runtime (cache is local disk only)

## Commands

```bash
# honest out-of-sample proxy (expect ~1.7 RMSLE)
uv run python scripts/validate_pipeline_30d.py

# production submission (expect ~1.3 proxy RMSLE, 12s warm)
uv run python -u run.py --test_data_path data/sample_submit.csv --output_path /tmp/submit.csv
```
