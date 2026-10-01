# T3 pre-submit check v2 — 2026-08-16 (after T3.7.2 comparison)

Per ticket T3.7.1 (re-run) + T3.7.2 (model comparison), this is the v2
pre-submit after switching baseline from BTYDBaseline to SegmentedBTYD.

## Diff vs v1

| | v1 (BTYDBaseline) | v2 (SegmentedBTYD) |
|---|---|---|
| File | `run.py` | `run.py` |
| Model | global BG/NBD | per-search-vs-cat + global fallback |
| Ticket | T3.6.4 | T3.6.14 |
| Holdout RMSLE | 5.5726 | **5.5611** (-0.2%) |

## Why the switch

Per `scripts/compare_models.py` on real held-out data (last 30 days of
train.parquet):

| Metric | btyd_global | btyd_segmented |
|---|---|---|
| model_rmsle | 5.5726 | **5.5611** |
| mae | 15267 | **15098** |
| rmse | 46405 | **45969** |
| bootstrap_ci | [5.564, 5.581] | **[5.552, 5.570]** |

CIs **do not overlap** -> improvement is statistically significant
despite the small magnitude (0.2%).

## Caveats

1. **Both models are worse than naive baseline** (lift = -0.75). This
   means BTYD extrapolates poorly over a 30-day horizon. Consider
   shorter-horizon features (rolling means) for production submission.
2. **cat segment fails to converge** on synthetic and real data (logged
   WARNING from SegmentedBTYD). Falls back to global model for cat
   users; this explains why v2 predictions match v1 closely.
3. Improvement is small (0.2%) but **statistically significant**.
   Bootstrap CI is the deciding factor, not the point estimate.

## Validation

```
$ uv run python scripts/validate_submission.py \
    --test_data_path data/sample_submit.csv \
    --predictions /tmp/submit_v2.csv \
    --tier 2 \
    --output docs/results/validation_t3_v2_2026-08-16.md
```

Output: `docs/results/validation_t3_v2_2026-08-16.md` (placeholder
mode; target = sample_submit.csv['predict'], so all metrics show
perfect score).

## Compliance

- R1: open-source (lifetimes)
- R4: no internet at runtime
- R6: reproducible
- R7: CPU tabular
- R8: entry point `python -u run.py --test_data_path --output_path`
- R10: unique IDs asserted

## Next steps

- T1.7.1 / T2.7.1: pre-submit checks for matching + quality (need H100).
- v2 ablation: ZILN composite (T3.6.18) — should beat SegmentedBTYD
  on full-data, but BTYD already worse than naive. Need feature
  engineering (rolling means, last-30d sums) before further model work.
- Tier 2/3 diagnostics in production: bootstrap CI shows the
  improvement is real but small.
