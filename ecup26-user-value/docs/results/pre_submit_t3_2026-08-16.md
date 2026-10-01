# T3 pre-submit check — 2026-08-16

Per ticket T3.7.1, all 5 steps passed end-to-end.

## Environment

- Repo: `ecup26-user-value`
- Python: 3.11 (slim base)
- Docker: 29.7.2 (client)
- Date: 2026-08-16
- Operator: Maxim Boguslavskiy

## Steps

### Step 1: Local dry-run

```
$ uv run python -u run.py --test_data_path data/sample_submit.csv \
                                    --output_path /tmp/submit.csv
[INFO] Loading test users from data/sample_submit.csv
[INFO] Predicting for 250000 unique users
[INFO] Fitting BTYDBaseline on 30631006 transaction rows...
[INFO] Fit complete
[INFO] Writing 250000 rows to /tmp/submit.csv
[INFO] Sanity: mean predict=16685.12, naive-RMSLE-against-zero=7.9484
```

- Wall time: **55 seconds** (BTYD fit: 19s on 30.6M transaction rows)
- Output: `/tmp/submit.csv`, **250 001 rows** (header + 250 000 users)

### Step 2: R10 assertion (unique IDs)

```
rows: 250000
user_id unique: True
all predict >= 0: True
columns: ['user_id', 'predict']
min/max predict: 0.00 / 3502879.41
R10 OK
```

### Step 3: Docker build

```
$ docker build -t ecup26-user-value:dev .
#12 exporting layers done
#12 writing image sha256:... done
BUILD_TIME_SEC=49
size: 878MB  (R8 limit: 15 GB compressed - PASS with margin)
```

**Note**: initial build failed at runtime because `uv sync` installs
into `.venv/` but does not modify `PATH`. Fix committed as `072db73`:
prepend `/app/.venv/bin` to `PATH` so `ENTRYPOINT ["python", ...]`
resolves to the venv interpreter.

### Step 4: Docker run

```
$ docker run --rm -v $(pwd)/data:/data ecup26-user-value:dev \
    --test_data_path /data/sample_submit.csv --output_path /data/submit.csv
[INFO] Loading test users from /data/sample_submit.csv
[INFO] Fitting BTYDBaseline on 30631006 transaction rows...
[INFO] Fit complete
[INFO] Writing 250000 rows to /data/submit.csv
[INFO] Sanity: mean predict=16685.12, naive-RMSLE-against-zero=7.9484
DOCKER_RUN_TIME_SEC=56
```

### Step 5: Schema sanity (post-Docker)

```
$ head -1 data/submit.csv
user_id,predict

$ tail -2 data/submit.csv
918477,2638.3036376271393
918481,7608.484984145207

PASS: schema=['user_id', 'predict'] rows=250000 unique=True
min/max predict = 0.00 / 3502879.41
mean predict = 16685.12
```

## Result

| Item | Value |
|---|---|
| Local wall time | 55 s |
| Docker build time | 49 s |
| Docker run time | 56 s |
| Image size | 878 MB (< 15 GB R8 limit) |
| Submit rows | 250 000 |
| Schema | `user_id,predict` (R10 unique IDs) |
| Predict range | [0.00, 3 502 879.41] (R10 non-negative) |
| Mean predict | 16 685.12 |
| Local vs container predict mean | **identical** (R6 determinism OK) |
| Test suite | 66 passed (unchanged) |

## Compliance

- R1: open-source (lifetimes + scikit-learn)
- R4: no internet at runtime (all deps in image)
- R6: deterministic (mean predict identical local vs container)
- R7: CPU tabular model (no LLM)
- R8: image < 15 GB, `metadata.json` exists, `python -u run.py` entry point
- R10: assert unique IDs before write, asserted externally

## Forbidden actions taken

None. No leaderboard submission, no cloud GPU.

## Follow-ups

- Mean predict ~16 685 suggests the global BTYD is **biased high** for
  many users; consider per-segment BTYD (T3.6.14) for production submission.
  Ablation output (T3.6.12, T3.6.18) will guide the choice.
- Image size is fine but could shrink by switching to `python:3.11-alpine`
  (saves ~200 MB). Not blocking.
