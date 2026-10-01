# T3 Submission Preparation Guide — 2026-08-17

> Status after commit `af0501e` / `81f7c7d` (master). Everything below was
> executed and verified on 2026-08-17.

## 1. What changed (summary)

| # | Change | Impact |
|---|---|---|
| 1 | **30-day GMV target** (was: lifetime sum) + **log1p label** / expm1 output | Proxy RMSLE **3.96 → 1.31** (naive = 3.18) |
| 2 | Vectorized `build_features` (groupby, no per-user loop) | Feature build: hours → **~7 s** on 30.6M rows |
| 3 | Resume cache (default-on, content-addressed) | Warm re-runs: **~12 s** |
| 4 | Explicit LightGBM `num_threads` | Deterministic CPU usage (12 cores) |
| 5 | `lightgbm`/`tqdm` → runtime deps + `libgomp1` in Docker | **Submission image runs** (was: ImportError / libgomp OSError) |
| 6 | `uv.lock` now tracked | `uv sync --frozen` works on clean checkout (R6) |

Details + numbers: `docs/results/validation_30d_target_2026-08-17.md`.

## 2. Pre-flight checks (run before every submission)

```bash
cd /home/maxbogus/Repositories/contest/ecup26-user-value

# 1) Tests
uv run pytest -q                          # expect: 144 passed

# 2) Lint
uv run ruff check run.py src scripts tests # expect: All checks passed

# 3) Forbidden patterns (must print CLEAN; only docstring mentions are OK)
grep -rni 'ollama\|vllm\|tgi\|lm.studio\|exllamav2' src/ run.py || echo CLEAN
grep -rni 'openai.com\|anthropic.com\|gpt-4\|claude-\|gemini' src/ run.py || echo CLEAN
grep -rni 'requests.get\|httpx.post\|urllib.request' src/ run.py || echo CLEAN
```

## 3. Generate the submission locally

```bash
uv run python -u run.py \
  --test_data_path data/sample_submit.csv \
  --output_path /tmp/submit_final.csv
```

- Expected wall time: **~30 s cold** (first run) / **~12 s warm** (cache hit).
- Expected log tail:
  `Sanity: mean predict=32.77, naive-RMSLE-against-zero=2.7279`
- Cache: enabled by default at `data/.cache` (`--no-cache` to disable,
  `--cache-dir` / `ECUP26_CACHE_DIR` to relocate).

## 4. Validate the file (R10 + schema)

```bash
uv run python - <<'EOF'
import pandas as pd
df = pd.read_csv('/tmp/submit_final.csv')
assert list(df.columns) == ['user_id', 'predict'], df.columns
assert len(df) == 250_000, len(df)
assert df['user_id'].is_unique, 'R10: duplicate user_id'
assert (df['predict'] >= 0).all(), 'R10: negative predictions'
print('schema OK:', len(df), 'rows, unique user_id, predict >= 0')
EOF
```

Honest out-of-sample quality gate (the number the leaderboard should be near):

```bash
uv run python scripts/validate_pipeline_30d.py
# model RMSLE ~1.73, naive 3.18, lift ~+45% -> PASS (must be < naive)
```

Production-submission proxy (in-distribution reference): RMSLE **1.31**.

## 5. Docker build & run (R8 contract)

```bash
docker build -t ecup26-user-value:dev .
# expect: image ~889 MB (limit 15 GB) — PASS with big margin

docker run --rm -v $(pwd)/data:/data ecup26-user-value:dev \
  --test_data_path /data/sample_submit.csv --output_path /data/submit_docker.csv
# expect: exit 0, ~40 s, same Sanity line as local
```

**Determinism (R6)**: local vs container predictions must be identical:

```bash
python3 -c "
import pandas as pd
a = pd.read_csv('/tmp/submit_final.csv'); b = pd.read_csv('data/submit_docker.csv')
print('max abs diff:', abs(a['predict'] - b['predict']).max())   # expect 0.0
"
```

Push to DockerHub with the name used in `metadata.json`.

## 6. metadata.json (must match the pushed image)

Current: `{"image": "REPLACE_WITH_YOUR_DOCKERHUB_USER/ecup26-user-value:latest", "entry_point": "python -u run.py"}`

- Set `image` to your actual DockerHub repo:tag (entry_point unchanged).
- Submit this file together with the solution archive.

## 7. Submission package checklist

- [ ] `metadata.json` — real image name, `entry_point: python -u run.py`
- [ ] `run.py` — entry point verified in Docker (Section 5)
- [ ] `submit_final.csv` — 250 000 rows, `user_id,predict`, unique, non-negative
- [ ] Docker image ≤ 15 GB compressed (actual: 889 MB)
- [ ] Archive ≤ 5 GB
- [ ] **No internet at runtime** (R4): nothing downloads inside the container
- [ ] **No pip install inside Docker** (R4)
- [ ] All libs open-source (R1): lightgbm Apache-2.0, pandas BSD, pyarrow Apache-2.0, etc.
- [ ] Model CPU-tabular, no LLM/VLM (R7), no external inference servers (R7/ADR-004)
- [ ] `uv.lock` tracked in the repo (R6), seeds fixed (`SEED = 42`)

## 8. Platform rules to respect

- Max **5 submissions/day**, final selection: **2 solutions** (official rules).
- Public/private = 20/80 (50k / 200k users). Private leaderboard decides the
  top-15 that advance to the repository-review stage.
- Metric: RMSLE with log1p; negative predictions are forced to 0.
- Eval runs on organizer hardware with a **fresh cache dir** → expect the
  cold-path time (~30–40 s), which is well within any plausible limit.

## 9. Known caveats / follow-ups

- Proxy RMSLE **1.31–1.73** is an estimate; the real leaderboard may differ
  (distribution drift). Verify with the public leaderboard.
- The model never outputs exactly 0; churned users get small positive values
  (~0.05) — negligible for RMSLE.
- Obvious next levers (if more quality is needed): walk-forward (multiple time
  origins), `objective="tweedie"`, post-hoc mean calibration, hyperparameter
  tuning on the full dataset, stronger recent-activity features.
- Old v1 cache files may linger in `data/.cache` (gitignored) — harmless.

## Update 2026-08-17 (hybrid, T3.8.10)

- **Hybrid integrated into production**: OOF leave-origin-out purchase_prob as
  a feature (114 base + purchase_prob = 115). CV 1.7123 vs 1.7155 (-0.0032);
  fold B 1.6821 (-0.0060) -> LB est ~1.651 (vs current LB 1.6588).
- New submission: `submit_final_hybrid.csv` (250k rows, unique ids, valid).
- Production run time: ~8 min incl. hybrid classifiers (T3 limit 40 min).
- Deterministic (seeded classifier, R6); cache key includes `hybrid` flag.
- Risk note: stacking features can partially lose gain on LB; fold-B proxy
  gain (-0.0060) is larger than fold-A (-0.0008). Expect LB in ~[1.649, 1.655].
