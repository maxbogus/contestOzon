# E-CUP 2026 — RUNBOOK

Master runbook for all 3 tracks (T1 matching, T2 quality, T3 user value).
Based on lessons learned from mlaw-rag (T-057 GPU run, T-015 OOM, T-014 chunk collisions).

## 0. Предусловия (однократно)

```bash
# Hardware check (RTX 5060 8GB expected)
nvidia-smi                   # VRAM >= 7.5 GB free
df -h                        # Disk >= 50 GB free

# Software
uv sync --all-extras         # ВАЖНО: plain uv sync НЕ ставит pytest/ruff
cp .env.example .env         # Настроить API keys, paths

# GPU env (для предотвращения OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1. Инфраструктура

| Трек | Docker image | Entrypoint | Time limit |
|------|--------------|------------|------------|
| T1 | `ecup26-matching:1.0` | `python -u run.py` | 13 мин private / 6 мин public |
| T2 | `ecup26-quality:1.0` | `python -u run.py` | 40 мин private / 20 мин public |
| T3 | `ecup26-user-value:1.0` | `python -u run.py` | ? |

## 2. Общие gotchas (все треки)

### 2.1 chunk_id collisions → 15% silent data loss (mlaw-rag T-014)

**Симптом**: predictions меньше ожидаемого, без явной ошибки.

**Диагностика**:
```python
import pandas as pd
df = pd.read_csv("submit.csv")
assert df["id"].is_unique(), f"Duplicate IDs: {df[df.duplicated()].head()}"
assert len(df) == EXPECTED_ROWS, f"Got {len(df)} rows, expected {EXPECTED_ROWS}"
```

**Fix**: использовать `uuid.uuid4().hex` или `sha256(content).hexdigest()[:16]` для ID.

См. `docs/notes/002-chunk-id-collision.md`.

### 2.2 GPU OOM на 8GB VRAM (mlaw-rag T-015)

**Симптом**: `torch.OutOfMemoryError: Tried to allocate ... total 7.5 GiB`.

**Fix** для моделей >1B параметров:
```python
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,          # НЕ float32
    device_map="auto",                   # автоматический offload
    max_memory={0: "6GiB", "cpu": "16GiB"},
)
```

**Best practices**:
- batch_size=16 (не 32)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- max_length=2048 для cross-encoder reranker

См. `docs/notes/001-gpu-oom-8gb.md`.

### 2.3 Docker submission

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY src/ ./src/
COPY models/ ./models/   # Локально, не качать!
COPY run.py ./
ENTRYPOINT ["python", "-u", "run.py"]
```

Build:
```bash
docker build -t ecup26-matching:1.0 .
docker run --gpus all --network none -v /data:/data ecup26-matching:1.0     --input /data/test.parquet --output /data/submit.csv
```

### 2.4 no internet в Docker

❌ Не работает:
- `pip install` внутри Docker
- `huggingface_hub.snapshot_download`
- `requests.get(...)`

✅ Работает:
- Локальные файлы в `/data/`
- Models в `./models/` (внутри Docker)
- Output в `/data/submit.csv`

### 2.5 Time limits

| Тест | Время | Fail action |
|------|-------|-------------|
| Check | 1 мин | fail -> no public score |
| Public | 6-20 мин | fail -> no public leaderboard |
| Private | 13-40 мин | fail -> disqualified |

**Стратегия**: pre-compute embeddings на полном корпусе, в Docker load только нужные test items.

## 3. Per-track runbooks

### 3.1 T1 matching (smoke test)

```bash
cd ecup26-matching
uv run pytest tests/ -v                    # Unit tests (50+ tests)
uv run python -m ecup26_matching.run     --input data/items_test_sample.parquet     --output /tmp/submit_sample.csv        # Smoke run
```

**Expected**: 
- BiEncoder encoded 500 items в ~30 сек
- PR-AUC на eval set ≥ baseline (0.45+)
- submit.csv с N строками, все id уникальны

### 3.2 T2 quality (smoke test)

```bash
cd ecup26-quality
uv run pytest tests/ -v
uv run python -m ecup26_quality.run     --input data/products_test_sample.json     --output /tmp/submit_sample.csv
```

**Expected**:
- Qwen3-VL-2B processed 100 products в ~2 мин
- CSV с комментарием 50-300 chars + вердикт "бан"/"не бан"

### 3.3 T3 user value (smoke test)

```bash
cd ecup26-user-value
uv run pytest tests/ -v
uv run python -m ecup26_user_value.run     --input data/users_test.parquet     --output /tmp/submit.csv
```

**Expected**:
- LightGBM/LTV predict для 250K users в <30 сек
- CSV: user_id,predict (непрерывное значение)

## 4. Eval pipeline (общий)

```bash
# На eval set (matches.parquet)
python scripts/run_eval.py --model submission_model

# Проверка метрик
python scripts/eval_results_writer.py     --predictions submit.csv     --ground-truth matches.parquet     --output reports/metrics.json
```

**Обязательно проверять**:
- [ ] PR-AUC (T1) / F1 (T2) / MAE (T3)
- [ ] Control test: random baseline должен collapse метрики
- [ ] Paired t-test при сравнении variants
- [ ] Latency P95 в time_limit * 0.7

## 5. Submission checklist

- [ ] `make test` — все тесты проходят
- [ ] `make lint` — ruff чисто
- [ ] `docker build` — successful
- [ ] Smoke test на 100 записях — успешен
- [ ] Eval на validation set — метрики в baseline range
- [ ] `submit.csv` проверен: `is_unique()`, `EXPECTED_ROWS == len(df)`
- [ ] `metadata.json` заполнен: `image`, `entry_point`
- [ ] Время выполнения на full test < time_limit

## 6. Если что-то пошло не так

| Проблема | Вероятная причина | Fix |
|----------|-------------------|-----|
| OOM в Docker | batch_size слишком большой | Уменьшить batch_size |
| submit.csv короче ожидаемого | ID collision или crash | Проверить `is_unique()` |
| Метрики сильно ниже baseline | Модель/transformer не загружен | Проверить `models/` в Docker |
| Time limit exceeded | Encoding полного корпуса | Pre-compute embeddings |
| Docker build fails | Не хватает RAM | Увеличить Docker memory |

## 7. References

- `docs/notes/001-gpu-oom-8gb.md` — урок про OOM
- `docs/notes/002-chunk-id-collision.md` — урок про ID
- `docs/backlog/decisions/ADR-001-bi-encoder-inference-strategy.md` — T1 inference strategy
- `docs/backlog/decisions/ADR-002-vlm-4b-rule.md` — T2 ≤4B rule
- `docs/backlog/decisions/ADR-003-mlaw-rag-reuse.md` — reuse strategy
