# 002: chunk_id collisions — silent data loss

- **Date:** 2026-08-15
- **Source:** mlaw-rag T-057/note 014
- **Tags:** chunking | indexing | data-loss | ids

## Finding

Sliding-window chunks из разных contents nodes одного акта **коллидили на `chunk_id`** (`doc_id@seq_no`). Окно перезапускало `seq_no` на 0, а chunk_id вычислялся один раз в `Chunk.__init__`. Qdrant's deterministic `uuid5("mlaw:{chunk_id}")` upsert перезаписывал более ранний point.

## Evidence

```
Первый прогон make build-index (PIT 2026-07-01, mlaw_0000):
  chunks=14821 uploaded=14821 (reported)
  Qdrant points_count = 12594 -> 2,227 chunks (15.0%) silently lost
```

Воспроизведение (act с двумя 600-word contents nodes, max_tokens=512):
```
8 chunks created, but only 4 unique chunk_ids
duplicates: ['MLAW_TEST@0', 'MLAW_TEST@1', ...]
```

## Impact на E-CUP

**Критично для submission**:
- ❌ Если submission пишет predictions с коллизиями ID — будет silent data loss
- ❌ Если один (item_id, pair_id) пишется дважды — второй overwrite первый
- ✅ FAISS index не страдает (там нет point_id), но CSV submission — страдает

## Prevention (правило для всех треков)

### Rule 1: уникальные ID всегда
```python
import uuid
unique_id = f"{prefix}-{uuid.uuid4().hex}"  # никогда не переиспользовать
```

### Rule 2: ID = content-hash
```python
import hashlib
content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
id = f"{doc_id}-{content_hash}"  # уникален по содержимому
```

### Rule 3: ID с monotonic prefix
```python
import time
id = f"{int(time.time() * 1e6)}-{item_id}"  # микросекунды + original id
```

### Rule 4: assert перед write
```python
assert len(set(row_ids)) == len(row_ids), f"duplicate IDs: {duplicates}"
```

## Verification

Перед каждым submission:
- [ ] `python -c "import pandas as pd; df = pd.read_csv('submit.csv'); assert df['id'].is_unique"`
- [ ] Подсчёт expected vs actual predictions
- [ ] Логировать `total_ids` и `unique_ids` — должны совпадать
