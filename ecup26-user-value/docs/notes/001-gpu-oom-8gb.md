# 001: GPU OOM на 8GB VRAM (RTX 5060)

- **Date:** 2026-08-15
- **Source:** mlaw-rag T-057/T-015 (learning)
- **Tags:** gpu | oom | reranker | eval

## Finding

При попытке запустить embedder (bge-m3, fp32) + reranker (bge-reranker-v2-m3, fp32) на 8GB VRAM получаем OOM.

**Причина**: FFN intermediate tensor reranker'а при default `max_seq_length=8192` достигает ~4.3 GB.

## Evidence

```
torch.OutOfMemoryError: Tried to allocate 858 MiB. total 7.53 GiB, 398 MiB free.
reranker.py: predict(pairs, batch_size=32) -> OOM in feed_forward_chunk
```

## Fix (verified)

8GB VRAM budget требует:
- ✅ embedder: `fp16` (~1.1 GB) вместо fp32 (~2.2 GB)
- ✅ reranker: `fp16` + `max_length=2048` (чанки ≤512 слов ≈ 1000 токенов, truncation не нужен)
- ✅ `batch_size=16` (вместо 32)
- ✅ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` для уменьшения фрагментации
- Итоговый footprint: ~2.5 GB, влезает с запасом

## Impact на E-CUP

- **T1 (Qwen3-Embedding-4B)**: модель 8 GB FP16 — НЕ влезает! Решение: `device_map="auto"` + CPU offload (см. ADR-001)
- **T2 (Qwen3-VL-2B)**: 2B = ~4 GB FP16 — влезает полностью
- **T2 (cross-encoder)**: если используем reranker, применять fix выше
- **T3 (LightGBM/sklearn)**: CPU only, GPU не нужна

## Checklist для T1/T2

- [ ] Проверить `nvidia-smi` перед запуском: VRAM должен быть 8+ GB
- [ ] Всегда `fp16` для моделей >1B параметров
- [ ] `expandable_segments:True` в env
- [ ] `batch_size=16` для начала (не 32)
- [ ] Если OOM — fallback на device_map="auto" (часть модели на CPU)
- [ ] Логировать VRAM usage каждый stage
