# ADR-001: BiEncoder inference strategy (T1.0.9)

> **Date**: 2026-08-14
> **Status**: Accepted
> **Context**: T1.0.9 — Qwen3-Embedding-4B for product matching

## Hardware

- ASUS V16 V3607VM
- **CPU**: Intel Core 5 210H, 12 cores
- **RAM**: 32 GiB
- **GPU**: NVIDIA RTX 5060 Laptop (8 GB VRAM, Blackwell, ~21 TFLOPS FP16)
- **Disk**: 1 TB NVMe

## Model size

- Qwen3-Embedding-4B FP16: ~8 GB weights + ~1-2 GB activations = 10-11 GB total
- 4-bit quantized: ~2.5 GB weights + ~0.5 GB activations = 3 GB total

## Decision: FP16 + CPU offload via `device_map="auto"`

HuggingFace will auto-distribute layers across GPU (8 GB) and CPU RAM (32 GB).
Expected split: 22/24 layers on GPU, 2/24 on CPU.

### Why this option

- **Reference quality**: FP16 dev matches FP16 prod (H100), no calibration needed
- **Fits hardware**: VRAM 6 GB + RAM 6 GB (vs 32 GB available)
- **Fast enough**: ~5 min to encode 500K test items locally
- **Same code, two environments**: dev on RTX 5060 = prod on H100, no quantization layer

### Rejected options

- **Pure CPU FP16**: ~1.5 h per 500K items, too slow for iteration
- **4-bit quantization**: -3-5% retrieval quality, threshold not portable to FP16 prod
- **Pure GPU FP16**: VRAM 10-11 GB > 8 GB available, OOM
- **Smaller model (jina-v3 570M)**: -2% vs Qwen3-4B, but bypasses all complexity

## Cost estimate: local vs cloud

| Variant | Dev cost | Notes |
|---------|----------|-------|
| RTX 5060 + CPU offload | 0 | electricity only |
| H100 spot (RunPod/Vast.ai) | ~$0.5-1/h | 5 min full encode = $0.04 |
| A100 spot | ~$0.3/h | 8 min full encode = $0.04 |

## When to consider cloud rental

Trigger conditions for moving to cloud H100/A100:
- Full-corpus encoding (13M items) takes >1 day locally
- Multiple iterations on full corpus (4+ runs)
- Calibration set computation requires multiple FP16 passes
- Time pressure: deadline <24 h

## Follow-up tickets

- T1.0.10: BiEncoder inference script (uses FP16 + CPU offload)
- T1.0.11: Full-corpus embedding cache strategy (where to store, invalidation)
- T1.0.12: Cloud rental POC (evaluate H100/A100 cost for one-shot run)
