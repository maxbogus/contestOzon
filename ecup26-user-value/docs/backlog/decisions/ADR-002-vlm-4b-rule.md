# ADR-002: VLM Model Size Limit (E-CUP 2026)

> **Date**: 2026-08-14
> **Status**: Accepted
> **Context**: E-CUP 2026 rules clarification

## Rule

E-CUP 2026 explicitly states: "Разрешено использовать малые (до 4B) модели с открытой лицензией".

Only models **≤4B parameters** are allowed for submission.

## Impact

### Disqualified models (removed from backlog)

| Model | Params | Status | Reason |
|-------|--------|--------|--------|
| DeepSeek-VL-distill | ~30B | ❌ Removed | >4B, disqualified |
| InternVL3.5-8B | ~8B | ❌ Removed | >4B, disqualified |

### Approved models (≤4B, Apache 2.0)

| Model | Params | Status | Notes |
|-------|--------|--------|-------|
| Qwen3-VL-Embedding-2B | 2B | ✅ Allowed | Primary model for T2 |
| Qwen3-VL-4B | 4B | ✅ Allowed | If needed |
| MiniCPM-V-4.6 | **~1.2B** | ✅ Allowed | "4.6" = version, NOT params. Qwen3.5-0.8B + SigLIP2-400M. Confirmed by README + official task docs. |
| Any model ≤4B | varies | ✅ Allowed | With open license |

> **Correction 2026-08-15**: ADR-002 originally classified MiniCPM-V-4.6 as 4.6B params.
> This was wrong — the "4.6" is the **model version number**, not parameter count.
> MiniCPM-V-4.6 is built on **Qwen3.5-0.8B (~0.8B) + SigLIP2-400M (~0.4B)** = **~1.2B total**.
> Official task docs `task2_quality_data.md` place MiniCPM-V-4.6 in the "≤3B models" list.
> Therefore MiniCPM-V-4.6 IS allowed for T2 submission.

## Consequences

1. **T2.6.14 (DeepSeek-distill)** — deleted, impossible to use
2. **T2.6.15 (InternVL3.5 study)** — deleted, model not allowed for submission
3. **T2.6.6 (MiniCPM)** — **CHANGED**: allowed for submission (≤4B confirmed). Was incorrectly marked research-only.
4. **T2.6.7 (VLM benchmark)** — can include MiniCPM-V-4.6 as a comparison candidate

## Updated T2 pipeline

```
Data → Qwen3-VL-Embedding-2B (2B) → classifier → submit.csv
```

No >4B models in submission pipeline.

## Cloud strategy for T2

- Local dev: Qwen3-VL-2B FP16 (fits in RTX 5060 8GB)
- Submission: Qwen3-VL-2B on H100 (fast, <20 min)
- No cloud rental needed for T2 (unlike T1)
