# ADR-004: Single HuggingFace transformers pipeline (no Ollama, no vLLM)

> **Date**: 2026-08-15
> **Status**: Accepted
> **Context**: Tooling decision after E-CUP rules audit (2026-08-15)

## Decision

Use HuggingFace `transformers` + `huggingface_hub.snapshot_download` as the
**SINGLE** inference path across all 3 tracks (T1/T2/T3), in dev, test, AND
production (submission Docker).

## Rules context

This decision enforces compliance with E-CUP 2026 contest rules:

- **§2.8 Правил** — "Все сторонние программы… должны быть выпущены под лицензией,
  позволяющей их свободное коммерческое использование." HF transformers is
  Apache 2.0 ✅.
- **§2.11 / Task 2 description §36** — "Разрешено использовать малые (до 4B)
  модели с открытой лицензией". In-process load lets us pin model SHA + verify
  ≤4B before submission.
- **Task 1 §111, Task 2 §122** — "Запуск решений происходит в изолированной
  среде без доступа в интернет". All deps must be baked in Docker image.
- **§2.13** — "плагиат, несамостоятельность решения". Use of `mlaw-rag` by
  same author is OK per ADR-003.

## Rejected alternatives

| Option | Why rejected |
|---|---|
| **Ollama** | +200ms HTTP overhead per batch × N items; +1GB image bloat; reproducibility drift across GGUF quant versions; no quality benefit over in-process HF on 8GB GPU with `device_map="auto"` |
| **vLLM** | T2 models ≤4B don't benefit from PagedAttention; +500MB image bloat; same reproducibility issues |
| **LM Studio / TGI** | Same as Ollama + worse reproducibility |
| **External API (OpenAI, Anthropic, Gemini)** | ❌ **VIOLATES §2.8 + §2.11** — proprietary + closed-source. Auto-disqualification. |
| **Different models per track (dev vs prod)** | ❌ Violates "single pipeline" principle; doubles debug surface; reproducibility risk |

## Consequences

### Positive
- ✅ One `from_pretrained()` path everywhere — dev, test, prod
- ✅ Minimal debugging surface (no HTTP, no port conflicts, no GPU sharing)
- ✅ Pin exact `transformers` / `torch` versions via `uv.lock`
- ✅ Works with `device_map="auto"` + CPU offload (ADR-001)
- ✅ Reproducible: same `model.safetensors` SHA on every host
- ✅ Fits in Docker image ≤15 GB / solution ≤5 GB (R8)

### Negative
- ❌ Lose fast iteration benefit of Ollama for prompt engineering
  - **Mitigation**: smoke-test prompts on small batches in-process with quantized
    model (e.g., `bnb 4bit` for dev, full FP16 for prod)

## Updates to existing docs

- `RUNBOOK.md` §2.4 "no internet" — **unchanged** (still applies)
- `ADR-002` (≤4B rule) — **unchanged**
- `ADR-003` (mlaw-rag reuse) — **unchanged**
- `scripts/ollama_client.py` — **deleted** (2026-08-15, this commit)

## Operational rules

1. **All model loading** MUST go through `transformers.AutoModel.from_pretrained(local_path)`.
2. **All model downloads** MUST go through `huggingface-cli download <repo> --local-dir <path>`
   or `huggingface_hub.snapshot_download(repo_id=..., local_dir=...)`.
3. **No inference servers** (Ollama, vLLM, TGI, LM Studio, ExLlamaV2) in any
   `run.py` path or Docker image.
4. **No external HTTP calls** at runtime (no `requests.get`, no `httpx.post` to
   third-party APIs).
5. **Tests** MUST mock the model, not start a server.

## Migration completed

- [x] `scripts/ollama_client.py` — deleted (2026-08-15)
- [x] `T1.1.3` title — corrected (was misleadingly referencing "Ollama bge-m3")
- [x] Tickets updated with Compliance + Forbidden sections
- [x] `.clinerules/02-ecup-rules.md` — created

## Follow-up

- No new tickets. Future ML tickets must reference this ADR + R-rules from
  `.clinerules/02-ecup-rules.md`.
