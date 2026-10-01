# E-CUP 2026 — Rules Cheatsheet

> **READ THIS FIRST** before any code change.
>
> **Source of truth**:
> 1. `docs/Правила E-CUP_23.07.26.pdf` (Russian, official rules)
> 2. `docs/tasks/task{1,2,3}_*.md` (English, official task descriptions)
> 3. `.clinerules/02-ecup-rules.md` (machine-readable constraints for Cline)
> 4. `docs/backlog/decisions/ADR-00{1,2,3,4}.md` (internal decisions)

## One-line summary

**Open-source only · T2 models ≤4B · No internet in Docker · Single HuggingFace transformers pipeline.**

## 10 hard rules (R1–R10)

| # | Rule | Reference |
|---|---|---|
| R1 | Open-source code only (Apache 2.0 / MIT / BSD) | §2.8 Правил |
| R2 | T2 models ≤4B params only | Task 2 §36, ADR-002 |
| R3 | All models open-license + commercially usable | §2.11, Task 1 §35, Task 2 §36 |
| R4 | No internet at runtime in Docker | Task 1 §111, Task 2 §122 |
| R5 | No plagiarism / self-work required | §2.13, ADR-003 (mlaw-rag OK, same author) |
| R6 | Reproducible code (pin versions, SHA) | Task 1 §35, Task 2 §47 |
| R7 | Single HF transformers pipeline (no Ollama/vLLM) | ADR-004 |
| R8 | Docker submission contract (≤5 GB archive, ≤15 GB image, `python -u run.py`) | Task 1 §84-100, Task 2 §98-114 |
| R9 | Time limits: T1 Private 13 min, T2 Private 40 min | Task 1 §153, Task 2 §162 |
| R10 | Unique IDs in `submit.csv` (no chunk_id collisions) | Note 002 |

## Per-task quick reference

| Task | Metric | Time (Private) | Models | Submit format |
|---|---|---|---|---|
| T1 matching | Macro PR-AUC | 13 min | HF transformers (no size cap; Qwen3-Embedding-4B + BGE-M3) | `submit.csv: id1,id2,predict` |
| T2 quality  | Macro F1 | 40 min | **≤4B VLM** (Qwen3-VL-Embedding-2B preferred) | `submit.csv: id,result` |
| T3 user-val | MAE/RMSLE | TBD (check task3 docs) | HF transformers, no LLM needed | `submit.csv: user_id,predict` |

## Pre-commit self-check (mandatory)

```bash
# 1. No external inference servers anywhere in src/
! grep -rni 'ollama\|vllm\|tgi\|lm.studio\|exllamav2' ecup26-*/src/ scripts/ 2>/dev/null

# 2. No proprietary APIs
! grep -rni 'openai\.com\|anthropic\.com\|gpt-4\|claude-\|gemini-pro' ecup26-*/src/ 2>/dev/null

# 3. All T2 models ≤4B (if any VLM is added)
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('models/Qwen3-VL-Embedding-2B').architectures)"

# 4. Tests pass
cd shared             && uv run pytest -q
cd ecup26-matching    && uv run pytest -q
cd ecup26-quality     && uv run pytest -q
cd ecup26-user-value  && uv run pytest -q

# 5. Each sub-repo has run.py + metadata.json
for r in ecup26-matching ecup26-quality ecup26-user-value; do
  ls $r/run.py $r/metadata.json
done

# 6. submit.csv IDs are unique
python -c "import pandas as pd; df=pd.read_csv('submit.csv'); assert df.iloc[:,0].is_unique, 'duplicate IDs'"
```

## Forbidden actions (NEVER do these)

❌ Use Ollama / vLLM / TGI / LM Studio / ExLlamaV2 as inference server.
❌ Use proprietary LLM (GPT-4, Claude, Gemini, GigaChat, YandexGPT).
❌ Use models **>4B params** for T2 submission (DeepSeek-VL, InternVL3.5-8B, MiniCPM-V-4.6 ❌).
❌ `pip install` inside Docker container.
❌ `requests.get()` / `httpx.post()` / network calls at runtime.
❌ Copy code from non-author sources (except `mlaw-rag` by same author — see ADR-003).
❌ Submit `submit.csv` without asserting `df.iloc[:, 0].is_unique`.
❌ Skip time-limit verification before commit.
❌ Use `ckpt` (not safetensors) for model weights.
❌ Modify `train.parquet` / `data.csv` in-place during training.

## Model whitelist (T2 only — R2)

| Model | Params | License | Status |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | 2B | Apache 2.0 | ✅ PRIMARY |
| `Qwen/Qwen3-VL-2B-instruct` | 2B | Apache 2.0 | ✅ |
| `Qwen/Qwen3.5-4B` | 4B | Apache 2.0 | ✅ |
| `google/gemma-4-E2B-it` | 2B | Apache 2.0 | ✅ |
| `google/gemma-4-E4B-it` | 4B | Apache 2.0 | ✅ |
| `OpenGVLab/InternVL3_5-2B` | 2B | Apache 2.0 | ✅ |
| `Nanbeige/Nanbeige4.2-3B` | 3B | Apache 2.0 | ✅ |
| `PaddlePaddle/PaddleOCR-VL-1.5` | 1.5B | Apache 2.0 | ✅ |
| `vikhyatk/moondream2` | 1.86B | Apache 2.0 | ✅ |
| `openbmb/MiniCPM-V-4.6` | **~1.2B** | Apache 2.0 | ✅ **Allowed** — "4.6" is version, not param count |
| `DeepSeek-VL-distill` | ~30B | varies | ❌ Disqualified |
| `InternVL3.5-8B` | 8B | Apache 2.0 | ❌ **>4B** |

For T1 and T3: no model-size cap, but R3 (open license) and R4 (no internet) still apply.

## Hardware at eval time (R8)

| | T1 | T2 |
|---|---|---|
| CPU | 20 cores | 20 cores |
| RAM | 200 GB | 100 GB |
| GPU | 1× NVIDIA H100 80 GB | 1× NVIDIA H100 80 GB |
| Image | ≤15 GB compressed | ≤15 GB compressed |
| Solution archive | ≤5 GB | ≤5 GB |
| Entry point | `python -u run.py` | `python -u run.py` (T2/T3 use `--test_data_path`/`--output_path`) |

## TDD workflow (per `.clinerules/`)

For every ticket:
1. **RED** — write failing test FIRST
2. **GREEN** — minimum code to pass
3. **REFACTOR** — clean up while green
4. Verify with `cd <repo> && uv run pytest tests/ -v`
5. Update ticket status: `done` after `make ready` shows it.
