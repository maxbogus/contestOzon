# ADR-003: mlaw-rag reuse strategy for E-CUP

> **Date**: 2026-08-15
> **Status**: Accepted
> **Context**: Reuse from mlaw-rag (56 tickets, 85 Python modules, 4111+ lines) in E-CUP contest

## Source project

**mlaw-rag** (Maxim Boguslavskiy, same author):
- End-to-end RAG pipeline for MLAW corpus
- TDD-first, 347 unit tests, ruff/mypy enforced
- 56 tickets completed (T-001..T-056 + T-057)
- No LICENSE file → code is copyrighted by author (Maxim Boguslavskiy)

## Legal status

- ✅ Author (Maxim Boguslavskiy) may reuse own code in any project
- ✅ E-CUP rules allow open-source models (Qwen3-Embedding-4B, bge-m3, etc.)
- ✅ All mlaw-rag dependencies are open-source (Apache 2.0 / MIT)
- ⚠️ Jury sees source code for top-15 teams
- ✅ mlaw-rag is NOT a "proprietary library" — it is author's own code

## Reuse strategy

### Directly reusable (copy with domain adaptation)

| Module | Lines | E-CUP Target | Effort Saved |
|--------|-------|---------------|-------------|
| `application/embedding_service.py` | 51 | T1.0.9 BiEncoder | ~4h |
| `application/embedding_model.py` | 56 | T1.0.9 model loader | ~2h |
| `application/embedding_batch.py` | 94 | T1.0.10 batch inference | ~3h |
| `application/mrl_sweep.py` | 89 | T1.0.13 MRL sweep | ~4h |
| `application/retrieval_cascade.py` | 79 | T1.4.7 cascade | ~6h |
| `domain/rrf_fusion.py` | 37 | T1.4.7 RRF | ~1h |
| `infrastructure/bm25_index.py` | 71 | T1.2.2 sparse features | ~2h |
| `infrastructure/bm25_search.py` | 34 | T1.2.2 BM25 search | ~1h |
| `infrastructure/sparse_embedding.py` | 57 | T1.2.2 sparse | ~1h |
| `infrastructure/lemmatizer.py` | 106 | T1.1.1 preprocessing | ~2h |
| `infrastructure/reranker.py` | 56 | T1.4.7 cross-encoder | ~2h |

### Eval framework (high-value reuse)

| Module | Lines | E-CUP Target | Effort Saved |
|--------|-------|---------------|-------------|
| `application/metrics.py` | 83 | T1.5.1 PR-AUC wrapper | ~2h |
| `application/eval_runner.py` | 103 | T1.5.1 eval pipeline | ~2h |
| `application/eval_results_writer.py` | 54 | T1.5.1 CSV/JSON writer | ~1h |
| `application/control_test.py` | 87 | T1.5.4 sanity check | ~1h |
| `application/ab_test.py` | 213 | T1.5.6 A/B test | ~3h |
| `application/eval_report.py` | 266 | T1.5.7 final report | ~3h |
| `application/figure_generator.py` | 107 | T1.5.5 figures | ~1h |
| `application/sweep_runner.py` | 70 | T1.5.3 sweeps | ~1h |
| `application/sweep_aggregator.py` | 122 | T1.5.3 Pareto | ~1h |
| `application/ablation.py` | 103 | T1.5.2 ablation | ~3h |

### Pipeline pattern (new discovery)

| Module | Lines | E-CUP Target | Effort Saved |
|--------|-------|---------------|-------------|
| `application/index_builder.py` | 125 | T1.4.8 submission pipeline | ~3h |
| `application/scope_sweep.py` | 181 | T1.5.3 scaling analysis | ~2h |

### Not reusable (legal-specific)

| Module | Reason |
|--------|--------|
| `domain/chunking.py` | MLAW contents tree structure |
| `domain/pit_filter.py` | Point-in-time legal validity |
| `domain/edition_*.py` | Legal act editions |
| `infrastructure/oix_reader.py` | MLAW .oix format |
| `infrastructure/qdrant_*.py` | Qdrant vector DB |
| `domain/oos_detector.py` | Legal out-of-scope detection |
| `infrastructure/faithfulness.py` | NLI for legal QA |

## Estimated savings

| Category | Hours Saved |
|----------|-------------|
| Embedding + MRL | ~10h |
| Eval framework | ~14h |
| Sweep infrastructure | ~4h |
| Sparse retrieval | ~4h |
| Cascade + RRF | ~8h |
| Pipeline patterns | ~5h |
| **TOTAL** | **~45h** |

## Total E-CUP effort (with reuse)

| Track | Without reuse | With reuse | Savings |
|-------|--------------|------------|---------|
| T1 matching | ~44h | ~12h | ~32h |
| T2 quality | ~25h | ~15h | ~10h |
| T3 user value | ~20h | ~12h | ~8h |
| **TOTAL** | **~89h** | **~39h** | **~50h** |

## Implementation notes

1. **Domain adaptation**: Replace `LegalAct`/`Chunk` with `Item`/`ItemPair`
2. **Qdrant → FAISS**: Replace vector DB calls with numpy/FAISS
3. **Metrics**: Replace nDCG/MRR with PR-AUC (E-CUP metric)
4. **No Celery/Redis**: Submission is single-process Docker, no distributed
5. **Citation precision**: Not needed (no LLM citations in E-CUP)

## Follow-up

- Create `shared/` package in contest/ root for cross-repo code
- Consider monorepo structure (contest/ + ecup26-*/ + shared/)
