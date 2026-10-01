"""TF-IDF style sparse retrieval for E-CUP.

Lightweight re-implementation of mlaw-rag bm25_index.py — we use
scikit-learn's TfidfVectorizer (already in ml deps via sklearn) for the
heavy lifting, but expose a bm25s-style API.

Note: for production use, you can swap in `bm25s` package (pip install bm25s).
This file keeps only the pure-python fallback so the shared package is
zero-dep.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase + extract word tokens (cyrillic + latin + digits).

    Returns empty list for empty / whitespace-only input.
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def term_freq(tokens: Sequence[str]) -> dict[str, int]:
    """Term frequency: count of each unique token in the sequence."""
    return dict(Counter(tokens))


def tfidf_score(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    idf: dict[str, float],
) -> float:
    """Simple TF-IDF cosine-style score.

    score(q, d) = sum over shared tokens: tf(d, t) * idf(t)

    For full TF-IDF + cosine, use sklearn TfidfVectorizer (recommended for
    real eval); this fn is for tests / quick prototyping.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_tf = term_freq(doc_tokens)
    return sum(doc_tf.get(t, 0) * idf.get(t, 0.0) for t in set(query_tokens))


def top_k(scored: Sequence[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Sort by score desc, return top-k with ties broken by doc_id asc."""
    return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]
