"""Lightweight Russian lemmatizer for E-CUP products.

Simplified version of mlaw-rag/infrastructure/lemmatizer.py.
For E-CUP we don't need legal-specific stopwords (статья, часть, etc);
product queries use common Russian vocabulary.

Approach:
- Lowercase + strip punctuation
- Drop non-cyrillic tokens (Latin, digits)
- Light suffix stripping (basic rule-based)
- Returns list of lemmas or joined string

For higher accuracy (e.g. ML model based), install mawo-pymorphy3
optional dependency in production. This minimal version is dependency-free.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

_CYR_RE = re.compile(r"^[а-яё]+$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^а-яёa-z0-9 ]", re.IGNORECASE)

# Very light suffix rules — sufficient for product matching where the corpus
# already has normalized category names. For high-stakes legal text, use
# mawo-pymorphy3 or natasha.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ыми", ""),
    ("ого", ""),
    ("ому", ""),
    ("ами", ""),
    ("ями", ""),
    ("ая", ""),
    ("яя", ""),
    ("ое", ""),
    ("ее", ""),
    ("ые", ""),
    ("ие", ""),
    ("ой", ""),
    ("ый", ""),
    ("ий", ""),
    ("ам", ""),
    ("ям", ""),
    ("ах", ""),
    ("ях", ""),
    ("ов", ""),
    ("ев", ""),
    ("а", ""),
    ("я", ""),
    ("о", ""),
    ("е", ""),
    ("ы", ""),
    ("и", ""),
    ("у", ""),
    ("ю", ""),
    ("ом", ""),
    ("ем", ""),
    ("ть", ""),
    ("ешь", ""),
    ("ет", ""),
    ("ем", ""),
    ("ут", ""),
    ("ют", ""),
)


def _light_stem(word: str) -> str:
    """Light suffix stripping. Not a real lemmatizer but consistent."""
    for sfx, repl in _SUFFIX_RULES:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            return word[: -len(sfx)] + repl
    return word


class RussianLemmatizer:
    """Russian text preprocessing for retrieval (TF-IDF / BM25 / queries)."""

    def __init__(self, stopwords: Sequence[str] = ()) -> None:
        self._stop = frozenset(stopwords)

    def tokenize(self, text: str) -> list[str]:
        """Lowercase + strip punctuation + keep only cyrillic tokens."""
        if not text:
            return []
        # Lowercase
        s = text.lower()
        # Strip punctuation; keep only letters/digits/spaces
        s = _PUNCT_RE.sub(" ", s)
        # Split + filter
        tokens = [t for t in s.split() if _CYR_RE.match(t)]
        # Apply stopwords
        return [t for t in tokens if t not in self._stop]

    def lemmatize(self, text: str) -> str:
        """Tokenize + light-stem + rejoin."""
        tokens = self.tokenize(text)
        return " ".join(_light_stem(t) for t in tokens)
