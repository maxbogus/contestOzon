"""Tests for shared.sparse.bm25_index (TF-IDF-like but BM25 - we use simple)."""
from __future__ import annotations


class TestBM25Index:
    def test_tokenize(self):
        from ecup26_shared.sparse.bm25_index import tokenize
        toks = tokenize("машина едет быстро машина")
        assert "машина" in toks
        assert toks.count("машина") == 2

    def test_tokenize_lowercase(self):
        from ecup26_shared.sparse.bm25_index import tokenize
        toks = tokenize("МАШИНА Едет")
        # Lowercased
        assert all(t == t.lower() for t in toks)

    def test_tokenize_strips_punct(self):
        from ecup26_shared.sparse.bm25_index import tokenize
        toks = tokenize("привет, мир!")
        assert "привет" in toks or "мир" in toks

    def test_empty(self):
        from ecup26_shared.sparse.bm25_index import tokenize
        assert tokenize("") == []
