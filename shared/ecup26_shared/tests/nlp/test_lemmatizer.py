"""Tests for shared.nlp.lemmatizer (Russian, lightweight)."""
from __future__ import annotations


class TestLemmatizer:
    def test_lemmatize_simple(self):
        from ecup26_shared.nlp.lemmatizer import RussianLemmatizer
        lem = RussianLemmatizer()
        out = lem.lemmatize("машины едут быстро")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_tokenize_simple(self):
        from ecup26_shared.nlp.lemmatizer import RussianLemmatizer
        lem = RussianLemmatizer()
        toks = lem.tokenize("Привет, как дела?")
        assert isinstance(toks, list)
        assert all(isinstance(t, str) for t in toks)

    def test_empty_input(self):
        from ecup26_shared.nlp.lemmatizer import RussianLemmatizer
        lem = RussianLemmatizer()
        assert lem.lemmatize("") == ""
        assert lem.tokenize("") == []

    def test_drops_non_cyrillic(self):
        from ecup26_shared.nlp.lemmatizer import RussianLemmatizer
        lem = RussianLemmatizer()
        out = lem.lemmatize("apple 123 test")
        # No Latin / digits; only cyrillic tokens
        assert "apple" not in out.lower()
