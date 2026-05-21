"""
tests/test_preprocessor.py — Unit tests for the preprocessing pipeline.

Run with:
    pytest tests/test_preprocessor.py -v
"""

import pytest
from src.search.preprocessor import (
    Preprocessor,
    PreprocessorConfig,
    make_stemming_preprocessor,
    make_lemmatisation_preprocessor,
    make_bare_preprocessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pp_stem_en():
    return make_stemming_preprocessor("english")

@pytest.fixture
def pp_lemma_en():
    return make_lemmatisation_preprocessor("english")

@pytest.fixture
def pp_stem_pt():
    return make_stemming_preprocessor("portuguese")

@pytest.fixture
def pp_bare():
    return make_bare_preprocessor()


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenisation:
    def test_basic_split(self, pp_bare):
        tokens = pp_bare.process("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self, pp_bare):
        assert pp_bare.process("") == []

    def test_whitespace_only(self, pp_bare):
        assert pp_bare.process("   ") == []

    def test_punctuation_removed(self, pp_stem_en):
        tokens = pp_stem_en.process("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_sentence_segmentation(self, pp_bare):
        sentences = pp_bare.segment_sentences("I love IR. It is great.")
        assert len(sentences) == 2


# ---------------------------------------------------------------------------
# Stop word removal
# ---------------------------------------------------------------------------

class TestStopWords:
    def test_english_stopwords_removed(self, pp_stem_en):
        tokens = pp_stem_en.process("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens

    def test_portuguese_stopwords_removed(self, pp_stem_pt):
        tokens = pp_stem_pt.process("os sistemas de recuperação de informação")
        assert "os" not in tokens
        assert "de" not in tokens

    def test_stopwords_disabled(self):
        pp = Preprocessor(PreprocessorConfig(remove_stopwords=False, use_stemming=False, use_lemmatisation=False))
        tokens = pp.process("the cat sat")
        assert "the" in tokens

    def test_extra_stopwords(self):
        pp = Preprocessor(PreprocessorConfig(extra_stopwords=["custom", "word"], use_stemming=False, use_lemmatisation=False))
        tokens = pp.process("this is a custom word test")
        assert "custom" not in tokens
        assert "word" not in tokens

    def test_get_stopwords_returns_set(self, pp_stem_en):
        sw = pp_stem_en.get_stopwords()
        assert isinstance(sw, set)
        assert "the" in sw


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------

class TestStemming:
    def test_porter_stemmer_english(self, pp_stem_en):
        tokens = pp_stem_en.process("information retrieval systems")
        assert "inform" in tokens
        assert "retriev" in tokens
        assert "system" in tokens

    def test_snowball_stemmer_portuguese(self, pp_stem_pt):
        tokens = pp_stem_pt.process("recuperação informação sistemas")
        # Snowball PT reduces these to their stems
        assert any("recup" in t for t in tokens)

    def test_stemming_reduces_variants(self, pp_stem_en):
        t1 = pp_stem_en.process("running")
        t2 = pp_stem_en.process("runs")
        # Porter stemmer reduces "running" and "runs" to the same stem ("run")
        # "runner" stems to "runner" in Porter — that's correct behaviour
        assert t1[0] == t2[0]  # "running" and "runs" share the same stem
        assert len(t1[0]) < len("running")  # stemming shortened the word


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------

class TestLemmatisation:
    def test_verb_lemmatisation(self, pp_lemma_en):
        tokens = pp_lemma_en.process("running dogs are better")
        assert "run" in tokens or "running" in tokens   # depends on POS context
        assert "dog" in tokens

    def test_plural_to_singular(self, pp_lemma_en):
        tokens = pp_lemma_en.process("systems queries documents")
        assert "system" in tokens
        assert "query" in tokens
        assert "document" in tokens

    def test_lemmatisation_disabled(self):
        pp = Preprocessor(PreprocessorConfig(use_lemmatisation=False, use_stemming=False, remove_stopwords=False))
        tokens = pp.process("documents")
        assert "documents" in tokens


# ---------------------------------------------------------------------------
# process_document
# ---------------------------------------------------------------------------

class TestProcessDocument:
    def test_processes_title_and_abstract(self, pp_stem_en):
        doc = {
            "title": "Information Retrieval",
            "abstract": "This paper discusses retrieval systems.",
            "year": "2023",
        }
        tokens = pp_stem_en.process_document(doc)
        assert len(tokens) > 0
        assert "retriev" in tokens

    def test_skips_na_fields(self, pp_stem_en):
        doc = {"title": "N/A", "abstract": "N/A"}
        tokens = pp_stem_en.process_document(doc)
        assert tokens == []

    def test_custom_fields(self, pp_stem_en):
        doc = {"title": "Ignored", "authors": ["Alice Smith", "Bob Jones"]}
        tokens = pp_stem_en.process_document(doc, fields=["authors"])
        assert "alice" in tokens or "smith" in tokens

    def test_authors_list_joined(self, pp_bare):
        doc = {"authors": ["Alice", "Bob"]}
        tokens = pp_bare.process_document(doc, fields=["authors"])
        assert "Alice" in tokens or "alice" in tokens


# ---------------------------------------------------------------------------
# Min token length
# ---------------------------------------------------------------------------

class TestMinLength:
    def test_short_tokens_removed(self):
        pp = Preprocessor(PreprocessorConfig(
            min_token_length=4,
            use_stemming=False,
            use_lemmatisation=False,
            remove_stopwords=False,
        ))
        tokens = pp.process("I am a good programmer")
        for t in tokens:
            assert len(t) >= 4

    def test_default_min_length_2(self, pp_bare):
        tokens = pp_bare.process("a bb ccc")
        assert "a" not in tokens
        assert "bb" in tokens


# ---------------------------------------------------------------------------
# Unicode normalisation
# ---------------------------------------------------------------------------

class TestUnicodeNormalisation:
    def test_accent_removal(self):
        result = Preprocessor.normalise_unicode("recuperação")
        assert result == "recuperacao"

    def test_no_change_for_ascii(self):
        result = Preprocessor.normalise_unicode("hello")
        assert result == "hello"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactories:
    def test_stemming_factory(self):
        pp = make_stemming_preprocessor()
        assert pp.config.use_stemming is True
        assert pp.config.use_lemmatisation is False

    def test_lemmatisation_factory(self):
        pp = make_lemmatisation_preprocessor()
        assert pp.config.use_lemmatisation is True
        assert pp.config.use_stemming is False

    def test_bare_factory(self):
        pp = make_bare_preprocessor()
        assert pp.config.use_stemming is False
        assert pp.config.use_lemmatisation is False
        assert pp.config.remove_stopwords is False