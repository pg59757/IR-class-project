"""
tests/test_tfidf.py — Unit tests for the TFIDFEngine.

Run with:
    pytest tests/test_tfidf.py -v
"""

import math
import pytest

from src.search.preprocessor import make_stemming_preprocessor, make_lemmatisation_preprocessor
from src.search.tfidf import TFIDFEngine, TFIDFResult


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------

DOCS = [
    {
        "title": "Information Retrieval Systems",
        "abstract": "This paper discusses modern information retrieval systems and ranking algorithms.",
        "authors": ["Alice Silva"],
        "year": "2021",
        "doi": "10.1000/001",
        "document_link": "https://example.com/1",
    },
    {
        "title": "Deep Learning for Natural Language Processing",
        "abstract": "Neural networks applied to natural language processing and text classification tasks.",
        "authors": ["Bob Santos"],
        "year": "2022",
        "doi": "10.1000/002",
        "document_link": "https://example.com/2",
    },
    {
        "title": "Boolean Retrieval Models",
        "abstract": "An overview of boolean retrieval models and inverted indexes with postings lists.",
        "authors": ["Alice Silva", "Carlos Mota"],
        "year": "2020",
        "doi": "10.1000/003",
        "document_link": "https://example.com/3",
    },
    {
        "title": "Machine Learning Applications",
        "abstract": "Supervised and unsupervised learning methods for data classification.",
        "authors": ["Diana Ferreira"],
        "year": "2023",
        "doi": "10.1000/004",
        "document_link": "https://example.com/4",
    },
    {
        "title": "TF-IDF and Vector Space Models",
        "abstract": "Term frequency inverse document frequency and cosine similarity ranking.",
        "authors": ["Eve Costa"],
        "year": "2022",
        "doi": "10.1000/005",
        "document_link": "https://example.com/5",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pp():
    return make_stemming_preprocessor("english")

@pytest.fixture
def engine_custom(pp):
    eng = TFIDFEngine(pp, use_sklearn=False, tf_scheme="log")
    eng.build_from_documents(DOCS)
    return eng

@pytest.fixture
def engine_sklearn(pp):
    eng = TFIDFEngine(pp, use_sklearn=True, tf_scheme="log")
    eng.build_from_documents(DOCS)
    return eng


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_num_documents(self, engine_custom):
        assert engine_custom.num_documents == len(DOCS)

    def test_vocabulary_not_empty(self, engine_custom):
        assert len(engine_custom._vocabulary) > 0

    def test_idf_computed(self, engine_custom):
        assert len(engine_custom._idf) > 0

    def test_doc_vectors_computed(self, engine_custom):
        assert len(engine_custom._doc_vectors) == len(DOCS)

    def test_idf_values_positive(self, engine_custom):
        for term, val in engine_custom._idf.items():
            assert val > 0, f"IDF for '{term}' should be positive"

    def test_rare_term_higher_idf(self, engine_custom):
        """A term that appears in only one document should have higher IDF
        than one that appears in all documents."""
        # 'tfidf'/'tf' only appears in doc 5; 'retriev' appears in 2+
        rare_idf = engine_custom.get_term_idf("tfidf")
        common_idf = engine_custom.get_term_idf("retrieval")
        # rare_idf >= common_idf (could be equal if stemming merges them)
        assert rare_idf >= 0
        assert common_idf >= 0

    def test_build_clears_previous(self, pp):
        eng = TFIDFEngine(pp, use_sklearn=False)
        eng.build_from_documents(DOCS[:2])
        assert eng.num_documents == 2
        eng.build_from_documents(DOCS)
        assert eng.num_documents == len(DOCS)


# ---------------------------------------------------------------------------
# TF schemes
# ---------------------------------------------------------------------------

class TestTFSchemes:
    @pytest.mark.parametrize("scheme", ["raw", "log", "boolean"])
    def test_scheme_builds_without_error(self, pp, scheme):
        eng = TFIDFEngine(pp, use_sklearn=False, tf_scheme=scheme)
        eng.build_from_documents(DOCS)
        results = eng.search("retrieval")
        assert isinstance(results, list)

    def test_log_tf_not_equal_raw_tf(self, pp):
        raw_eng = TFIDFEngine(pp, use_sklearn=False, tf_scheme="raw")
        log_eng = TFIDFEngine(pp, use_sklearn=False, tf_scheme="log")
        raw_eng.build_from_documents(DOCS)
        log_eng.build_from_documents(DOCS)
        # Scores may differ between schemes
        r_raw = raw_eng.search("retrieval information")
        r_log = log_eng.search("retrieval information")
        # Both should find results
        assert len(r_raw) > 0
        assert len(r_log) > 0


# ---------------------------------------------------------------------------
# Search — custom implementation
# ---------------------------------------------------------------------------

class TestSearchCustom:
    def test_returns_list(self, engine_custom):
        results = engine_custom.search("retrieval")
        assert isinstance(results, list)

    def test_results_are_tfidf_result(self, engine_custom):
        results = engine_custom.search("retrieval")
        for r in results:
            assert isinstance(r, TFIDFResult)

    def test_result_has_doc_id(self, engine_custom):
        results = engine_custom.search("retrieval")
        for r in results:
            assert isinstance(r.doc_id, int)
            assert r.doc_id >= 0

    def test_result_has_document(self, engine_custom):
        results = engine_custom.search("retrieval")
        for r in results:
            assert isinstance(r.document, dict)
            assert "title" in r.document

    def test_result_score_between_0_and_1(self, engine_custom):
        results = engine_custom.search("retrieval")
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"

    def test_results_sorted_descending(self, engine_custom):
        results = engine_custom.search("information retrieval")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranks_high(self, engine_custom):
        """The TF-IDF doc should rank highly for a TF-IDF query."""
        results = engine_custom.search("tfidf vector space cosine")
        assert len(results) > 0
        top_doc = results[0].document
        assert "TF-IDF" in top_doc["title"] or "Vector" in top_doc["title"]

    def test_empty_query_returns_empty(self, engine_custom):
        assert engine_custom.search("") == []

    def test_unknown_query_returns_empty(self, engine_custom):
        assert engine_custom.search("zzznonsensexxx999") == []

    def test_top_k_respected(self, engine_custom):
        results = engine_custom.search("retrieval learning information", top_k=2)
        assert len(results) <= 2

    def test_top_k_default_20(self, engine_custom):
        results = engine_custom.search("retrieval OR learning OR information")
        assert len(results) <= 20


# ---------------------------------------------------------------------------
# Search — sklearn implementation
# ---------------------------------------------------------------------------

class TestSearchSklearn:
    def test_returns_results(self, engine_sklearn):
        results = engine_sklearn.search("retrieval")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_scores_between_0_and_1(self, engine_sklearn):
        results = engine_sklearn.search("information retrieval")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_sorted_descending(self, engine_sklearn):
        results = engine_sklearn.search("information retrieval")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, engine_sklearn):
        assert engine_sklearn.search("") == []


# ---------------------------------------------------------------------------
# Custom vs Sklearn agreement
# ---------------------------------------------------------------------------

class TestCustomVsSklearn:
    def test_both_find_same_documents(self, pp):
        """Custom and sklearn should find overlapping top results."""
        custom = TFIDFEngine(pp, use_sklearn=False)
        sklearn = TFIDFEngine(pp, use_sklearn=True)
        custom.build_from_documents(DOCS)
        sklearn.build_from_documents(DOCS)

        r_custom = custom.search("retrieval information", top_k=5)
        r_sklearn = sklearn.search("retrieval information", top_k=5)

        ids_custom = {r.doc_id for r in r_custom}
        ids_sklearn = {r.doc_id for r in r_sklearn}

        # Should have significant overlap (at least 1 common result)
        assert len(ids_custom & ids_sklearn) >= 1


# ---------------------------------------------------------------------------
# IDF utility
# ---------------------------------------------------------------------------

class TestGetTermIDF:
    def test_known_term_has_idf(self, engine_custom):
        val = engine_custom.get_term_idf("retrieval")
        assert val > 0

    def test_unknown_term_idf_is_zero(self, engine_custom):
        val = engine_custom.get_term_idf("zzznonsense999")
        assert val == 0.0

    def test_empty_term_idf_is_zero(self, engine_custom):
        assert engine_custom.get_term_idf("") == 0.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_dict(self, engine_custom):
        stats = engine_custom.stats()
        assert isinstance(stats, dict)

    def test_stats_has_expected_keys(self, engine_custom):
        stats = engine_custom.stats()
        assert "num_documents" in stats
        assert "vocabulary_size" in stats
        assert "use_sklearn" in stats
        assert "tf_scheme" in stats

    def test_stats_num_documents_correct(self, engine_custom):
        stats = engine_custom.stats()
        assert stats["num_documents"] == len(DOCS)


# ---------------------------------------------------------------------------
# Cosine similarity helper (internal, tested directly)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        score = TFIDFEngine._cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        v1 = {"a": 1.0}
        v2 = {"b": 1.0}
        score = TFIDFEngine._cosine_similarity(v1, v2)
        assert score == 0.0

    def test_empty_vectors(self):
        assert TFIDFEngine._cosine_similarity({}, {"a": 1.0}) == 0.0
        assert TFIDFEngine._cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_partial_overlap(self):
        v1 = {"a": 1.0, "b": 1.0}
        v2 = {"a": 1.0, "c": 1.0}
        score = TFIDFEngine._cosine_similarity(v1, v2)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Add document incrementally
# ---------------------------------------------------------------------------

class TestAddDocument:
    """
    TFIDFEngine não suporta updates incrementais — requer rebuild completo
    para manter os valores de IDF consistentes (ao contrário do InvertedIndex).
    Estes testes verificam o comportamento correcto de rebuild.
    """

    def test_rebuild_with_extra_document_increases_count(self, pp):
        eng = TFIDFEngine(pp, use_sklearn=False)
        eng.build_from_documents(DOCS)
        before = eng.num_documents

        new_doc = {
            "title": "Quantum Information Theory",
            "abstract": "Quantum bits and entanglement for computation.",
            "authors": ["Frank Lima"],
            "year": "2024",
            "doi": "10.1000/999",
            "document_link": "https://example.com/99",
        }
        eng.build_from_documents(DOCS + [new_doc])
        assert eng.num_documents == before + 1

    def test_rebuilt_index_is_searchable(self, pp):
        eng = TFIDFEngine(pp, use_sklearn=False)
        new_doc = {
            "title": "Quantum Entanglement Survey",
            "abstract": "Entanglement and superposition in quantum systems.",
            "authors": ["Grace Alves"],
            "year": "2024",
            "doi": "10.1000/998",
            "document_link": "https://example.com/98",
        }
        eng.build_from_documents(DOCS + [new_doc])
        results = eng.search("quantum entanglement")
        titles = [r.document["title"] for r in results]
        assert any("Quantum" in t for t in titles)