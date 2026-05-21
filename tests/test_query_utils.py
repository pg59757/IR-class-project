"""
tests/test_query_utils.py — Tests for query expansion, phrase search and snippet generation.

Run:
    pytest tests/test_query_utils.py -v
"""

import pytest


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
]


# ---------------------------------------------------------------------------
# QueryExpander  (REQ-B47)
# ---------------------------------------------------------------------------

class TestQueryExpander:

    @pytest.fixture
    def expander(self):
        from src.search.query_utils import QueryExpander
        return QueryExpander(max_synonyms=2)

    def test_expand_returns_string(self, expander):
        result = expander.expand("information retrieval")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_original_term_preserved(self, expander):
        result = expander.expand("database")
        assert "database" in result.lower()

    def test_boolean_operators_preserved(self, expander):
        result = expander.expand("information AND retrieval")
        assert "AND" in result

    def test_or_operator_preserved(self, expander):
        result = expander.expand("machine OR learning")
        assert "OR" in result

    def test_expand_list_returns_list(self, expander):
        terms = expander.expand_list("information retrieval")
        assert isinstance(terms, list)
        assert len(terms) >= 2  # at least the original terms

    def test_expand_list_contains_original_terms(self, expander):
        terms = expander.expand_list("document")
        assert "document" in terms

    def test_stopwords_not_expanded(self, expander):
        # "the" is a stopword — it should be returned unchanged with no OR group
        result = expander.expand("the information")
        # "the" should NOT become "the OR ..."
        assert result.count("OR") <= 1  # at most from "information", not from "the"

    def test_expand_empty_query(self, expander):
        result = expander.expand("")
        assert result == ""

    def test_expand_single_term(self, expander):
        result = expander.expand("neural")
        assert "neural" in result.lower()

    def test_max_synonyms_respected(self, expander):
        # With max_synonyms=2, "retrieve OR a OR b" — at most 2 synonyms added
        result = expander.expand("retrieve")
        parts = result.split(" OR ")
        assert len(parts) <= 3  # original + up to 2 synonyms


# ---------------------------------------------------------------------------
# PhraseQuery  (REQ-B48)
# ---------------------------------------------------------------------------

class TestPhraseQuery:

    @pytest.fixture
    def pq(self):
        from src.search.query_utils import PhraseQuery
        return PhraseQuery(window=0)

    @pytest.fixture
    def proximity_pq(self):
        from src.search.query_utils import PhraseQuery
        return PhraseQuery(window=3)

    def test_exact_phrase_match(self, pq):
        assert pq.matches(
            "information retrieval",
            "This paper is about information retrieval systems."
        )

    def test_exact_phrase_no_match(self, pq):
        assert not pq.matches(
            "information retrieval",
            "This paper is about retrieval of information."
        )

    def test_case_insensitive(self, pq):
        assert pq.matches("Information Retrieval", "information retrieval systems")

    def test_proximity_match_within_window(self, proximity_pq):
        # "information" and "retrieval" within 3 words of each other
        assert proximity_pq.matches(
            "information retrieval",
            "Modern information-based retrieval approaches."
        )

    def test_find_all_returns_positions(self, pq):
        text = "information retrieval is used in information retrieval systems"
        positions = pq.find_all("information retrieval", text)
        assert len(positions) == 2

    def test_find_all_empty_when_no_match(self, pq):
        positions = pq.find_all("quantum computing", "information retrieval systems")
        assert positions == []

    def test_single_word_phrase(self, pq):
        assert pq.matches("retrieval", "information retrieval models")

    def test_empty_phrase(self, pq):
        # Empty phrase shouldn't crash
        result = pq.matches("", "some text")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SnippetGenerator  (REQ-B50)
# ---------------------------------------------------------------------------

class TestSnippetGenerator:

    @pytest.fixture
    def gen(self):
        from src.search.query_utils import SnippetGenerator
        return SnippetGenerator(window=80, highlight_template="**{term}**")

    def test_extract_returns_string(self, gen):
        snippet = gen.extract(
            "This paper discusses information retrieval systems.",
            query_terms=["information", "retrieval"],
        )
        assert isinstance(snippet, str)

    def test_snippet_contains_highlighted_term(self, gen):
        snippet = gen.extract(
            "Information retrieval systems are widely used.",
            query_terms=["retrieval"],
        )
        assert "**retrieval**" in snippet or "**Retrieval**" in snippet

    def test_snippet_truncated_with_ellipsis(self, gen):
        long_text = "irrelevant words " * 50 + "information retrieval" + " more words " * 50
        snippet = gen.extract(long_text, query_terms=["information", "retrieval"])
        assert "…" in snippet

    def test_fallback_when_no_term_found(self, gen):
        snippet = gen.extract(
            "This is a completely unrelated text without any query term.",
            query_terms=["quantum", "physics"],
        )
        assert isinstance(snippet, str)
        assert len(snippet) > 0

    def test_empty_text_returns_empty(self, gen):
        assert gen.extract("", query_terms=["retrieval"]) == ""

    def test_extract_for_document(self, gen):
        doc = {
            "title": "Information Retrieval",
            "abstract": "This paper discusses modern information retrieval systems.",
        }
        snippet = gen.extract_for_document(doc, query_terms=["retrieval"])
        assert isinstance(snippet, str)
        assert len(snippet) > 0

    def test_extract_for_document_falls_back_to_title(self, gen):
        doc = {
            "title": "Retrieval Systems",
            "abstract": "N/A",
        }
        snippet = gen.extract_for_document(doc, query_terms=["retrieval"])
        assert "retrieval" in snippet.lower() or "Retrieval" in snippet

    def test_multiple_terms_highlighted(self, gen):
        snippet = gen.extract(
            "Information retrieval and indexing systems.",
            query_terms=["information", "indexing"],
        )
        # At least one term should be highlighted
        assert "**" in snippet

    def test_html_highlight_template(self):
        from src.search.query_utils import SnippetGenerator
        gen_html = SnippetGenerator(window=80, highlight_template="<mark>{term}</mark>")
        snippet = gen_html.extract(
            "Information retrieval systems.",
            query_terms=["retrieval"],
        )
        assert "<mark>" in snippet


# ---------------------------------------------------------------------------
# PerformanceEvaluator  (REQ-B56 to B62)
# ---------------------------------------------------------------------------

class TestPerformanceEvaluator:

    @pytest.fixture
    def evaluator(self):
        from src.search.evaluation import PerformanceEvaluator
        return PerformanceEvaluator()

    def test_benchmark_indexing_returns_stats(self, evaluator):
        stats = evaluator.benchmark_indexing(DOCS)
        assert len(stats) == 2
        methods = [s.method for s in stats]
        assert "stemming" in methods
        assert "lemmatisation" in methods

    def test_benchmark_indexing_elapsed_positive(self, evaluator):
        stats = evaluator.benchmark_indexing(DOCS)
        for s in stats:
            assert s.elapsed_seconds > 0

    def test_benchmark_indexing_vocab_positive(self, evaluator):
        stats = evaluator.benchmark_indexing(DOCS)
        for s in stats:
            assert s.vocabulary_size > 0

    def test_benchmark_indexing_memory_positive(self, evaluator):
        stats = evaluator.benchmark_indexing(DOCS)
        for s in stats:
            assert s.peak_memory_mb >= 0

    def test_benchmark_queries_returns_query_stats(self, evaluator):
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.tfidf import TFIDFEngine
        pp = make_stemming_preprocessor("english")
        engine = TFIDFEngine(pp)
        engine.build_from_documents(DOCS)

        queries = ["information retrieval", "neural network"]
        stats = evaluator.benchmark_queries(engine, queries, method_label="custom")
        assert len(stats) == 2
        for s in stats:
            assert s.elapsed_ms >= 0
            assert s.method == "custom"

    def test_compare_rankings_returns_comparisons(self, evaluator):
        comparisons = evaluator.compare_rankings(DOCS, ["information retrieval"])
        assert len(comparisons) == 1
        c = comparisons[0]
        assert isinstance(c.custom_top5, list)
        assert isinstance(c.sklearn_top5, list)
        assert 0 <= c.overlap_count <= 5

    def test_full_report_structure(self, evaluator):
        report = evaluator.full_report(DOCS, queries=["retrieval"])
        assert len(report.indexing) >= 2
        assert len(report.queries) >= 2
        assert len(report.ranking_comparisons) >= 1

    def test_full_report_summary_is_string(self, evaluator):
        report = evaluator.full_report(DOCS, queries=["retrieval"])
        summary = report.summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_full_report_to_dict(self, evaluator):
        report = evaluator.full_report(DOCS, queries=["retrieval"])
        d = report.to_dict()
        assert "indexing" in d
        assert "queries" in d
        assert "ranking_comparisons" in d


# ---------------------------------------------------------------------------
# Batch indexing (REQ-B59)
# ---------------------------------------------------------------------------

class TestBatchIndexing:

    def test_batch_indexing_produces_same_vocab(self):
        from src.search.evaluation import build_index_in_batches
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.inverted_index import InvertedIndex

        pp = make_stemming_preprocessor("english")
        idx_normal = InvertedIndex(pp)
        idx_normal.build_from_documents(DOCS)

        idx_batch = build_index_in_batches(DOCS, batch_size=2)
        assert idx_batch.num_documents == idx_normal.num_documents
        assert idx_batch.vocabulary_size == idx_normal.vocabulary_size

    def test_batch_size_one(self):
        from src.search.evaluation import build_index_in_batches
        idx = build_index_in_batches(DOCS, batch_size=1)
        assert idx.num_documents == len(DOCS)


# ---------------------------------------------------------------------------
# Config (REQ-B67)
# ---------------------------------------------------------------------------

class TestConfig:

    def test_settings_importable(self):
        from src.config import settings
        assert settings is not None

    def test_settings_has_required_fields(self):
        from src.config import settings
        assert hasattr(settings, "TOP_K_DEFAULT")
        assert hasattr(settings, "TOP_K_MAX")
        assert hasattr(settings, "DEFAULT_PREPROCESSING")
        assert hasattr(settings, "REMOVE_STOPWORDS")
        assert hasattr(settings, "TF_SCHEME")
        assert hasattr(settings, "BATCH_SIZE")

    def test_settings_defaults_are_sensible(self):
        from src.config import settings
        assert settings.TOP_K_DEFAULT > 0
        assert settings.TOP_K_MAX >= settings.TOP_K_DEFAULT
        assert settings.DEFAULT_PREPROCESSING in ("stemming", "lemmatisation", "bare")
        assert settings.TF_SCHEME in ("raw", "log", "boolean")
        assert settings.BATCH_SIZE > 0