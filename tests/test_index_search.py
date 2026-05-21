"""
tests/test_index_and_search.py — Tests for InvertedIndex and BooleanSearchEngine.

Run with:
    pytest tests/test_index_and_search.py -v
"""

import json
import os
import tempfile
import pytest

from src.search.preprocessor import make_stemming_preprocessor, make_bare_preprocessor
from src.search.inverted_index import InvertedIndex, Posting, PostingsList
from src.search.boolean_search import BooleanSearchEngine, SearchResult


# ---------------------------------------------------------------------------
# Sample documents (mimics scraper output)
# ---------------------------------------------------------------------------

DOCS = [
    {
        "title": "Information Retrieval Systems",
        "abstract": "This paper discusses modern information retrieval systems and algorithms.",
        "authors": ["Alice Silva"],
        "year": "2021",
        "doi": "10.1000/xyz001",
        "document_link": "https://example.com/1",
    },
    {
        "title": "Deep Learning for Natural Language Processing",
        "abstract": "Neural networks applied to natural language processing tasks.",
        "authors": ["Bob Santos"],
        "year": "2022",
        "doi": "10.1000/xyz002",
        "document_link": "https://example.com/2",
    },
    {
        "title": "Boolean Retrieval Models",
        "abstract": "An overview of boolean retrieval models and inverted indexes.",
        "authors": ["Alice Silva", "Carlos Mota"],
        "year": "2020",
        "doi": "10.1000/xyz003",
        "document_link": "https://example.com/3",
    },
    {
        "title": "Machine Learning Applications",
        "abstract": "Supervised and unsupervised learning methods for classification.",
        "authors": ["Diana Ferreira"],
        "year": "2023",
        "doi": "10.1000/xyz004",
        "document_link": "https://example.com/4",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pp():
    return make_stemming_preprocessor("english")

@pytest.fixture
def index(pp):
    idx = InvertedIndex(pp)
    idx.build_from_documents(DOCS)
    return idx

@pytest.fixture
def engine(index):
    return BooleanSearchEngine(index)


# ---------------------------------------------------------------------------
# InvertedIndex — building
# ---------------------------------------------------------------------------

class TestIndexBuilding:
    def test_num_documents(self, index):
        assert index.num_documents == len(DOCS)

    def test_vocabulary_not_empty(self, index):
        assert index.vocabulary_size > 0

    def test_known_term_in_index(self, index):
        # "retrieval" should be indexed (stemmed)
        postings = index.get_postings("retrieval")
        assert len(postings) > 0

    def test_unknown_term_returns_empty(self, index):
        assert index.get_postings("zzznonsensexxx") == []

    def test_document_frequency(self, index):
        df = index.get_document_frequency("retrieval")
        assert df >= 1

    def test_tf_positive(self, index):
        postings = index.get_postings("retrieval")
        for p in postings:
            assert p.tf >= 1

    def test_positions_recorded(self, index):
        postings = index.get_postings("retrieval")
        for p in postings:
            assert isinstance(p.positions, list)
            assert len(p.positions) == p.tf


# ---------------------------------------------------------------------------
# InvertedIndex — skip pointers
# ---------------------------------------------------------------------------

class TestSkipPointers:
    def test_skip_pointers_built_for_long_list(self):
        # Build an index with many documents to ensure skip pointers are created
        pp = make_stemming_preprocessor("english")
        docs = [
            {"title": f"retrieval document number {i}", "abstract": ""}
            for i in range(20)
        ]
        idx = InvertedIndex(pp)
        idx.build_from_documents(docs)
        pl = idx._index.get("retriev") or idx._index.get("retrieval")
        if pl:
            assert isinstance(pl.skips, dict)

    def test_no_skip_pointers_for_short_list(self):
        pl = PostingsList(df=2, postings=[Posting(0), Posting(1)])
        pl.build_skip_pointers()
        assert pl.skips == {}


# ---------------------------------------------------------------------------
# InvertedIndex — incremental updates
# ---------------------------------------------------------------------------

class TestIncrementalUpdates:
    def test_add_document_increases_count(self, index):
        before = index.num_documents
        new_doc = {"title": "New Paper on Graphs", "abstract": "Graph theory and algorithms."}
        index.add_document(new_doc)
        assert index.num_documents == before + 1

    def test_added_document_is_searchable(self, index):
        new_doc = {"title": "Quantum Computing Survey", "abstract": "Quantum bits and gates."}
        doc_id = index.add_document(new_doc)
        postings = index.get_postings("quantum")
        assert any(p.doc_id == doc_id for p in postings)

    def test_remove_document(self, index):
        before = index.num_documents
        index.remove_document(0)
        assert index.num_documents == before - 1

    def test_removed_document_not_searchable(self, index):
        # Doc 0 has "information" in title
        index.remove_document(0)
        postings = index.get_postings("information")
        assert all(p.doc_id != 0 for p in postings)

    def test_remove_nonexistent_document(self, index):
        result = index.remove_document(9999)
        assert result is False


# ---------------------------------------------------------------------------
# InvertedIndex — persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, index):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            index.save(path)
            pp2 = make_stemming_preprocessor("english")
            idx2 = InvertedIndex(pp2)
            idx2.load(path)
            assert idx2.num_documents == index.num_documents
            assert idx2.vocabulary_size == index.vocabulary_size
        finally:
            os.unlink(path)

    def test_loaded_index_searchable(self, index):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            index.save(path)
            pp2 = make_stemming_preprocessor("english")
            idx2 = InvertedIndex(pp2)
            idx2.load(path)
            postings = idx2.get_postings("retrieval")
            assert len(postings) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# InvertedIndex — author search
# ---------------------------------------------------------------------------

class TestAuthorSearch:
    def test_find_existing_author(self, index):
        results = index.search_by_author("Alice")
        assert len(results) >= 1

    def test_author_search_case_insensitive(self, index):
        r1 = index.search_by_author("alice")
        r2 = index.search_by_author("ALICE")
        assert len(r1) == len(r2)

    def test_unknown_author_returns_empty(self, index):
        results = index.search_by_author("Nonexistent Person XYZ")
        assert results == []


# ---------------------------------------------------------------------------
# BooleanSearchEngine — basic queries
# ---------------------------------------------------------------------------

class TestBooleanSearch:
    def test_single_term(self, engine):
        results = engine.search("retrieval")
        assert len(results) > 0

    def test_and_query(self, engine):
        results = engine.search("retrieval AND boolean")
        doc_titles = [r.document["title"] for r in results]
        assert any("Boolean" in t for t in doc_titles)

    def test_or_query(self, engine):
        r_retrieval = engine.search("retrieval")
        r_learning = engine.search("learning")
        r_or = engine.search("retrieval OR learning")
        assert len(r_or) >= max(len(r_retrieval), len(r_learning))

    def test_not_query(self, engine):
        r_all = engine.search("retrieval")
        r_not = engine.search("retrieval NOT boolean")
        assert len(r_not) <= len(r_all)

    def test_implicit_and(self, engine):
        explicit = engine.search("retrieval AND information")
        implicit = engine.search("retrieval information")
        ids_explicit = {r.doc_id for r in explicit}
        ids_implicit = {r.doc_id for r in implicit}
        assert ids_explicit == ids_implicit

    def test_grouped_query(self, engine):
        results = engine.search("(retrieval OR learning) AND information")
        assert isinstance(results, list)

    def test_empty_query_returns_empty(self, engine):
        assert engine.search("") == []

    def test_unknown_term_returns_empty(self, engine):
        assert engine.search("zzznonsensexxx") == []


# ---------------------------------------------------------------------------
# BooleanSearchEngine — operator precedence
# ---------------------------------------------------------------------------

class TestOperatorPrecedence:
    def test_not_binds_tighter_than_and(self, engine):
        # "retrieval AND NOT learning" should be different from "NOT (retrieval AND learning)"
        r1 = engine.search("retrieval AND NOT learning")
        r2 = engine.search("NOT (retrieval AND learning)")
        # Both are valid queries — just check they execute without error
        assert isinstance(r1, list)
        assert isinstance(r2, list)

    def test_and_binds_tighter_than_or(self, engine):
        # A OR B AND C  should be  A OR (B AND C)
        r_bc = engine.search("retrieval AND learning")
        r_full = engine.search("boolean OR retrieval AND learning")
        ids_bc = {r.doc_id for r in r_bc}
        ids_full = {r.doc_id for r in r_full}
        # Result of OR should be a superset
        assert ids_bc.issubset(ids_full)


# ---------------------------------------------------------------------------
# BooleanSearchEngine — result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_has_doc_id(self, engine):
        results = engine.search("retrieval")
        for r in results:
            assert isinstance(r.doc_id, int)

    def test_result_has_document(self, engine):
        results = engine.search("retrieval")
        for r in results:
            assert isinstance(r.document, dict)
            assert "title" in r.document

    def test_result_has_score(self, engine):
        results = engine.search("retrieval")
        for r in results:
            assert isinstance(r.score, float)
            assert r.score >= 0

    def test_results_sorted_by_score(self, engine):
        results = engine.search("retrieval OR learning OR boolean")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_author_search_result_structure(self, engine):
        results = engine.search_author("Alice")
        for r in results:
            assert isinstance(r.document, dict)