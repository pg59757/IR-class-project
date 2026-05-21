"""
tests/test_api.py — Integration tests for the FastAPI REST API.

Uses FastAPI's TestClient (no server needed).

Run with:
    pytest tests/test_api.py -v
"""

import json
import pytest
from fastapi.testclient import TestClient

# Patch the data path before importing the app
import src.api.fastapi_app as app_module

SAMPLE_DOCS = [
    {
        "title": "Information Retrieval Systems",
        "abstract": "Modern information retrieval and ranking algorithms.",
        "authors": ["Alice Silva"],
        "year": "2021",
        "doi": "10.1000/001",
        "document_link": "https://example.com/1",
    },
    {
        "title": "Deep Learning for NLP",
        "abstract": "Neural networks applied to natural language processing.",
        "authors": ["Bob Santos"],
        "year": "2022",
        "doi": "10.1000/002",
        "document_link": "https://example.com/2",
    },
    {
        "title": "Boolean Retrieval Models",
        "abstract": "Boolean retrieval with inverted indexes and postings.",
        "authors": ["Carlos Mota"],
        "year": "2020",
        "doi": "10.1000/003",
        "document_link": "https://example.com/3",
    },
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a TestClient with patched document loading."""
    tmp = tmp_path_factory.mktemp("data")
    data_file = tmp / "docs.json"
    data_file.write_text(json.dumps(SAMPLE_DOCS), encoding="utf-8")

    # Patch the data path
    original_path = app_module.DATA_PATH
    app_module.DATA_PATH = data_file

    with TestClient(app_module.app) as c:
        yield c

    app_module.DATA_PATH = original_path


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Free-text search
# ---------------------------------------------------------------------------

class TestSearchFreetext:
    def test_search_returns_200(self, client):
        resp = client.get("/search?q=retrieval")
        assert resp.status_code == 200

    def test_search_response_structure(self, client):
        resp = client.get("/search?q=retrieval")
        data = resp.json()
        assert "query" in data
        assert "total" in data
        assert "results" in data

    def test_search_returns_results(self, client):
        resp = client.get("/search?q=retrieval")
        data = resp.json()
        assert data["total"] > 0
        assert len(data["results"]) > 0

    def test_result_has_required_fields(self, client):
        resp = client.get("/search?q=retrieval")
        result = resp.json()["results"][0]
        for field in ["doc_id", "title", "authors", "abstract", "year", "doi", "score"]:
            assert field in result, f"Missing field: {field}"

    def test_results_sorted_by_score(self, client):
        resp = client.get("/search?q=retrieval information")
        scores = [r["score"] for r in resp.json()["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, client):
        resp = client.get("/search?q=retrieval&top_k=1")
        assert len(resp.json()["results"]) <= 1

    def test_empty_query_rejected(self, client):
        resp = client.get("/search?q=zzznonsensexxx999")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_sklearn_algorithm(self, client):
        resp = client.get("/search?q=retrieval&algorithm=sklearn")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 0

    def test_year_filter(self, client):
        resp = client.get("/search?q=retrieval&year_from=2021&year_to=2022")
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            year = int(r["year"]) if r["year"].isdigit() else 0
            assert 2021 <= year <= 2022


# ---------------------------------------------------------------------------
# Boolean search
# ---------------------------------------------------------------------------

class TestSearchBoolean:
    def test_single_term(self, client):
        resp = client.get("/search/boolean?q=retrieval")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_and_query(self, client):
        resp = client.get("/search/boolean?q=retrieval AND boolean")
        assert resp.status_code == 200

    def test_or_query(self, client):
        resp = client.get("/search/boolean?q=retrieval OR learning")
        assert resp.status_code == 200
        # OR should return at least as many as a single term
        r_single = client.get("/search/boolean?q=retrieval").json()["total"]
        r_or = resp.json()["total"]
        assert r_or >= r_single

    def test_not_query(self, client):
        resp = client.get("/search/boolean?q=retrieval NOT boolean")
        assert resp.status_code == 200

    def test_empty_query(self, client):
        resp = client.get("/search/boolean?q=zzznonsense999")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# Author search
# ---------------------------------------------------------------------------

class TestSearchAuthor:
    def test_known_author(self, client):
        resp = client.get("/search/author?name=Alice")
        assert resp.status_code == 200
        assert resp.json()["total"] > 0

    def test_case_insensitive(self, client):
        r1 = client.get("/search/author?name=alice").json()["total"]
        r2 = client.get("/search/author?name=ALICE").json()["total"]
        assert r1 == r2

    def test_unknown_author(self, client):
        resp = client.get("/search/author?name=ZZZUnknownXYZ")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_partial_name_match(self, client):
        resp = client.get("/search/author?name=Sil")  # matches "Alice Silva"
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1


# ---------------------------------------------------------------------------
# Get document by ID
# ---------------------------------------------------------------------------

class TestGetDocument:
    def test_valid_id(self, client):
        resp = client.get("/documents/0")
        assert resp.status_code == 200
        doc = resp.json()
        assert "title" in doc
        assert doc["doc_id"] == 0

    def test_invalid_id_returns_404(self, client):
        resp = client.get("/documents/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_structure(self, client):
        data = resp = client.get("/stats").json()
        assert "num_documents" in data
        assert "vocabulary_size" in data
        assert data["num_documents"] == len(SAMPLE_DOCS)


# ---------------------------------------------------------------------------
# IDF endpoint
# ---------------------------------------------------------------------------

class TestIDF:
    def test_known_term(self, client):
        resp = client.get("/tfidf/idf?term=retrieval")
        assert resp.status_code == 200
        assert resp.json()["idf"] >= 0

    def test_unknown_term(self, client):
        resp = client.get("/tfidf/idf?term=zzznonsense999")
        assert resp.status_code == 200
        assert resp.json()["idf"] == 0.0