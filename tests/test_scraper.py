"""
tests/test_scraper.py — Unit tests for the UMinhoDSpace8Scraper.

Tests do NOT make real HTTP requests — they validate data-extraction logic,
area filters, and the result structure using the existing scraper_results.json
and synthetic document fixtures.

Run:
    pytest tests/test_scraper.py -v
"""

import json
import pytest
from pathlib import Path

_RESULTS_FILE = (
    Path(__file__).parent.parent / "src" / "scraper" / "scraper_results.json"
)


def _load_sample_results() -> list[dict]:
    if _RESULTS_FILE.exists():
        with open(_RESULTS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return []


SAMPLE_RESULTS = _load_sample_results()
HAS_RESULTS = len(SAMPLE_RESULTS) > 0


# ---------------------------------------------------------------------------
# Tests on scraped result structure (REQ-B02 to B05)
# ---------------------------------------------------------------------------

class TestScraperResultStructure:

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_results_is_list(self):
        assert isinstance(SAMPLE_RESULTS, list)

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_minimum_document_count(self):
        """REQ-B07: collection must contain at least some documents."""
        assert len(SAMPLE_RESULTS) >= 1

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_required_fields_present(self):
        """REQ-B02: title and authors must be present in every document."""
        for doc in SAMPLE_RESULTS:
            assert "title" in doc
            assert "authors" in doc

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_title_is_non_empty_string(self):
        for doc in SAMPLE_RESULTS:
            title = doc.get("title", "")
            assert isinstance(title, str)
            assert len(title.strip()) > 0

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_authors_is_list(self):
        """REQ-B05: authors must be a list."""
        for doc in SAMPLE_RESULTS:
            assert isinstance(doc.get("authors", []), list)

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_abstract_field_present(self):
        """REQ-B03: abstract field must exist (may be 'N/A')."""
        for doc in SAMPLE_RESULTS:
            assert "abstract" in doc

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_document_link_field(self):
        """REQ-B04: document_link must be a string."""
        for doc in SAMPLE_RESULTS:
            assert isinstance(doc.get("document_link", ""), str)

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_year_field_present(self):
        """REQ-B02: year field must be present."""
        for doc in SAMPLE_RESULTS:
            assert "year" in doc

    @pytest.mark.skipif(not HAS_RESULTS, reason="No scraper results available")
    def test_no_duplicate_titles(self):
        titles = [doc.get("title", "") for doc in SAMPLE_RESULTS]
        n_dupes = len(titles) - len(set(titles))
        assert n_dupes / len(titles) < 0.05, f"Too many duplicates: {n_dupes}"


# ---------------------------------------------------------------------------
# Tests on KNOWN_COLLECTIONS and area filter (REQ-B08)
# ---------------------------------------------------------------------------

class TestKnownCollections:

    def test_known_collections_imported(self):
        from src.scraper.scraper import KNOWN_COLLECTIONS
        assert isinstance(KNOWN_COLLECTIONS, dict) and len(KNOWN_COLLECTIONS) > 0

    def test_known_collections_have_required_keys(self):
        from src.scraper.scraper import KNOWN_COLLECTIONS
        for handle, info in KNOWN_COLLECTIONS.items():
            assert "name" in info
            assert "area" in info

    def test_known_collections_areas_are_valid(self):
        from src.scraper.scraper import KNOWN_COLLECTIONS
        valid_areas = {
            "computer_science", "health_medicine", "engineering",
            "social_sciences", "mathematics", "other",
        }
        for handle, info in KNOWN_COLLECTIONS.items():
            assert info["area"] in valid_areas


# ---------------------------------------------------------------------------
# Tests on scraper instantiation (REQ-B01, B07)
# ---------------------------------------------------------------------------

class TestScraperInstantiation:
    """
    Testes de instanciação usando unittest.mock para evitar lançar Chrome.
    O __init__ do scraper chama find_chrome_executable() e webdriver.Chrome(),
    por isso fazemos patch de ambos para que os testes corram sem browser.
    """

    def test_scraper_class_importable(self):
        from src.scraper.scraper import UMinhoDSpace8Scraper
        assert UMinhoDSpace8Scraper is not None

    def test_scraper_base_url_set(self):
        from src.scraper.scraper import BASE_REPO_URL
        assert BASE_REPO_URL.startswith("https://")

    def test_scraper_accepts_collections_list(self):
        """REQ-B01 / REQ-B07: instanciação com lista de coleções e max_items."""
        from unittest.mock import patch, MagicMock
        from src.scraper.scraper import UMinhoDSpace8Scraper
        with patch("src.scraper.scraper.find_chrome_executable", return_value="/fake/chrome"),              patch("src.scraper.scraper.webdriver.Chrome", return_value=MagicMock()):
            s = UMinhoDSpace8Scraper(collections=["1822/21293"], max_items=10)
            assert s is not None

    def test_scraper_default_max_items(self):
        """REQ-B07: max_items deve ser positivo por defeito."""
        from unittest.mock import patch, MagicMock
        from src.scraper.scraper import UMinhoDSpace8Scraper
        with patch("src.scraper.scraper.find_chrome_executable", return_value="/fake/chrome"),              patch("src.scraper.scraper.webdriver.Chrome", return_value=MagicMock()):
            s = UMinhoDSpace8Scraper(collections=["1822/21293"])
            assert hasattr(s, "max_items") and s.max_items > 0

    def test_scraper_area_filter_attribute(self):
        """REQ-B08: area_filter deve ser guardado em minúsculas."""
        from unittest.mock import patch, MagicMock
        from src.scraper.scraper import UMinhoDSpace8Scraper
        with patch("src.scraper.scraper.find_chrome_executable", return_value="/fake/chrome"),              patch("src.scraper.scraper.webdriver.Chrome", return_value=MagicMock()):
            s = UMinhoDSpace8Scraper(
                collections=["1822/21293"], max_items=10, area_filter="engineering"
            )
            assert hasattr(s, "area_filter") and s.area_filter == "engineering"

    def test_scraper_invalid_area_raises(self):
        """REQ-B08: area_filter inválida deve lançar ValueError."""
        from unittest.mock import patch, MagicMock
        from src.scraper.scraper import UMinhoDSpace8Scraper
        with patch("src.scraper.scraper.find_chrome_executable", return_value="/fake/chrome"),              patch("src.scraper.scraper.webdriver.Chrome", return_value=MagicMock()),              pytest.raises(ValueError):
            UMinhoDSpace8Scraper(collections=["1822/21293"], area_filter="nonexistent_area")

    def test_scraper_no_chrome_raises(self):
        """Sem Chrome disponível deve lançar FileNotFoundError."""
        from unittest.mock import patch
        from src.scraper.scraper import UMinhoDSpace8Scraper
        with patch("src.scraper.scraper.find_chrome_executable", return_value=None),              pytest.raises(FileNotFoundError):
            UMinhoDSpace8Scraper(collections=["1822/21293"])

    def test_affiliations_field_in_targets(self):
        """REQ-B05: o scraper deve extrair afiliações (dc.contributor.affiliation)."""
        from src.scraper.scraper import UMinhoDSpace8Scraper
        from unittest.mock import patch, MagicMock
        with patch("src.scraper.scraper.find_chrome_executable", return_value="/fake/chrome"),              patch("src.scraper.scraper.webdriver.Chrome", return_value=MagicMock()):
            s = UMinhoDSpace8Scraper(collections=["1822/21293"], max_items=5)
            assert "dc.contributor.affiliation" in s._targets
            assert s._targets["dc.contributor.affiliation"] == "affiliations"


# ---------------------------------------------------------------------------
# Synthetic document — integration smoke tests
# ---------------------------------------------------------------------------

class TestResultParsing:

    @pytest.fixture
    def doc(self):
        return {
            "title": "A Study on Information Retrieval",
            "authors": ["Ana Costa", "Bruno Lima"],
            "year": "2023",
            "abstract": "This paper presents a new approach to document retrieval.",
            "keywords": ["information retrieval", "indexing"],
            "doi": "10.1000/xyz",
            "document_link": "https://repositorium.uminho.pt/handle/1822/99999",
            "doc_type": "article",
            "subject": "computer_science",
        }

    def test_doc_passes_through_preprocessor(self, doc):
        from src.search.preprocessor import make_stemming_preprocessor
        pp = make_stemming_preprocessor("english")
        tokens = pp.process_document(doc, fields=["title", "abstract"])
        assert len(tokens) > 0

    def test_doc_indexable(self, doc):
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.inverted_index import InvertedIndex
        pp = make_stemming_preprocessor("english")
        idx = InvertedIndex(pp)
        idx.build_from_documents([doc])
        assert idx.num_documents == 1
        assert idx.vocabulary_size > 0

    def test_doc_searchable_via_tfidf(self, doc):
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.tfidf import TFIDFEngine
        pp = make_stemming_preprocessor("english")
        engine = TFIDFEngine(pp)
        engine.build_from_documents([doc])
        results = engine.search("information retrieval", top_k=5)
        assert len(results) == 1
        assert results[0].score > 0

    def test_author_search_works(self, doc):
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.inverted_index import InvertedIndex
        from src.search.boolean_search import BooleanSearchEngine
        pp = make_stemming_preprocessor("english")
        idx = InvertedIndex(pp)
        idx.build_from_documents([doc])
        engine = BooleanSearchEngine(idx)
        results = engine.search_author("Ana")
        assert len(results) == 1