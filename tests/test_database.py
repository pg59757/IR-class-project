"""
tests/test_database.py — Testes para o módulo de armazenamento (REQ-B09 a B12).

Executar:
    pytest tests/test_database.py -v
"""

import json
import pytest
from src.storage.database import DocumentStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "title": "Information Retrieval Fundamentals",
        "year": "2021",
        "doi": "10.1000/test.001",
        "abstract": "This paper covers the fundamentals of information retrieval systems.",
        "authors": ["Silva, João", "Santos, Maria"],
        "document_link": "https://example.com/doc1",
        "doc_type": "article",
        "subject": "computer_science",
    },
    {
        "title": "Machine Learning for Document Classification",
        "year": "2022",
        "doi": "10.1000/test.002",
        "abstract": "We present a machine learning approach to document classification.",
        "authors": ["Rodrigues, Ana", "Silva, João"],
        "document_link": "https://example.com/doc2",
        "doc_type": "thesis",
        "subject": "computer_science",
    },
    {
        "title": "Natural Language Processing in Portuguese",
        "year": "2020",
        "doi": "10.1000/test.003",
        "abstract": "Técnicas de processamento de linguagem natural aplicadas ao português.",
        "authors": ["Costa, Pedro"],
        "document_link": "",
        "doc_type": "article",
        "subject": "linguistics",
    },
]


@pytest.fixture
def store():
    """DocumentStore em memória (descartado após cada teste)."""
    s = DocumentStore(":memory:")
    s.init_schema()
    return s


@pytest.fixture
def populated_store(store):
    """DocumentStore com documentos de exemplo já guardados."""
    store.save_documents(SAMPLE_DOCUMENTS)
    return store


# ---------------------------------------------------------------------------
# REQ-B09: Esquema — tabelas criadas
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_tables_exist(self, store):
        """Verifica que todas as tabelas obrigatórias foram criadas."""
        expected = {
            "authors", "documents", "document_authors",
            "document_metadata", "document_versions",
            "index_terms", "index_postings", "tfidf_weights",
            "operation_log",
        }
        store._conn.row_factory = None
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {row[0] for row in cur.fetchall()}
        assert expected.issubset(existing), f"Tabelas em falta: {expected - existing}"

    def test_foreign_keys_enabled(self, store):
        """PRAGMA foreign_keys deve estar ON."""
        store._conn.row_factory = None
        cur = store._conn.execute("PRAGMA foreign_keys")
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# REQ-B09: Documentos e metadados
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_save_and_retrieve_document(self, store):
        """Guardar e recuperar um documento pelo ID."""
        store.save_documents([SAMPLE_DOCUMENTS[0]])
        doc = store.get_document(0)
        assert doc is not None
        assert doc["title"] == "Information Retrieval Fundamentals"
        assert doc["year"] == "2021"
        assert doc["doi"] == "10.1000/test.001"
        assert doc["doc_type"] == "article"

    def test_save_multiple_documents(self, populated_store):
        """Verificar contagem de documentos guardados."""
        all_docs = populated_store.get_all_documents()
        assert len(all_docs) == 3

    def test_document_not_found(self, store):
        """get_document com ID inexistente deve retornar None."""
        assert store.get_document(999) is None

    def test_document_raw_content_stored(self, store):
        """REQ-B10: conteúdo bruto deve ser guardado."""
        store.save_documents([SAMPLE_DOCUMENTS[0]])
        doc = store.get_document(0)
        assert "fundamentals of information retrieval" in doc["raw_content"]

    def test_metadata_saved(self, store):
        """REQ-B09: campos extra devem ser guardados como metadados."""
        doc_with_extra = {**SAMPLE_DOCUMENTS[0], "language": "en", "pages": 12}
        store.save_documents([doc_with_extra])
        with store._cursor() as cur:
            cur.execute(
                "SELECT key, value FROM document_metadata WHERE document_id = 0"
            )
            meta = {r["key"]: r["value"] for r in cur.fetchall()}
        assert "language" in meta
        assert meta["language"] == "en"


# ---------------------------------------------------------------------------
# REQ-B10: Conteúdo processado e versões
# ---------------------------------------------------------------------------

class TestProcessedContent:
    def test_processed_tokens_stored(self, store):
        """REQ-B10: tokens pré-processados devem ser guardados."""
        processed = {0: ["inform", "retriev", "fundament"]}
        store.save_documents([SAMPLE_DOCUMENTS[0]], processed_texts=processed)
        doc = store.get_document(0)
        assert doc["processed_tokens"] == ["inform", "retriev", "fundament"]

    def test_save_version(self, store):
        """REQ-B10: guardar uma versão histórica de um documento."""
        store.save_documents([SAMPLE_DOCUMENTS[0]])
        store.save_version(
            doc_id=0,
            raw_content="Versão anterior do abstract.",
            processed_tokens=["versão", "anterior"],
            change_summary="Atualização do abstract",
        )
        with store._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM document_versions WHERE document_id = 0"
            )
            assert cur.fetchone()["n"] == 1

    def test_version_number_increments(self, store):
        """REQ-B10: número de versão deve incrementar automaticamente."""
        store.save_documents([SAMPLE_DOCUMENTS[0]])
        store.save_version(0, "v1", [], "primeira")
        store.save_version(0, "v2", [], "segunda")
        with store._cursor() as cur:
            cur.execute(
                "SELECT version_number FROM document_versions WHERE document_id = 0 ORDER BY id"
            )
            versions = [r["version_number"] for r in cur.fetchall()]
        assert versions == [1, 2]


# ---------------------------------------------------------------------------
# REQ-B11: Relações documentos ↔ autores
# ---------------------------------------------------------------------------

class TestAuthors:
    def test_authors_linked_to_document(self, populated_store):
        """REQ-B11: autores devem estar associados ao documento correto."""
        doc = populated_store.get_document(0)
        assert "Silva, João" in doc["authors"]
        assert "Santos, Maria" in doc["authors"]

    def test_author_order_preserved(self, populated_store):
        """REQ-B11: a ordem dos autores deve ser preservada."""
        doc = populated_store.get_document(0)
        assert doc["authors"][0] == "Silva, João"
        assert doc["authors"][1] == "Santos, Maria"

    def test_shared_author_across_documents(self, populated_store):
        """REQ-B11: um autor pode estar em múltiplos documentos."""
        with populated_store._cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT da.document_id) as n
                FROM document_authors da
                JOIN authors a ON a.id = da.author_id
                WHERE a.full_name = 'Silva, João'
                """
            )
            count = cur.fetchone()["n"]
        assert count == 2  # aparece em doc 0 e doc 1

    def test_search_by_author(self, populated_store):
        """REQ-B11: pesquisa por autor deve devolver documentos corretos."""
        results = populated_store.get_documents_by_author("Rodrigues")
        assert len(results) == 1
        assert results[0]["title"] == "Machine Learning for Document Classification"

    def test_search_by_author_partial(self, populated_store):
        """REQ-B11: pesquisa parcial no nome do autor."""
        results = populated_store.get_documents_by_author("Silva")
        assert len(results) == 2

    def test_author_deduplicated(self, populated_store):
        """REQ-B11: o mesmo autor não deve ser inserido duas vezes na tabela authors."""
        with populated_store._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM authors WHERE full_name = 'Silva, João'"
            )
            assert cur.fetchone()["n"] == 1


# ---------------------------------------------------------------------------
# REQ-B12: Índice invertido
# ---------------------------------------------------------------------------

class TestInvertedIndex:
    def _build_index(self, documents):
        """Constrói um InvertedIndex mínimo para testes."""
        from src.search.preprocessor import make_stemming_preprocessor
        from src.search.inverted_index import InvertedIndex
        pp = make_stemming_preprocessor("english")
        idx = InvertedIndex(pp)
        idx.build_from_documents(documents)
        return idx

    def test_save_and_retrieve_postings(self, store):
        """REQ-B12: postings list de um termo deve ser recuperável."""
        store.save_documents(SAMPLE_DOCUMENTS)
        idx = self._build_index(SAMPLE_DOCUMENTS)
        store.save_inverted_index(idx)

        # "retriev" é o stem de "retrieval"
        result = store.get_postings("retriev")
        assert result is not None
        assert result["term"] == "retriev"
        assert result["df"] >= 1
        assert len(result["postings"]) >= 1
        assert result["postings"][0]["doc_id"] == 0

    def test_vocabulary_stored(self, store):
        """REQ-B12: vocabulário deve estar persistido."""
        store.save_documents(SAMPLE_DOCUMENTS)
        idx = self._build_index(SAMPLE_DOCUMENTS)
        store.save_inverted_index(idx)

        vocab = store.get_vocabulary()
        assert len(vocab) > 0
        assert "retriev" in vocab  # stem de retrieval

    def test_unknown_term_returns_none(self, store):
        """REQ-B12: termo inexistente deve retornar None."""
        store.save_documents(SAMPLE_DOCUMENTS)
        idx = self._build_index(SAMPLE_DOCUMENTS)
        store.save_inverted_index(idx)

        result = store.get_postings("xyznonexistent123")
        assert result is None

    def test_posting_tf_correct(self, store):
        """REQ-B12: term frequency na posting deve ser >= 1."""
        store.save_documents(SAMPLE_DOCUMENTS)
        idx = self._build_index(SAMPLE_DOCUMENTS)
        store.save_inverted_index(idx)

        result = store.get_postings("retriev")
        assert result["postings"][0]["tf"] >= 1


# ---------------------------------------------------------------------------
# Estatísticas
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_counts(self, populated_store):
        """get_stats deve devolver contagens para as tabelas principais."""
        stats = populated_store.get_stats()
        assert stats["documents"] == 3
        assert stats["authors"] >= 4  # 4 autores únicos
        assert "index_terms" in stats

    def test_export_to_json(self, populated_store, tmp_path):
        """export_to_json deve criar um ficheiro JSON válido."""
        out = tmp_path / "export.json"
        populated_store.export_to_json(out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 3
        assert data[0]["title"] == "Information Retrieval Fundamentals"
        assert isinstance(data[0]["authors"], list) 