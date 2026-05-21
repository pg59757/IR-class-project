"""
database.py — Camada de persistência SQLite para o motor de pesquisa IR.

### 1.2 Data Storage
- **REQ-B09**: Design database schema for documents, authors, and metadata
- **REQ-B10**: Store raw text content and processed versions
- **REQ-B11**: Maintain document-author relationships
- **REQ-B12**: Store indexing data structures efficiently
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- REQ-B09: tabelas para documentos, autores e metadados
-- REQ-B11: relação N:N entre documentos e autores

CREATE TABLE IF NOT EXISTS authors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_authors_name ON authors(full_name);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY,   -- doc_id usado em todo o motor
    title           TEXT NOT NULL,
    year            TEXT,
    doi             TEXT,
    document_link   TEXT,
    doc_type        TEXT,
    subject         TEXT,
    -- REQ-B10: conteúdo bruto e processado
    raw_content     TEXT,                  -- abstract original
    processed_text  TEXT,                  -- tokens após pré-processamento (JSON array)
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- REQ-B11: tabela de junção documentos ↔ autores
CREATE TABLE IF NOT EXISTS document_authors (
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    author_id    INTEGER NOT NULL REFERENCES authors(id)   ON DELETE CASCADE,
    role         TEXT    NOT NULL DEFAULT 'author',
    position     INTEGER NOT NULL DEFAULT 0,   -- ordem dos autores
    PRIMARY KEY (document_id, author_id)
);

-- REQ-B09: metadados flexíveis por documento (pares chave/valor)
CREATE TABLE IF NOT EXISTS document_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT,
    data_type   TEXT NOT NULL DEFAULT 'string'   -- string | json | int | float
);

CREATE INDEX IF NOT EXISTS idx_metadata_doc   ON document_metadata(document_id);
CREATE INDEX IF NOT EXISTS idx_metadata_key   ON document_metadata(key);

-- REQ-B10: histórico de versões de conteúdo
CREATE TABLE IF NOT EXISTS document_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number  INTEGER NOT NULL,
    raw_content     TEXT,
    processed_text  TEXT,
    change_summary  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_versions_doc ON document_versions(document_id);

-- REQ-B12: índice invertido persistente
CREATE TABLE IF NOT EXISTS index_terms (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    term  TEXT NOT NULL UNIQUE,
    df    INTEGER NOT NULL DEFAULT 0   -- document frequency
);

CREATE INDEX IF NOT EXISTS idx_terms_term ON index_terms(term);

-- REQ-B12: postings list (uma linha por (termo, documento))
CREATE TABLE IF NOT EXISTS index_postings (
    term_id     INTEGER NOT NULL REFERENCES index_terms(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id)   ON DELETE CASCADE,
    tf          INTEGER NOT NULL DEFAULT 0,    -- term frequency no documento
    positions   TEXT,                          -- JSON array de posições no texto
    PRIMARY KEY (term_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON index_postings(term_id);
CREATE INDEX IF NOT EXISTS idx_postings_doc  ON index_postings(document_id);

-- REQ-B12: pesos TF-IDF pré-computados por (termo, documento)
CREATE TABLE IF NOT EXISTS tfidf_weights (
    term_id     INTEGER NOT NULL REFERENCES index_terms(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id)   ON DELETE CASCADE,
    tf_raw      REAL NOT NULL DEFAULT 0.0,
    tf_log      REAL NOT NULL DEFAULT 0.0,
    idf         REAL NOT NULL DEFAULT 0.0,
    tfidf       REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (term_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_tfidf_term ON tfidf_weights(term_id);
CREATE INDEX IF NOT EXISTS idx_tfidf_doc  ON tfidf_weights(document_id);

-- Tabela de auditoria / log de operações
CREATE TABLE IF NOT EXISTS operation_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    operation  TEXT NOT NULL,
    details    TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------

class DocumentStore:
    """
    Camada de acesso à base de dados SQLite para o motor de pesquisa.

    Parâmetros
    ----------
    db_path : str | Path
        Caminho para o ficheiro SQLite.  Use ":memory:" para testes.
    """

    def __init__(self, db_path: str | Path = "data/repositorium.db"):
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        logger.info("DocumentStore inicializado: %s", self.db_path)

    # ------------------------------------------------------------------
    # Ligação
    # ------------------------------------------------------------------

    def connect(self) -> "DocumentStore":
        """Abre a ligação à base de dados (idempotente)."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self

    def close(self):
        """Fecha a ligação, se estiver aberta."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Gestor de contexto que garante commit ou rollback."""
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Cria todas as tabelas se ainda não existirem (REQ-B09)."""
        if self._conn is None:
            self.connect()
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("Schema inicializado com sucesso.")

    # ------------------------------------------------------------------
    # REQ-B09 / REQ-B10 / REQ-B11 — Guardar documentos
    # ------------------------------------------------------------------

    def save_documents(
        self,
        documents: list[dict],
        processed_texts: Optional[dict[int, list[str]]] = None,
    ) -> int:
        """
        Persiste uma lista de documentos (formato do scraper).

        Parâmetros
        ----------
        documents       : lista de dicts com chaves title, abstract, authors, …
        processed_texts : {doc_id: [token, …]} — tokens pós-pré-processamento

        Retorna
        -------
        Número de documentos inseridos/actualizados.
        """
        t0 = time.monotonic()
        count = 0

        with self._cursor() as cur:
            for doc_id, doc in enumerate(documents):
                raw_content = doc.get("abstract", "") or ""
                processed = processed_texts.get(doc_id, []) if processed_texts else []

                # Normalizar subject
                subject = doc.get("subject", doc.get("area", doc.get("keywords", "")))
                if isinstance(subject, list):
                    subject = "; ".join(subject)

                # REQ-B09 / REQ-B10: inserir ou atualizar documento
                cur.execute(
                    """
                    INSERT INTO documents
                        (id, title, year, doi, document_link, doc_type,
                         subject, raw_content, processed_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title          = excluded.title,
                        year           = excluded.year,
                        doi            = excluded.doi,
                        document_link  = excluded.document_link,
                        doc_type       = excluded.doc_type,
                        subject        = excluded.subject,
                        raw_content    = excluded.raw_content,
                        processed_text = excluded.processed_text,
                        version        = documents.version + 1,
                        updated_at     = datetime('now')
                    """,
                    (
                        doc_id,
                        doc.get("title", "N/A"),
                        doc.get("year", ""),
                        doc.get("doi", ""),
                        doc.get("document_link", ""),
                        doc.get("doc_type", doc.get("type", "")),
                        subject,
                        raw_content,
                        json.dumps(processed, ensure_ascii=False),
                    ),
                )

                # REQ-B11: autores
                authors = doc.get("authors", [])
                if isinstance(authors, str):
                    authors = [authors]
                self._upsert_authors(cur, doc_id, authors)

                # REQ-B09: metadados extra
                self._upsert_metadata(cur, doc_id, doc)

                count += 1

        duration_ms = int((time.monotonic() - t0) * 1000)
        self._log_operation("save_documents", f"{count} docs", duration_ms)
        logger.info("Guardados %d documentos em %d ms.", count, duration_ms)
        return count

    def _upsert_authors(
        self, cur: sqlite3.Cursor, doc_id: int, authors: list[str]
    ) -> None:
        """Insere autores e cria as ligações documento ↔ autor (REQ-B11)."""
        # Apagar ligações anteriores para este documento
        cur.execute("DELETE FROM document_authors WHERE document_id = ?", (doc_id,))

        for position, name in enumerate(authors):
            name = name.strip()
            if not name:
                continue

            # Inserir autor se ainda não existir
            cur.execute(
                "INSERT OR IGNORE INTO authors (full_name) VALUES (?)", (name,)
            )
            cur.execute("SELECT id FROM authors WHERE full_name = ?", (name,))
            author_id = cur.fetchone()["id"]

            # Ligação documento ↔ autor com posição
            cur.execute(
                """
                INSERT OR REPLACE INTO document_authors
                    (document_id, author_id, role, position)
                VALUES (?, ?, 'author', ?)
                """,
                (doc_id, author_id, position),
            )

    def _upsert_metadata(
        self, cur: sqlite3.Cursor, doc_id: int, doc: dict
    ) -> None:
        """Guarda campos extra como metadados genéricos (REQ-B09)."""
        # Campos já tratados na tabela principal — ignorar aqui
        skip = {"title", "year", "doi", "document_link", "doc_type",
                 "type", "abstract", "authors", "subject", "area", "keywords"}

        cur.execute("DELETE FROM document_metadata WHERE document_id = ?", (doc_id,))

        for key, value in doc.items():
            if key in skip or value is None:
                continue
            if isinstance(value, (dict, list)):
                data_type = "json"
                value_str = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, int):
                data_type = "int"
                value_str = str(value)
            elif isinstance(value, float):
                data_type = "float"
                value_str = str(value)
            else:
                data_type = "string"
                value_str = str(value)

            cur.execute(
                """
                INSERT INTO document_metadata (document_id, key, value, data_type)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, key, value_str, data_type),
            )

    # ------------------------------------------------------------------
    # REQ-B10 — Versões de conteúdo
    # ------------------------------------------------------------------

    def save_version(
        self,
        doc_id: int,
        raw_content: str,
        processed_tokens: list[str],
        change_summary: str = "",
    ) -> None:
        """Guarda uma nova versão do conteúdo de um documento (REQ-B10)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM document_versions WHERE document_id = ?",
                (doc_id,),
            )
            next_version = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO document_versions
                    (document_id, version_number, raw_content, processed_text, change_summary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    next_version,
                    raw_content,
                    json.dumps(processed_tokens, ensure_ascii=False),
                    change_summary,
                ),
            )

    # ------------------------------------------------------------------
    # REQ-B12 — Índice invertido
    # ------------------------------------------------------------------

    def save_inverted_index(self, inv_index) -> None:
        """
        Persiste o InvertedIndex na base de dados (REQ-B12).

        Parâmetros
        ----------
        inv_index : InvertedIndex  — objeto com atributo _index (dict)
        """
        t0 = time.monotonic()
        index_dict = inv_index._index  # {term: PostingsList}

        with self._cursor() as cur:
            # Limpar índice anterior
            cur.execute("DELETE FROM index_postings")
            cur.execute("DELETE FROM index_terms")

            for term, postings_list in index_dict.items():
                # Inserir termo
                cur.execute(
                    "INSERT INTO index_terms (term, df) VALUES (?, ?)",
                    (term, postings_list.df),
                )
                term_id = cur.lastrowid

                # Inserir postings
                for posting in postings_list.postings:
                    cur.execute(
                        """
                        INSERT INTO index_postings (term_id, document_id, tf, positions)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            term_id,
                            posting.doc_id,
                            posting.tf,
                            json.dumps(posting.positions),
                        ),
                    )

        duration_ms = int((time.monotonic() - t0) * 1000)
        self._log_operation(
            "save_inverted_index",
            f"{len(index_dict)} termos",
            duration_ms,
        )
        logger.info(
            "Índice invertido guardado: %d termos em %d ms.",
            len(index_dict),
            duration_ms,
        )

    def save_tfidf_weights(self, tfidf_engine) -> None:
        """
        Persiste os pesos TF-IDF pré-computados (REQ-B12).

        Parâmetros
        ----------
        tfidf_engine : TFIDFEngine (use_sklearn=False)
        """
        import math as _math

        t0 = time.monotonic()
        documents = tfidf_engine._documents
        vocab = tfidf_engine._vocabulary
        idf_values = tfidf_engine._idf  # {term: idf}
        count = 0

        with self._cursor() as cur:
            cur.execute("DELETE FROM tfidf_weights")

            for term, term_idx in vocab.items():
                idf = idf_values.get(term, 0.0)

                # Obter term_id da tabela index_terms
                cur.execute("SELECT id FROM index_terms WHERE term = ?", (term,))
                row = cur.fetchone()
                if row is None:
                    continue
                term_id = row["id"]

                for doc_id, doc_tokens in enumerate(
                    tfidf_engine._document_tokens
                ):
                    tf_raw = doc_tokens.count(term) if isinstance(doc_tokens, list) else 0
                    if tf_raw == 0:
                        continue

                    tf_log = 1 + _math.log(tf_raw) if tf_raw > 0 else 0.0
                    tfidf = tf_log * idf

                    cur.execute(
                        """
                        INSERT OR REPLACE INTO tfidf_weights
                            (term_id, document_id, tf_raw, tf_log, idf, tfidf)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (term_id, doc_id, float(tf_raw), tf_log, idf, tfidf),
                    )
                    count += 1

        duration_ms = int((time.monotonic() - t0) * 1000)
        self._log_operation("save_tfidf_weights", f"{count} pesos", duration_ms)
        logger.info("Pesos TF-IDF guardados: %d entradas em %d ms.", count, duration_ms)

    # ------------------------------------------------------------------
    # Leituras — documentos
    # ------------------------------------------------------------------

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Devolve um documento pelo seu ID interno."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_doc(cur, dict(row))

    def get_all_documents(self) -> list[dict]:
        """Devolve todos os documentos com os respetivos autores."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents ORDER BY id")
            rows = cur.fetchall()
            return [self._row_to_doc(cur, dict(r)) for r in rows]

    def get_documents_by_author(self, name_fragment: str) -> list[dict]:
        """
        Devolve documentos cujo autor contém *name_fragment*
        (pesquisa parcial, sem distinção de maiúsculas) — REQ-B11.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT d.*
                FROM documents d
                JOIN document_authors da ON da.document_id = d.id
                JOIN authors a ON a.id = da.author_id
                WHERE a.full_name LIKE ? COLLATE NOCASE
                ORDER BY d.id
                """,
                (f"%{name_fragment}%",),
            )
            rows = cur.fetchall()
            return [self._row_to_doc(cur, dict(r)) for r in rows]

    def _row_to_doc(self, cur: sqlite3.Cursor, row: dict) -> dict:
        """Enriquece uma linha da tabela documents com autores e metadados."""
        doc_id = row["id"]

        # Autores ordenados por posição (REQ-B11)
        cur.execute(
            """
            SELECT a.full_name
            FROM authors a
            JOIN document_authors da ON da.author_id = a.id
            WHERE da.document_id = ?
            ORDER BY da.position
            """,
            (doc_id,),
        )
        row["authors"] = [r["full_name"] for r in cur.fetchall()]

        # Tokens processados
        processed_raw = row.get("processed_text") or "[]"
        row["processed_tokens"] = json.loads(processed_raw)
        del row["processed_text"]

        return row

    # ------------------------------------------------------------------
    # Leituras — índice (REQ-B12)
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> Optional[dict]:
        """
        Devolve a postings list de um termo.

        Retorna
        -------
        Dict com keys: term, df, postings=[{doc_id, tf, positions}]
        ou None se o termo não existir.
        """
        with self._cursor() as cur:
            cur.execute("SELECT id, df FROM index_terms WHERE term = ?", (term,))
            term_row = cur.fetchone()
            if term_row is None:
                return None

            cur.execute(
                "SELECT document_id, tf, positions FROM index_postings WHERE term_id = ? ORDER BY document_id",
                (term_row["id"],),
            )
            postings = [
                {
                    "doc_id": r["document_id"],
                    "tf": r["tf"],
                    "positions": json.loads(r["positions"] or "[]"),
                }
                for r in cur.fetchall()
            ]
            return {"term": term, "df": term_row["df"], "postings": postings}

    def get_vocabulary(self) -> list[str]:
        """Devolve o vocabulário completo do índice (REQ-B12)."""
        with self._cursor() as cur:
            cur.execute("SELECT term FROM index_terms ORDER BY term")
            return [r["term"] for r in cur.fetchall()]

    def get_top_tfidf_terms(
        self, doc_id: int, top_k: int = 10
    ) -> list[dict]:
        """
        Devolve os top-k termos com maior peso TF-IDF para um documento (REQ-B12).
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT t.term, w.tf_raw, w.tf_log, w.idf, w.tfidf
                FROM tfidf_weights w
                JOIN index_terms t ON t.id = w.term_id
                WHERE w.document_id = ?
                ORDER BY w.tfidf DESC
                LIMIT ?
                """,
                (doc_id, top_k),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Estatísticas
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Devolve estatísticas gerais da base de dados."""
        with self._cursor() as cur:
            stats = {}
            for table in ("documents", "authors", "index_terms", "index_postings", "tfidf_weights"):
                cur.execute(f"SELECT COUNT(*) as n FROM {table}")
                stats[table] = cur.fetchone()["n"]

            cur.execute(
                "SELECT COUNT(DISTINCT document_id) as n FROM document_authors"
            )
            stats["documents_with_authors"] = cur.fetchone()["n"]
            return stats

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _log_operation(
        self, operation: str, details: str = "", duration_ms: int = 0
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO operation_log (operation, details, duration_ms) VALUES (?, ?, ?)",
                (operation, details, duration_ms),
            )
            self._conn.commit()
        except Exception:
            pass  # log nunca deve interromper o fluxo principal

    def export_to_json(self, output_path: str | Path) -> None:
        """Exporta todos os documentos para um ficheiro JSON (compatível com o scraper)."""
        docs = self.get_all_documents()
        out = []
        for d in docs:
            out.append({
                "title": d.get("title", ""),
                "year": d.get("year", ""),
                "doi": d.get("doi", ""),
                "abstract": d.get("raw_content", ""),
                "authors": d.get("authors", []),
                "document_link": d.get("document_link", ""),
                "doc_type": d.get("doc_type", ""),
                "subject": d.get("subject", ""),
            })
        Path(output_path).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Exportados %d documentos para %s.", len(out), output_path)