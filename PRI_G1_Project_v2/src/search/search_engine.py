"""
search_engine.py — Ponto de entrada unificado do motor de Recuperação de Informação.

Orquestra os quatro módulos principais:
    - text_pipeline       → pré-processamento (tokenização, stopwords, stemming, lematização)
    - term_document_matrix → modelo booleano sobre a matriz termo-documento
    - inverted_index      → índice invertido com skip pointers
    - tfidf               → TF-IDF custom + sklearn com similaridade do cosseno

Uso rápido
----------
    from search_engine import SearchEngine

    engine = SearchEngine(language="english", tfidf_backend="custom")
    engine.build(documents)          # documents = lista de dicts do scraper

    # Pesquisa booleana (modelo de matriz termo-documento)
    results = engine.boolean_search("neural AND network NOT survey")

    # Pesquisa por relevância (TF-IDF + cosseno)
    results = engine.tfidf_search("deep learning image classification", top_k=10)

    # Pesquisa por autor (inverted index)
    results = engine.author_search("Silva")

    # Estatísticas de todos os módulos
    stats = engine.stats()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

# ── módulos locais ──────────────────────────────────────────────────────────
# ── módulos locais ──────────────────────────────────────────────────────────
from src.search.text_pipeline import (
    Preprocessor,
    PreprocessorConfig,
    make_stemming_preprocessor,
    make_lemmatisation_preprocessor,
    make_bare_preprocessor
)
from src.search.term_document_matrix import TermDocumentMatrix
from src.search.boolean_search import BooleanSearchEngine, SearchResult as BooleanResult
from src.search.inverted_index import InvertedIndex
from src.search.tfidf import TFIDFEngine, TFIDFResult, SimilarityResult
logger = logging.getLogger(__name__)

# ── tipos públicos ───────────────────────────────────────────────────────────
Language      = Literal["english", "portuguese"]
TFIDFBackend  = Literal["custom", "sklearn"]
TFScheme      = Literal["raw", "log", "boolean"]
PreprocessMode = Literal["stemming", "lemmatisation", "bare", "full"]


# ─────────────────────────────────────────────────────────────────────────────
# Resultado unificado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    """Resultado normalizado independente do motor de pesquisa usado."""
    doc_id:   int
    document: dict
    score:    float
    source:   str = ""   # "boolean" | "tfidf" | "author" | "inverted"

    # helpers de acesso rápido a campos comuns do scraper
    @property
    def title(self) -> str:
        return self.document.get("title", "N/A")

    @property
    def authors(self) -> list[str]:
        a = self.document.get("authors", [])
        return a if isinstance(a, list) else [a]

    @property
    def year(self) -> str:
        return self.document.get("year", "N/A")

    @property
    def abstract(self) -> str:
        return self.document.get("abstract", "N/A")

    @property
    def doi(self) -> str:
        return self.document.get("doi", "N/A")

    def __repr__(self) -> str:
        return f"Result(id={self.doc_id}, score={self.score:.4f}, title={self.title!r})"


# ─────────────────────────────────────────────────────────────────────────────
# SearchEngine — orquestrador
# ─────────────────────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Motor de Recuperação de Informação completo.

    Integra:
        • Pré-processamento NLP configurável (text_pipeline)
        • Modelo Booleano via matriz termo-documento (term_document_matrix + boolean_search)
        • Índice Invertido com skip pointers (inverted_index)
        • TF-IDF com backend custom ou sklearn (tfidf)

    Parâmetros
    ----------
    language : "english" | "portuguese"
        Idioma para stopwords e stemmer/lematizador.
    preprocess_mode : "stemming" | "lemmatisation" | "bare" | "full"
        Modo de pré-processamento.
        "full" aplica lematização seguida de stemming.
    tfidf_backend : "custom" | "sklearn"
        Implementação TF-IDF a usar.
    tf_scheme : "raw" | "log" | "boolean"
        Esquema de ponderação TF.
    tdm_mode : "binary" | "tf" | "tfidf"
        Modo da matriz termo-documento.
    fields : list[str]
        Campos do documento a indexar (default: ["title", "abstract"]).
    """

    def __init__(
        self,
        language:        Language       = "english",
        preprocess_mode: PreprocessMode = "stemming",
        tfidf_backend:   TFIDFBackend   = "custom",
        tf_scheme:       TFScheme       = "log",
        tdm_mode:        str            = "binary",
        fields:          Optional[list[str]] = None,
    ):
        self.language        = language
        self.preprocess_mode = preprocess_mode
        self.tfidf_backend   = tfidf_backend
        self.tf_scheme       = tf_scheme
        self.tdm_mode        = tdm_mode
        self.fields          = fields or ["title", "abstract"]

        # 3.2.1 — Pré-processamento
        self.preprocessor: Preprocessor = self._make_preprocessor(language, preprocess_mode)

        # 3.2.2 — Modelo Booleano (matriz termo-documento + motor booleano)
        self._tdm: TermDocumentMatrix       = TermDocumentMatrix(
            preprocessor=self.preprocessor,
            fields=self.fields,
            mode=tdm_mode,
        )
        self._boolean_engine: Optional[BooleanSearchEngine] = None

        # 3.2.3 — Índice Invertido
        self._inverted_index: InvertedIndex = InvertedIndex(
            preprocessor=self.preprocessor,
            fields=self.fields,
        )

        # 3.2.4 — TF-IDF
        self._tfidf: TFIDFEngine = TFIDFEngine(
            preprocessor=self.preprocessor,
            fields=self.fields,
            tf_scheme=tf_scheme,
            use_sklearn=(tfidf_backend == "sklearn"),
        )

        self._built = False
        logger.info(
            "SearchEngine criado — lang=%s | modo=%s | tfidf=%s | tdm=%s",
            language, preprocess_mode, tfidf_backend, tdm_mode,
        )

    # ── construção do índice ─────────────────────────────────────────────────

    def build(self, documents: list[dict]) -> "SearchEngine":
        """
        Constrói todos os índices a partir de uma lista de documentos.

        Cada documento deve ser um dict com campos compatíveis com o scraper:
            {"title": ..., "abstract": ..., "authors": [...], "year": ..., "doi": ...}

        Retorna self para permitir encadeamento:
            engine.build(docs).boolean_search("neural network")
        """
        if not documents:
            logger.warning("build() chamado com lista vazia.")
            return self

        logger.info("A construir índices para %d documentos …", len(documents))

        # Matriz termo-documento (modelo booleano)
        self._tdm.build_from_documents(documents)
        self._boolean_engine = BooleanSearchEngine(self._tdm)

        # Índice invertido
        self._inverted_index.build_from_documents(documents)

        # TF-IDF
        self._tfidf.build_from_documents(documents)

        self._built = True
        logger.info("Todos os índices construídos com sucesso.")
        return self

    def add_document(self, doc: dict) -> int:
        """
        Adiciona um documento incrementalmente ao índice invertido e TF-IDF.

        Nota: a matriz termo-documento não suporta adição incremental eficiente;
        chamar build() novamente é necessário para a atualizar.

        Retorna o doc_id atribuído.
        """
        doc_id = self._inverted_index.add_document(doc)
        logger.info("Documento adicionado ao índice invertido com id=%d", doc_id)
        return doc_id

    # ── pesquisa booleana ────────────────────────────────────────────────────

    def boolean_search(self, query: str) -> list[Result]:
        """
        Executa uma pesquisa booleana sobre a matriz termo-documento.

        Operadores suportados: AND, OR, NOT, parênteses, AND implícito.

        Exemplos:
            "neural network"              → neural AND network (implícito)
            "neural OR deep"              → OR explícito
            "(neural OR deep) AND learning NOT survey"
        """
        self._require_built()
        raw: list[BooleanResult] = self._boolean_engine.search(query)
        return [Result(doc_id=r.doc_id, document=r.document, score=r.score, source="boolean")
                for r in raw]

    # ── pesquisa por relevância ──────────────────────────────────────────────

    def tfidf_search(self, query: str, top_k: int = 20) -> list[Result]:
        """
        Executa uma pesquisa por relevância usando TF-IDF + similaridade do cosseno.

        O backend (custom ou sklearn) é definido no construtor.
        """
        self._require_built()
        raw: list[TFIDFResult] = self._tfidf.search(query, top_k=top_k)
        return [Result(doc_id=r.doc_id, document=r.document, score=r.score, source="tfidf")
                for r in raw]

    # ── pesquisa por autor ───────────────────────────────────────────────────

    def author_search(self, author_name: str) -> list[Result]:
        """
        Pesquisa documentos pelo nome do autor (substring, case-insensitive).

        Delega ao índice invertido (metadados não indexados na matriz).
        """
        self._require_built()
        docs = self._inverted_index.search_by_author(author_name)
        return [
            Result(doc_id=i, document=d, score=1.0, source="author")
            for i, d in enumerate(docs)
        ]

    # ── interseção de termos via índice invertido ────────────────────────────

    def intersect(self, term1: str, term2: str) -> list[dict]:
        """
        Interseção de postings dos dois termos usando skip pointers.

        Retorna os documentos que contêm ambos os termos.
        """
        self._require_built()
        postings = self._inverted_index.intersect(term1, term2)
        results = []
        for p in postings:
            doc = self._inverted_index.get_document(p.doc_id)
            if doc:
                results.append(doc)
        return results

    # ── similaridade entre documentos ───────────────────────────────────────

    def similar_to(self, doc_id: int, top_k: int = 10) -> list[Result]:
        """
        Retorna os top_k documentos mais similares ao documento dado,
        com base nos vetores TF-IDF e similaridade do cosseno.
        """
        self._require_built()
        raw: list[SimilarityResult] = self._tfidf.similar_to(doc_id, top_k=top_k)
        return [Result(doc_id=r.doc_id, document=r.document, score=r.similarity, source="similarity")
                for r in raw]

    def similarity_matrix(self):
        """
        Calcula a matriz N×N de similaridade do cosseno entre todos os documentos.

        Retorna (matrix: np.ndarray, doc_ids: list[int]).
        """
        self._require_built()
        return self._tfidf.similarity_matrix()

    # ── acesso a componentes internos ────────────────────────────────────────

    @property
    def tdm(self) -> TermDocumentMatrix:
        """Acesso direto à matriz termo-documento."""
        return self._tdm

    @property
    def boolean_engine(self) -> Optional[BooleanSearchEngine]:
        """Acesso direto ao motor de pesquisa booleana."""
        return self._boolean_engine

    @property
    def inverted_index(self) -> InvertedIndex:
        """Acesso direto ao índice invertido."""
        return self._inverted_index

    @property
    def tfidf_engine(self) -> TFIDFEngine:
        """Acesso direto ao motor TF-IDF."""
        return self._tfidf

    # ── estatísticas ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Devolve estatísticas consolidadas de todos os módulos.

        Útil para o painel educativo no frontend.
        """
        result = {
            "built": self._built,
            "language": self.language,
            "preprocess_mode": self.preprocess_mode,
            "tfidf_backend": self.tfidf_backend,
            "tf_scheme": self.tf_scheme,
        }
        if self._built:
            result["term_document_matrix"] = self._tdm.stats()
            result["inverted_index"]       = self._inverted_index.stats()
            result["tfidf"]                = self._tfidf.stats()
        return result

    # ── configuração dinâmica do backend TF-IDF ──────────────────────────────

    def set_tfidf_backend(self, backend: TFIDFBackend, documents: list[dict]) -> None:
        """
        Muda o backend TF-IDF (custom ↔ sklearn) e reconstrói o índice.

        Útil para a funcionalidade de comparação interativa no frontend.
        """
        self.tfidf_backend = backend
        self._tfidf = TFIDFEngine(
            preprocessor=self.preprocessor,
            fields=self.fields,
            tf_scheme=self.tf_scheme,
            use_sklearn=(backend == "sklearn"),
        )
        self._tfidf.build_from_documents(documents)
        logger.info("Backend TF-IDF alterado para '%s' e índice reconstruído.", backend)

    # ── helpers privados ─────────────────────────────────────────────────────

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "O motor ainda não foi inicializado. Chama engine.build(documents) primeiro."
            )

    @staticmethod
    def _make_preprocessor(language: Language, mode: PreprocessMode) -> Preprocessor:
        """Instancia o Preprocessor certo com base no modo escolhido."""
        if mode == "stemming":
            return make_stemming_preprocessor(language)
        if mode == "lemmatisation":
            return make_lemmatisation_preprocessor(language)
        if mode == "bare":
            return make_bare_preprocessor(language)
        # "full" — lematização + stemming
        return Preprocessor(PreprocessorConfig(
            language=language,
            use_stemming=True,
            use_lemmatisation=True,
            remove_stopwords=True,
            min_token_length=2,
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=False,
        ))

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"SearchEngine(lang={self.language!r}, mode={self.preprocess_mode!r}, "
            f"tfidf={self.tfidf_backend!r}, {status})"
        )