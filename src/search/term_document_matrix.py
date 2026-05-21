"""
term_document_matrix.py — Term-Document Matrix for the IR search engine.

The term-document matrix is a core data structure in Information Retrieval.
Each row represents a term from the vocabulary, each column a document.
The cell value can be:
    - Binary  (1 if term present, 0 otherwise)          → mode="binary"
    - TF      (raw term frequency)                       → mode="tf"
    - TF-IDF  (TF-IDF weight)                           → mode="tfidf"

Boolean retrieval is performed directly on the binary matrix by applying
bitwise AND / OR / NOT operations on term vectors.

Structure
---------
    vocabulary : list[str]       — sorted list of unique terms
    doc_ids    : list[int]       — document IDs (column order)
    matrix     : np.ndarray      — shape (|vocabulary|, |documents|)

Quick start
-----------
    from src.search.term_document_matrix import TermDocumentMatrix
    from src.search.preprocessor import make_stemming_preprocessor

    pp = make_stemming_preprocessor("english")
    tdm = TermDocumentMatrix(pp)
    tdm.build_from_documents(documents)

    # Boolean search directly on the matrix
    results = tdm.boolean_search("information AND retrieval NOT neural")

    # Inspect the matrix
    print(tdm.stats())
"""

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from src.search.preprocessor import Preprocessor, make_stemming_preprocessor

logger = logging.getLogger(__name__)

MatrixMode = Literal["binary", "tf", "tfidf"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TDMResult:
    """A single result from a term-document matrix boolean search."""
    doc_id: int
    document: dict
    score: float  # number of query terms matched (for ranking)


# ---------------------------------------------------------------------------
# Term-Document Matrix
# ---------------------------------------------------------------------------

class TermDocumentMatrix:
    """
    Term-Document Matrix supporting binary, TF and TF-IDF weightings.

    Boolean retrieval operates on the binary matrix using bitwise operations,
    which is conceptually simpler (and more memory-intensive) than the
    inverted index approach — making it ideal for educational comparison.

    Parameters
    ----------
    preprocessor : Preprocessor
        Text preprocessor to use for tokenisation, stemming, etc.
    fields : list[str]
        Document fields to index. Defaults to ["title", "abstract"].
    mode : "binary" | "tf" | "tfidf"
        Cell weighting scheme.
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        fields: Optional[list[str]] = None,
        mode: MatrixMode = "binary",
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.fields = fields or ["title", "abstract"]
        self.mode = mode

        # Core data
        self.vocabulary: list[str] = []          # row labels
        self.doc_ids: list[int] = []             # column labels
        self._documents: dict[int, dict] = {}    # id → original doc
        self.matrix: Optional[np.ndarray] = None # shape: (|vocab|, |docs|)

        # Internal lookup maps
        self._term_to_row: dict[str, int] = {}
        self._doc_id_to_col: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[dict]) -> None:
        """
        Build the term-document matrix from a list of document dicts.

        Steps:
            1. Tokenise every document with the preprocessor.
            2. Collect the global vocabulary (sorted for reproducibility).
            3. Allocate the matrix (|vocab| × |docs|).
            4. Fill each cell according to the chosen mode.

        Args:
            documents: List of publication dicts (as produced by the scraper).
        """
        logger.info("Building term-document matrix (%s) for %d documents …", self.mode, len(documents))

        # --- Step 1: tokenise all documents ---
        token_lists: dict[int, list[str]] = {}
        for idx, doc in enumerate(documents):
            self._documents[idx] = doc
            tokens = self.preprocessor.process_document(doc, self.fields)
            token_lists[idx] = tokens

        # --- Step 2: build vocabulary ---
        vocab_set: set[str] = set()
        for tokens in token_lists.values():
            vocab_set.update(tokens)

        self.vocabulary = sorted(vocab_set)
        self.doc_ids = list(token_lists.keys())
        self._term_to_row = {term: i for i, term in enumerate(self.vocabulary)}
        self._doc_id_to_col = {doc_id: j for j, doc_id in enumerate(self.doc_ids)}

        n_terms = len(self.vocabulary)
        n_docs = len(self.doc_ids)

        # --- Step 3: allocate matrix ---
        self.matrix = np.zeros((n_terms, n_docs), dtype=np.float32)

        # --- Step 4: fill matrix ---
        # First pass: compute raw TF for all cells
        for doc_id, tokens in token_lists.items():
            col = self._doc_id_to_col[doc_id]
            tf_counts: dict[str, int] = {}
            for token in tokens:
                if token in self._term_to_row:
                    tf_counts[token] = tf_counts.get(token, 0) + 1
            for term, count in tf_counts.items():
                row = self._term_to_row[term]
                self.matrix[row, col] = count

        # Second pass: apply weighting
        if self.mode == "binary":
            self.matrix = (self.matrix > 0).astype(np.float32)
        elif self.mode == "tfidf":
            self.matrix = self._apply_tfidf(self.matrix, n_docs)
        # mode == "tf": already filled with raw counts

        logger.info(
            "Matrix built: %d terms × %d documents (mode=%s, size=%.1f KB)",
            n_terms, n_docs, self.mode,
            self.matrix.nbytes / 1024,
        )

    # ------------------------------------------------------------------
    # Boolean search on the matrix
    # ------------------------------------------------------------------

    def boolean_search(self, query: str) -> list[TDMResult]:
        """
        Execute a boolean query directly on the term-document matrix.

        Uses bitwise operations on binary row vectors — conceptually the
        simplest form of Boolean retrieval, as described in Manning et al.
        Chapter 1.

        Supports AND, OR, NOT with correct precedence (NOT > AND > OR)
        and implicit AND for space-separated terms.

        Args:
            query: Boolean query string.

        Returns:
            List of TDMResult sorted by score (number of matching terms).
        """
        if self.matrix is None or not query.strip():
            return []

        try:
            result_vector = self._evaluate_query(query)
        except Exception as exc:
            logger.error("Boolean query error: %s", exc)
            return []

        matched_cols = np.where(result_vector > 0)[0]

        # Score = number of distinct query terms that appear in the document
        query_terms = self._extract_query_terms(query)

        results = []
        for col in matched_cols:
            doc_id = self.doc_ids[col]
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            score = sum(
                1 for term in query_terms
                if term in self._term_to_row and self.matrix[self._term_to_row[term], col] > 0
            )
            results.append(TDMResult(doc_id=doc_id, document=doc, score=float(score)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_term_vector(self, term: str) -> Optional[np.ndarray]:
        """
        Return the row vector for a term (after preprocessing).

        Args:
            term: Raw term string (will be preprocessed).

        Returns:
            1-D numpy array of length |documents|, or None if not in vocab.
        """
        processed = self.preprocessor.process(term)
        if not processed:
            return None
        key = processed[0]
        row = self._term_to_row.get(key)
        if row is None:
            return None
        return self.matrix[row].copy()

    def get_document_vector(self, doc_id: int) -> Optional[np.ndarray]:
        """
        Return the column vector for a document.

        Args:
            doc_id: Internal document identifier.

        Returns:
            1-D numpy array of length |vocabulary|, or None if not found.
        """
        col = self._doc_id_to_col.get(doc_id)
        if col is None:
            return None
        return self.matrix[:, col].copy()

    # ------------------------------------------------------------------
    # Stats & inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary of the matrix dimensions and sparsity."""
        if self.matrix is None:
            return {"built": False}

        n_terms, n_docs = self.matrix.shape
        non_zero = int(np.count_nonzero(self.matrix))
        total = n_terms * n_docs
        sparsity = 1.0 - non_zero / total if total > 0 else 1.0

        top_terms = sorted(
            self._term_to_row.items(),
            key=lambda kv: int(np.count_nonzero(self.matrix[kv[1]])),
            reverse=True,
        )[:10]

        return {
            "mode": self.mode,
            "num_terms": n_terms,
            "num_documents": n_docs,
            "non_zero_cells": non_zero,
            "sparsity": round(sparsity, 4),
            "matrix_size_kb": round(self.matrix.nbytes / 1024, 2),
            "top_terms_by_doc_freq": [(t, int(np.count_nonzero(self.matrix[r]))) for t, r in top_terms],
        }

    def to_dict(self) -> dict:
        """
        Serialise the matrix to a JSON-friendly dict (for educational display).

        Warning: only suitable for small matrices (demo / debugging).
        """
        if self.matrix is None:
            return {}
        return {
            "vocabulary": self.vocabulary,
            "doc_ids": self.doc_ids,
            "matrix": self.matrix.tolist(),
        }

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape as (n_terms, n_docs)."""
        if self.matrix is None:
            return (0, 0)
        return self.matrix.shape  # type: ignore

    def __repr__(self) -> str:
        if self.matrix is None:
            return "TermDocumentMatrix(not built)"
        n_terms, n_docs = self.matrix.shape
        return f"TermDocumentMatrix(mode={self.mode!r}, terms={n_terms}, docs={n_docs})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_tfidf(self, tf_matrix: np.ndarray, n_docs: int) -> np.ndarray:
        """Apply TF-IDF weighting to the raw TF matrix (log-normalised TF)."""
        result = np.zeros_like(tf_matrix)
        for row in range(tf_matrix.shape[0]):
            df = np.count_nonzero(tf_matrix[row])
            if df == 0:
                continue
            idf = math.log((n_docs + 1) / (df + 1)) + 1.0
            for col in range(tf_matrix.shape[1]):
                raw_tf = tf_matrix[row, col]
                if raw_tf > 0:
                    log_tf = 1.0 + math.log(raw_tf)
                    result[row, col] = log_tf * idf
        return result

    def _get_binary_vector(self, term: str) -> np.ndarray:
        """
        Return a binary presence/absence vector for a (pre-processed) term.
        Returns all-zeros if the term is not in the vocabulary.
        """
        n_docs = len(self.doc_ids)
        row = self._term_to_row.get(term)
        if row is None:
            return np.zeros(n_docs, dtype=np.float32)
        # Treat any non-zero value as present (works for all modes)
        return (self.matrix[row] > 0).astype(np.float32)

    def _all_ones(self) -> np.ndarray:
        return np.ones(len(self.doc_ids), dtype=np.float32)

    def _all_zeros(self) -> np.ndarray:
        return np.zeros(len(self.doc_ids), dtype=np.float32)

    # ------ recursive-descent boolean parser ------
    # Grammar (after implicit AND insertion):
    #   expr      → and_expr  (OR and_expr)*
    #   and_expr  → not_expr  (AND not_expr)*
    #   not_expr  → NOT not_expr | atom
    #   atom      → TERM | '(' expr ')'

    _OPS = {"AND", "OR", "NOT"}

    def _tokenise(self, query: str) -> list[str]:
        """Split query into operator/term tokens and insert implicit AND."""
        import re
        query = re.sub(r"\(", " ( ", query)
        query = re.sub(r"\)", " ) ", query)
        raw = query.split()

        tokens: list[str] = []
        for tok in raw:
            tokens.append(tok.upper() if tok.upper() in self._OPS or tok in ("(", ")") else tok)

        # Insert implicit AND between adjacent terms / RPAREN+term / term+LPAREN
        result: list[str] = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                curr = tok
                nxt = tokens[i + 1]
                curr_is_value = curr not in self._OPS and curr != "("
                nxt_is_value = nxt not in self._OPS and nxt != ")"
                if (curr_is_value or curr == ")") and (nxt_is_value or nxt in ("(", "NOT")):
                    result.append("AND")
        result.append("EOF")
        return result

    def _evaluate_query(self, query: str) -> np.ndarray:
        tokens = self._tokenise(query)
        self._tok_list = tokens
        self._tok_pos = 0
        return self._parse_or()

    def _cur(self) -> str:
        return self._tok_list[self._tok_pos]

    def _consume(self) -> str:
        tok = self._cur()
        self._tok_pos += 1
        return tok

    def _parse_or(self) -> np.ndarray:
        left = self._parse_and()
        while self._cur() == "OR":
            self._consume()
            right = self._parse_and()
            left = np.minimum(left + right, 1.0)  # binary OR
        return left

    def _parse_and(self) -> np.ndarray:
        left = self._parse_not()
        while self._cur() == "AND":
            self._consume()
            right = self._parse_not()
            left = left * right  # binary AND
        return left

    def _parse_not(self) -> np.ndarray:
        if self._cur() == "NOT":
            self._consume()
            operand = self._parse_not()
            return self._all_ones() - operand  # binary NOT
        return self._parse_atom()

    def _parse_atom(self) -> np.ndarray:
        tok = self._cur()
        if tok == "(":
            self._consume()
            result = self._parse_or()
            if self._cur() == ")":
                self._consume()
            return result
        if tok in ("EOF", ")"):
            return self._all_zeros()
        self._consume()
        # Preprocess the term before lookup
        processed = self.preprocessor.process(tok)
        if not processed:
            return self._all_zeros()
        return self._get_binary_vector(processed[0])

    def _extract_query_terms(self, query: str) -> list[str]:
        """Extract and preprocess all non-operator terms from a query string."""
        import re
        query = re.sub(r"[()]", " ", query)
        terms = []
        for tok in query.split():
            if tok.upper() in self._OPS:
                continue
            processed = self.preprocessor.process(tok)
            terms.extend(processed)
        return terms