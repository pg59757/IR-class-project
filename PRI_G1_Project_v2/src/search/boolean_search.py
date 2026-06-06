"""
boolean_search.py — Boolean search engine built on top of TermDocumentMatrix.

Executes AND/OR/NOT with correct precedence (incl. implicit AND) using bitwise ops on the binary matrix.
Supports parentheses and returns results ranked by matched query terms.

"""

import logging
from dataclasses import dataclass

from src.search.term_document_matrix import TermDocumentMatrix, TDMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single result returned by the boolean search engine.
    """
    doc_id: int         #Internal document identifier
    document: dict      #Original document dict (from scraper)
    score: float = 0.0  #Relevance score — number of distinct query terms matched.

    @classmethod
    def from_tdm_result(cls, tdm_result: TDMResult) -> "SearchResult":
        """Convert a TDMResult into a SearchResult."""
        return cls(
            doc_id=tdm_result.doc_id,
            document=tdm_result.document,
            score=tdm_result.score,
        )


# ---------------------------------------------------------------------------
# Boolean Search Engine
# ---------------------------------------------------------------------------

class BooleanSearchEngine:
    """
    Boolean search engine backed by a TermDocumentMatrix.
    Executes AND/OR/NOT via bitwise ops on the binary matrix, using the matrix as the core retrieval structure.
    NOT > AND > OR.
    """

    def __init__(self, tdm: TermDocumentMatrix):
        if tdm.matrix is None:
            raise ValueError(
                "TermDocumentMatrix has not been built yet. "
                "Call tdm.build_from_documents(documents) first."
            )
        self.tdm = tdm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[SearchResult]:
        """
        Executes a boolean query on the term-document matrix and returns ranked results.
        Query parsing and bitwise AND/OR/NOT are handled by TermDocumentMatrix._evaluate_query().
        Supports implicit AND and parentheses. Returns SearchResult list sorted by matched terms.
        """
        query = query.strip()
        if not query:
            return []

        logger.info("Boolean search (TDM): '%s'", query)

        # Delegate boolean evaluation entirely to the TermDocumentMatrix.
        # boolean_search() internally calls _evaluate_query() which performs
        # bitwise operations on the binary matrix rows.
        tdm_results: list[TDMResult] = self.tdm.boolean_search(query)

        results = [SearchResult.from_tdm_result(r) for r in tdm_results]
        logger.info("Found %d results for query '%s'", len(results), query)
        return results

    def search_author(self, author_name: str) -> list[SearchResult]:
        """
        # Searches documents by author name (case‑insensitive substring match).
        # Performed on metadata rather than the matrix, since authors are not indexed.
        # Returns SearchResult objects with score = 1.0.
        """
        author_lower = author_name.lower()
        results = []
        for doc_id, doc in self.tdm._documents.items():
            authors = doc.get("authors", [])
            # authors can be a list or a comma-separated string
            if isinstance(authors, list):
                authors_str = " ".join(authors).lower()
            else:
                authors_str = str(authors).lower()

            if author_lower in authors_str:
                results.append(SearchResult(doc_id=doc_id, document=doc, score=1.0))

        return results

    # ------------------------------------------------------------------
    # Convenience / inspection helpers
    # ------------------------------------------------------------------

    def matrix_stats(self) -> dict:
        """
        Return statistics about the underlying term-document matrix.

        Delegates to TermDocumentMatrix.stats().
        Useful for the educational panel in the frontend.
        """
        return self.tdm.stats()

    def get_term_vector(self, term: str):
        """
        Returns the binary presence/absence vector for a term across documents.
        Delegates to TermDocumentMatrix.get_term_vector(). Returns a 1‑D array or None.
        """
        return self.tdm.get_term_vector(term)

    def get_document_vector(self, doc_id: int):
        """
        Returns the document’s column vector (all term weights).
        Delegates to TermDocumentMatrix.get_document_vector(). Returns a 1‑D array or None.
        """
        return self.tdm.get_document_vector(doc_id)

    def matrix_to_dict(self) -> dict:
        """
        Serialise the term-document matrix to a JSON-friendly dict.
        Delegates to TermDocumentMatrix.to_dict().
        Only suitable for small matrices (demo / educational display).
        """
        return self.tdm.to_dict()