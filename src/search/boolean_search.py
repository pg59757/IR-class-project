"""
boolean_search.py — Boolean search engine built on top of InvertedIndex.

Supports:
    - AND, OR, NOT operators with correct precedence (NOT > AND > OR)
    - Implicit AND for space-separated terms ("information retrieval" → AND)
    - Parentheses for grouping
    - Returns ranked results (by number of matching terms)

Query examples
--------------
    "information retrieval"          → information AND retrieval  (implicit)
    "information AND retrieval"      → explicit AND
    "neural OR deep"                 → OR
    "machine NOT learning"           → AND NOT
    "(neural OR deep) AND learning"  → grouped query

Quick start
-----------
    from preprocessor import make_stemming_preprocessor
    from inverted_index import InvertedIndex
    from boolean_search import BooleanSearchEngine

    pp  = make_stemming_preprocessor("english")
    idx = InvertedIndex(pp)
    idx.build_from_documents(documents)

    engine = BooleanSearchEngine(idx)
    results = engine.search("information AND retrieval NOT neural")
"""

import logging
import re
from dataclasses import dataclass

from src.search.inverted_index import InvertedIndex, Posting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token types for the query parser
# ---------------------------------------------------------------------------

_AND = "AND"
_OR  = "OR"
_NOT = "NOT"
_LPAREN = "("
_RPAREN = ")"
_TERM   = "TERM"
_EOF    = "EOF"


@dataclass
class _Token:
    type: str
    value: str = ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single result returned by the boolean search engine.

    Attributes:
        doc_id:    Internal document identifier.
        document:  Original document dict (from scraper).
        score:     Simple relevance score (number of query terms matched).
    """
    doc_id: int
    document: dict
    score: float = 0.0


# ---------------------------------------------------------------------------
# Boolean Search Engine
# ---------------------------------------------------------------------------

class BooleanSearchEngine:
    """
    Boolean search engine with AND / OR / NOT support.

    Operator precedence (high → low):
        1. NOT   (unary, right-binding)
        2. AND
        3. OR

    Parameters
    ----------
    index : InvertedIndex
        A populated InvertedIndex instance.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[SearchResult]:
        """
        Execute a boolean query and return a ranked list of results.

        Args:
            query: Boolean query string.

        Returns:
            List of SearchResult, sorted by score (descending).
            Returns an empty list if the query is empty or invalid.
        """
        query = query.strip()
        if not query:
            return []

        logger.info("Boolean search: '%s'", query)

        try:
            tokens = self._tokenise_query(query)
            doc_ids = self._parse(tokens)
            results = self._build_results(doc_ids, query)
            logger.info("Found %d results for query '%s'", len(results), query)
            return results
        except Exception as exc:
            logger.error("Error processing query '%s': %s", query, exc)
            return []

    def search_author(self, author_name: str) -> list[SearchResult]:
        """
        Search for documents by author name.

        Args:
            author_name: Partial or full author name (case-insensitive).

        Returns:
            List of SearchResult objects.
        """
        docs = self.index.search_by_author(author_name)
        return [
            SearchResult(
                doc_id=self._find_doc_id(doc),
                document=doc,
                score=1.0,
            )
            for doc in docs
        ]

    # ------------------------------------------------------------------
    # Query tokeniser
    # ------------------------------------------------------------------

    def _tokenise_query(self, query: str) -> list[_Token]:
        """
        Convert a raw query string into a flat list of _Token objects.

        Handles:
            - AND / OR / NOT keywords (case-insensitive)
            - Parentheses
            - Quoted phrases (treated as single terms)
            - Bare words (implicit AND inserted between consecutive terms)
        """
        # Normalise spacing around parentheses
        query = re.sub(r"\(", " ( ", query)
        query = re.sub(r"\)", " ) ", query)

        raw_tokens = query.split()
        tokens: list[_Token] = []

        for raw in raw_tokens:
            upper = raw.upper()
            if upper == "AND":
                tokens.append(_Token(_AND))
            elif upper == "OR":
                tokens.append(_Token(_OR))
            elif upper == "NOT":
                tokens.append(_Token(_NOT))
            elif raw == "(":
                tokens.append(_Token(_LPAREN))
            elif raw == ")":
                tokens.append(_Token(_RPAREN))
            else:
                # Strip surrounding quotes if present
                term = raw.strip('"\'')
                tokens.append(_Token(_TERM, term))

        # Insert implicit AND between consecutive terms / close-paren+term, etc.
        tokens = self._insert_implicit_and(tokens)
        tokens.append(_Token(_EOF))
        return tokens

    @staticmethod
    def _insert_implicit_and(tokens: list[_Token]) -> list[_Token]:
        """
        Insert AND tokens between adjacent terms or after closing parens
        where no explicit operator is present.

        Rule: if the current token is a TERM or RPAREN and the next is a
        TERM, LPAREN, or NOT → insert AND.
        """
        result: list[_Token] = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                curr_type = tok.type
                next_type = tokens[i + 1].type
                if curr_type in (_TERM, _RPAREN) and next_type in (_TERM, _LPAREN, _NOT):
                    result.append(_Token(_AND))
        return result

    # ------------------------------------------------------------------
    # Recursive-descent parser
    # ------------------------------------------------------------------
    # Grammar (after implicit AND insertion):
    #   expr   → and_expr (OR and_expr)*
    #   and_expr → not_expr (AND not_expr)*
    #   not_expr → NOT not_expr | atom
    #   atom   → TERM | LPAREN expr RPAREN

    def _parse(self, tokens: list[_Token]) -> set[int]:
        """Parse the token list and return a set of matching doc_ids."""
        self._tokens = tokens
        self._pos = 0
        result = self._parse_or()
        return result

    def _current(self) -> _Token:
        return self._tokens[self._pos]

    def _consume(self, expected_type: str) -> _Token:
        tok = self._current()
        if tok.type != expected_type:
            raise SyntaxError(
                f"Expected {expected_type} but got {tok.type} ('{tok.value}')"
            )
        self._pos += 1
        return tok

    def _parse_or(self) -> set[int]:
        left = self._parse_and()
        while self._current().type == _OR:
            self._pos += 1
            right = self._parse_and()
            left = left | right
        return left

    def _parse_and(self) -> set[int]:
        operands = [self._parse_not()]
        while self._current().type == _AND:
            self._pos += 1
            operands.append(self._parse_not())
        
        operands.sort(key=len)  # Sort by size for efficient intersection
        result = operands[0]
        for op in operands[1:]:
            result = result & op
            if not result:
                break  # Early exit if intersection is empty
        return result

    def _parse_not(self) -> set[int]:
        if self._current().type == _NOT:
            self._pos += 1
            operand = self._parse_not()
            all_ids = set(self.index._documents.keys())
            return all_ids - operand
        return self._parse_atom()

    def _parse_atom(self) -> set[int]:
        tok = self._current()
        if tok.type == _TERM:
            self._pos += 1
            postings = self.index.get_postings(tok.value)
            return {p.doc_id for p in postings}
        if tok.type == _LPAREN:
            self._pos += 1  # consume '('
            result = self._parse_or()
            self._consume(_RPAREN)
            return result
        if tok.type == _EOF:
            return set()
        raise SyntaxError(f"Unexpected token: {tok.type} ('{tok.value}')")

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_results(self, doc_ids: set[int], query: str) -> list[SearchResult]:
        """
        Build SearchResult objects for matched doc_ids and sort by score.

        Score = number of query terms (after preprocessing) that appear
        in the document's postings (simple term-overlap score).
        """
        # Extract query terms (ignoring operators)
        stopwords_ops = {"and", "or", "not"}
        query_terms = [
            t for t in query.lower().split()
            if t not in stopwords_ops and t not in ("(", ")")
        ]
        processed_terms = []
        for t in query_terms:
            processed = self.index.preprocessor.process(t)
            processed_terms.extend(processed)

        results = []
        for doc_id in doc_ids:
            doc = self.index.get_document(doc_id)
            if doc is None:
                continue

            # Score = how many distinct query terms match this document
            score = 0.0
            for term in set(processed_terms):
                postings = self.index.get_postings_raw(term)
                if any(p.doc_id == doc_id for p in postings):
                    score += 1.0

            results.append(SearchResult(doc_id=doc_id, document=doc, score=score))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _find_doc_id(self, doc: dict) -> int:
        """Find the doc_id for a given document dict (by object identity)."""
        for doc_id, stored in self.index._documents.items():
            if stored is doc:
                return doc_id
        return -1