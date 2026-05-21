"""
inverted_index.py — Inverted index with postings lists for the IR search engine.

### 3.2 Inverted Index
- REQ-B27: Build inverted index data structure
- REQ-B28: Implement postings lists for each term
- REQ-B29: Optimize postings list intersection with skip pointers
- REQ-B30: Store term frequencies and document frequencies
- REQ-B31: Support incremental index updates
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.search.preprocessor import Preprocessor, make_stemming_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    doc_id: int
    tf: int = 0
    positions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "tf": self.tf, "positions": self.positions}

    @staticmethod
    def from_dict(d: dict) -> "Posting":
        return Posting(doc_id=d["doc_id"], tf=d["tf"], positions=d.get("positions", []))


@dataclass
class PostingsList:
    df: int = 0
    postings: list[Posting] = field(default_factory=list)
    skips: dict[int, int] = field(default_factory=dict)

    def build_skip_pointers(self) -> None:
        n = len(self.postings)

        if n < 3:
            self.skips = {}
            return

        skip_interval = max(1, int(math.sqrt(n)))
        self.skips = {}

        for i in range(0, n - skip_interval, skip_interval):
            self.skips[i] = i + skip_interval

    def to_dict(self) -> dict:
        return {
            "df": self.df,
            "postings": [p.to_dict() for p in self.postings],
            "skips": {str(k): v for k, v in self.skips.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "PostingsList":
        return PostingsList(
            df=d["df"],
            postings=[Posting.from_dict(p) for p in d["postings"]],
            skips={int(k): v for k, v in d.get("skips", {}).items()},
        )


# ---------------------------------------------------------------------------
# Inverted Index
# ---------------------------------------------------------------------------

class InvertedIndex:

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        fields: Optional[list[str]] = None,
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.fields = fields or ["title", "abstract"]

        self._index: dict[str, PostingsList] = {}
        self._documents: dict[int, dict] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[dict]) -> None:
        self._index.clear()
        self._documents.clear()
        self._next_id = 0

        for doc in documents:
            self._index_document(doc)

        self._rebuild_skip_pointers()

    def add_document(self, doc: dict) -> int:
        doc_id = self._index_document(doc)

        tokens = set(self.preprocessor.process_document(doc, self.fields))

        for term in tokens:
            pl = self._index.get(term)
            if pl:
                self._update_skip_incremental(pl)

        logger.debug("Added document %d", doc_id)
        return doc_id

    # ------------------------------------------------------------------
    # CORE INDEXING
    # ------------------------------------------------------------------

    def _index_document(self, doc: dict) -> int:
        doc_id = self._next_id
        self._documents[doc_id] = doc
        self._next_id += 1

        position = 0
        term_data: dict[str, Posting] = {}

        for fname in self.fields:
            value = doc.get(fname, "")
            if isinstance(value, list):
                value = " ".join(value)
            if not isinstance(value, str) or not value.strip():
                continue

            tokens = self.preprocessor.process(value)

            for token in tokens:
                if token not in term_data:
                    term_data[token] = Posting(doc_id=doc_id, tf=0, positions=[])
                term_data[token].tf += 1
                term_data[token].positions.append(position)
                position += 1

        for term, posting in term_data.items():
            if term not in self._index:
                self._index[term] = PostingsList(df=0, postings=[])

            pl = self._index[term]
            pl.postings.append(posting)
            pl.df += 1

        # 🔥 garante ordenação (OBRIGATÓRIO para skip pointers)
        for pl in self._index.values():
            pl.postings.sort(key=lambda p: p.doc_id)

        return doc_id

    # ------------------------------------------------------------------
    # SKIP POINTERS
    # ------------------------------------------------------------------

    def _rebuild_skip_pointers(self) -> None:
        for pl in self._index.values():
            pl.build_skip_pointers()

    def _update_skip_incremental(self, pl: PostingsList) -> None:
        n = len(pl.postings)

        if n < 3:
            pl.skips = {}
            return

        skip_interval = max(1, int(math.sqrt(n)))
        pl.skips = {}

        for i in range(0, n - skip_interval, skip_interval):
            pl.skips[i] = i + skip_interval

    # ------------------------------------------------------------------
    # QUERY OPS
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> list[Posting]:
        processed = self.preprocessor.process(term)
        if not processed:
            return []
        pl = self._index.get(processed[0])
        return pl.postings if pl else []

    def get_postings_raw(self, term: str) -> list[Posting]:
        pl = self._index.get(term)
        return pl.postings if pl else []

    def intersect(self, term1: str, term2: str) -> list[Posting]:
        t1 = self.preprocessor.process(term1)
        t2 = self.preprocessor.process(term2)

        if not t1 or not t2:
            return []

        pl1 = self._index.get(t1[0])
        pl2 = self._index.get(t2[0])

        if not pl1 or not pl2:
            return []

        return self._intersect_postings(pl1, pl2)

    # ------------------------------------------------------------------
    # SKIP-AWARE INTERSECTION (B29 FIX)
    # ------------------------------------------------------------------

    def _intersect_postings(self, pl1: PostingsList, pl2: PostingsList) -> list[Posting]:
        result = []

        p1 = pl1.postings
        p2 = pl2.postings

        skips1 = pl1.skips
        skips2 = pl2.skips

        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            doc1 = p1[i].doc_id
            doc2 = p2[j].doc_id

            if doc1 == doc2:
                result.append(p1[i])
                i += 1
                j += 1

            elif doc1 < doc2:
                if i in skips1:
                    skip_i = skips1[i]
                    if skip_i < len(p1) and p1[skip_i].doc_id <= doc2:
                        i = skip_i
                    else:
                        i += 1
                else:
                    i += 1

            else:
                if j in skips2:
                    skip_j = skips2[j]
                    if skip_j < len(p2) and p2[skip_j].doc_id <= doc1:
                        j = skip_j
                    else:
                        j += 1
                else:
                    j += 1

        return result

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------

    def get_document(self, doc_id: int) -> dict | None:
        """Return the document dict for a given doc_id, or None if not found."""
        return self._documents.get(doc_id)

    def search_by_author(self, author_name: str) -> list[dict]:
        """Return all documents whose author list contains author_name (case-insensitive)."""
        name_lower = author_name.lower()
        results = []
        for doc in self._documents.values():
            authors = doc.get("authors", [])
            if any(name_lower in a.lower() for a in authors):
                results.append(doc)
        return results

    def stats(self) -> dict:
        return {
            "num_documents": self.num_documents,
            "vocabulary_size": self.vocabulary_size,
        }

    def __repr__(self) -> str:
        return f"InvertedIndex(docs={self.num_documents}, terms={self.vocabulary_size})"