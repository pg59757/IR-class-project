import re
import logging
from dataclasses import dataclass
from typing import Optional, Any
from collections import defaultdict

import nltk
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)

# Ensure WordNet
for _pkg in ("wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

# ============================================================
# 🔎 QUERY AST (B45)
# ============================================================

@dataclass
class QueryNode:
    op: str  # AND, OR, NOT, TERM
    value: Optional[str] = None
    left: Optional["QueryNode"] = None
    right: Optional["QueryNode"] = None


class BooleanQueryParser:
    """
    Simple Boolean parser with precedence:
    NOT > AND > OR
    """

    def parse(self, query: str) -> QueryNode:
        tokens = self._tokenize(query)
        return self._parse_or(tokens)

    def _tokenize(self, q: str):
        return re.findall(r"\(|\)|AND|OR|NOT|\w+", q.upper())

    def _parse_or(self, tokens):
        node = self._parse_and(tokens)
        while tokens and tokens[0] == "OR":
            tokens.pop(0)
            node = QueryNode("OR", left=node, right=self._parse_and(tokens))
        return node

    def _parse_and(self, tokens):
        node = self._parse_not(tokens)
        while tokens and tokens[0] == "AND":
            tokens.pop(0)
            node = QueryNode("AND", left=node, right=self._parse_not(tokens))
        return node

    def _parse_not(self, tokens):
        if tokens and tokens[0] == "NOT":
            tokens.pop(0)
            return QueryNode("NOT", left=self._parse_term(tokens))
        return self._parse_term(tokens)

    def _parse_term(self, tokens):
        token = tokens.pop(0)
        if token == "(":
            node = self._parse_or(tokens)
            tokens.pop(0)  # )
            return node
        return QueryNode("TERM", value=token)


# ============================================================
# 🔁 QUERY EXPANSION (B47)
# ============================================================

class QueryExpander:
    _STOPWORDS = {"a","the","and","or","not","in","of","to","for"}

    def __init__(self, max_synonyms: int = 2):
        self.max_synonyms = max_synonyms

    def expand(self, query: str) -> str:
        tokens = query.split()
        out = []

        for t in tokens:
            if t.lower() in self._STOPWORDS:
                out.append(t)
                continue

            syns = self._synonyms(t)
            if syns:
                out.append(" OR ".join([t] + syns))
            else:
                out.append(t)

        return " ".join(out)

    def _synonyms(self, word: str):
        res = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name().replace("_", " ").lower()
                if w != word and w not in res:
                    res.append(w)
                    if len(res) >= self.max_synonyms:
                        return res
        return res


# ============================================================
# 📄 PHRASE + PROXIMITY (B48 improved)
# ============================================================

class PhraseQuery:

    def matches(self, phrase: str, text: str) -> bool:
        return phrase.lower() in text.lower()

    def proximity(self, terms: list[str], text: str, window: int = 3) -> bool:
        tokens = text.lower().split()

        positions = defaultdict(list)
        for i, tok in enumerate(tokens):
            positions[tok].append(i)

        if not all(t in positions for t in terms):
            return False

        for p0 in positions[terms[0]]:
            ok = True
            for i in range(1, len(terms)):
                if not any(abs(p - p0) <= window for p in positions[terms[i]]):
                    ok = False
                    break
            if ok:
                return True
        return False


# ============================================================
# 📊 SEARCH RESULT MODEL (B49)
# ============================================================

@dataclass
class SearchResult:
    doc_id: int
    title: str
    score: float
    snippet: str
    url: Optional[str] = None


# ============================================================
# 🧠 SNIPPETS (B50 improved)
# ============================================================

class SnippetGenerator:

    def extract(self, text: str, terms: list[str], window: int = 120) -> str:
        text = " ".join(text.split())
        text_lower = text.lower()

        for term in terms:
            idx = text_lower.find(term.lower())
            if idx != -1:
                start = max(0, idx - window)
                end = min(len(text), idx + window)
                return "..." + text[start:end] + "..."

        return text[:300]


# ============================================================
# 📤 EXPORT FORMATTER (B52)
# ============================================================

class ResultFormatter:

    def to_json(self, results: list[SearchResult]):
        return [r.__dict__ for r in results]

    def to_xml(self, results: list[SearchResult]):
        xml = "<results>"
        for r in results:
            xml += f"""
            <doc>
                <id>{r.doc_id}</id>
                <title>{r.title}</title>
                <score>{r.score}</score>
            </doc>
            """
        xml += "</results>"
        return xml


# ============================================================
# 👤 AUTHOR SEARCH (B53–B55)
# ============================================================

class AuthorIndex:

    def __init__(self):
        self.index = defaultdict(list)

    def add_document(self, doc_id: int, authors: list[str]):
        for a in authors:
            self.index[a.lower()].append(doc_id)

    def search(self, name: str):
        name = name.lower()
        results = []
        for author, docs in self.index.items():
            if name in author:  # partial match (B54)
                results.extend(docs)
        return list(set(results))