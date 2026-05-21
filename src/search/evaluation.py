"""
evaluation.py — Performance measurement and evaluation for the IR search engine.

Covers requirements:
    REQ-B56: Measure and log indexing time performance
    REQ-B57: Compare indexing speed: stems vs lemmas
    REQ-B58: Monitor memory usage during indexing
    REQ-B59: Implement batch processing for large collections
    REQ-B60: Measure query response times
    REQ-B61: Evaluate search result relevance
    REQ-B62: Compare ranking effectiveness across different methods

Quick start
-----------
    from src.search.evaluation import PerformanceEvaluator

    evaluator = PerformanceEvaluator()
    report = evaluator.full_report(documents, queries=["information retrieval", "deep learning"])
    print(report.summary())
"""

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional

from src.search.preprocessor import (
    make_stemming_preprocessor,
    make_lemmatisation_preprocessor,
)
from src.search.inverted_index import InvertedIndex
from src.search.tfidf import TFIDFEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth relevance judgements (qrels)  — REQ-B61
# ---------------------------------------------------------------------------
#
# A minimal set of queries with manually annotated relevant document titles
# (substring match is used so partial titles work).  Add more entries to
# improve evaluation coverage.
#
DEFAULT_QRELS: dict[str, list[str]] = {
    "information retrieval": [
        "information retrieval",
        "search engine",
        "inverted index",
        "ranking",
        "boolean retrieval",
    ],
    "machine learning": [
        "machine learning",
        "deep learning",
        "neural network",
        "classification",
        "supervised",
    ],
    "natural language processing": [
        "natural language",
        "text mining",
        "sentiment",
        "named entity",
        "nlp",
    ],
    "deep learning": [
        "deep learning",
        "neural network",
        "convolutional",
        "recurrent",
        "transformer",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RelevanceMetrics:
    """
    Relevance evaluation metrics for a single query.  (REQ-B61/B62)

    Attributes:
        query:         The query string.
        method:        Engine label (e.g. \"custom\", \"sklearn\").
        precision_at_k: Fraction of top-K results that are relevant.
        recall_at_k:   Fraction of known relevant docs retrieved in top-K.
        avg_precision: Average precision (area under precision-recall curve).
        k:             Cutoff used.
    """
    query: str
    method: str
    precision_at_k: float
    recall_at_k: float
    avg_precision: float
    k: int


@dataclass
class IndexingStats:
    """Performance metrics for a single indexing run."""
    method: str                   # "stemming" or "lemmatisation"
    num_documents: int
    vocabulary_size: int
    elapsed_seconds: float
    peak_memory_mb: float
    docs_per_second: float


@dataclass
class QueryStats:
    """Performance metrics for a single query execution."""
    query: str
    method: str
    elapsed_ms: float
    num_results: int
    top_score: float


@dataclass
class RankingComparison:
    """Comparison of ranking results between two methods for the same query."""
    query: str
    custom_top5: list[str]    # top-5 titles from custom TF-IDF
    sklearn_top5: list[str]   # top-5 titles from sklearn TF-IDF
    overlap_count: int        # how many titles appear in both top-5


@dataclass
class PerformanceReport:
    """Full evaluation report combining all metrics."""
    indexing: list[IndexingStats] = field(default_factory=list)
    queries: list[QueryStats] = field(default_factory=list)
    ranking_comparisons: list[RankingComparison] = field(default_factory=list)
    relevance: list[RelevanceMetrics] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "  IR Engine — Performance Report", "=" * 60]

        if self.indexing:
            lines.append("\n[Indexing Performance]")
            for s in self.indexing:
                lines.append(
                    f"  {s.method:<18} | docs={s.num_documents:>5} | "
                    f"vocab={s.vocabulary_size:>6} | "
                    f"time={s.elapsed_seconds:.3f}s | "
                    f"mem={s.peak_memory_mb:.1f}MB | "
                    f"{s.docs_per_second:.1f} docs/s"
                )

        if self.queries:
            lines.append("\n[Query Response Times]")
            for q in self.queries:
                lines.append(
                    f"  [{q.method:<10}] {q.query!r:<30} → "
                    f"{q.elapsed_ms:.1f}ms | {q.num_results} results | "
                    f"top_score={q.top_score:.4f}"
                )

        if self.ranking_comparisons:
            lines.append("\n[Ranking Comparison: custom vs sklearn]")
            for c in self.ranking_comparisons:
                lines.append(f"  Query: {c.query!r}")
                lines.append(f"    custom  top-5: {c.custom_top5}")
                lines.append(f"    sklearn top-5: {c.sklearn_top5}")
                lines.append(f"    Overlap: {c.overlap_count}/5")

        if self.relevance:
            lines.append("\n[Relevance Evaluation (P@K / R@K / MAP)]")
            for r in self.relevance:
                lines.append(
                    f"  [{r.method:<10}] {r.query!r:<35} "
                    f"P@{r.k}={r.precision_at_k:.3f} | "
                    f"R@{r.k}={r.recall_at_k:.3f} | "
                    f"AP={r.avg_precision:.3f}"
                )
            # MAP per method
            methods = list({r.method for r in self.relevance})
            for method in sorted(methods):
                aps = [r.avg_precision for r in self.relevance if r.method == method]
                map_score = sum(aps) / len(aps) if aps else 0.0
                lines.append(f"  MAP ({method}): {map_score:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "indexing": [
                {
                    "method": s.method,
                    "num_documents": s.num_documents,
                    "vocabulary_size": s.vocabulary_size,
                    "elapsed_seconds": round(s.elapsed_seconds, 4),
                    "peak_memory_mb": round(s.peak_memory_mb, 2),
                    "docs_per_second": round(s.docs_per_second, 2),
                }
                for s in self.indexing
            ],
            "queries": [
                {
                    "query": q.query,
                    "method": q.method,
                    "elapsed_ms": round(q.elapsed_ms, 2),
                    "num_results": q.num_results,
                    "top_score": round(q.top_score, 6),
                }
                for q in self.queries
            ],
            "ranking_comparisons": [
                {
                    "query": c.query,
                    "custom_top5": c.custom_top5,
                    "sklearn_top5": c.sklearn_top5,
                    "overlap_count": c.overlap_count,
                }
                for c in self.ranking_comparisons
            ],
            "relevance": [
                {
                    "query": r.query,
                    "method": r.method,
                    "precision_at_k": round(r.precision_at_k, 4),
                    "recall_at_k": round(r.recall_at_k, 4),
                    "avg_precision": round(r.avg_precision, 4),
                    "k": r.k,
                }
                for r in self.relevance
            ],
        }


# ---------------------------------------------------------------------------
# PerformanceEvaluator
# ---------------------------------------------------------------------------

class PerformanceEvaluator:
    """
    Measures and compares indexing and search performance.

    Usage
    -----
        evaluator = PerformanceEvaluator()
        report = evaluator.full_report(documents, queries=["machine learning"])
        print(report.summary())
    """

    # ------------------------------------------------------------------
    # Indexing benchmarks (REQ-B56, B57, B58)
    # ------------------------------------------------------------------

    def benchmark_indexing(self, documents: list[dict]) -> list[IndexingStats]:
        """
        Build both a stemming and a lemmatisation inverted index, measuring
        time (REQ-B56, B57) and peak memory usage (REQ-B58).

        Args:
            documents: List of document dicts from the scraper.

        Returns:
            List of two IndexingStats entries (stemming + lemmatisation).
        """
        results = []
        for method, make_pp in [
            ("stemming", make_stemming_preprocessor),
            ("lemmatisation", make_lemmatisation_preprocessor),
        ]:
            logger.info("Benchmarking indexing with %s …", method)
            pp = make_pp("english")
            idx = InvertedIndex(pp)

            tracemalloc.start()
            t0 = time.perf_counter()
            idx.build_from_documents(documents)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            stats = IndexingStats(
                method=method,
                num_documents=idx.num_documents,
                vocabulary_size=idx.vocabulary_size,
                elapsed_seconds=elapsed,
                peak_memory_mb=peak / (1024 * 1024),
                docs_per_second=idx.num_documents / elapsed if elapsed > 0 else 0.0,
            )
            results.append(stats)
            logger.info(
                "  %s: %.3fs | vocab=%d | %.1f docs/s | peak=%.1fMB",
                method, elapsed, idx.vocabulary_size,
                stats.docs_per_second, stats.peak_memory_mb,
            )

        return results

    # ------------------------------------------------------------------
    # TF-IDF indexing benchmark
    # ------------------------------------------------------------------

    def benchmark_tfidf_indexing(self, documents: list[dict]) -> list[IndexingStats]:
        """
        Benchmark TF-IDF index building for both stemming and lemmatisation.

        Args:
            documents: List of document dicts.

        Returns:
            List of IndexingStats for custom and sklearn TF-IDF.
        """
        results = []
        configs = [
            ("tfidf-stemming-custom",      make_stemming_preprocessor,      False),
            ("tfidf-lemmatisation-sklearn", make_lemmatisation_preprocessor, True),
        ]
        for method, make_pp, use_sklearn in configs:
            pp = make_pp("english")
            engine = TFIDFEngine(pp, use_sklearn=use_sklearn, tf_scheme="log")

            tracemalloc.start()
            t0 = time.perf_counter()
            engine.build_from_documents(documents)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            stats = IndexingStats(
                method=method,
                num_documents=engine.num_documents,
                vocabulary_size=len(engine._vocabulary),
                elapsed_seconds=elapsed,
                peak_memory_mb=peak / (1024 * 1024),
                docs_per_second=engine.num_documents / elapsed if elapsed > 0 else 0.0,
            )
            results.append(stats)

        return results

    # ------------------------------------------------------------------
    # Query response time (REQ-B60)
    # ------------------------------------------------------------------

    def benchmark_queries(
        self,
        engine: TFIDFEngine,
        queries: list[str],
        method_label: str = "custom",
        top_k: int = 10,
    ) -> list[QueryStats]:
        """
        Measure response time for a list of queries against a built engine.

        Args:
            engine:       A built TFIDFEngine instance.
            queries:      List of query strings.
            method_label: Label for this engine (e.g. "custom", "sklearn").
            top_k:        Number of results to retrieve per query.

        Returns:
            List of QueryStats, one per query.
        """
        results = []
        for q in queries:
            t0 = time.perf_counter()
            hits = engine.search(q, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            top_score = hits[0].score if hits else 0.0
            results.append(QueryStats(
                query=q,
                method=method_label,
                elapsed_ms=elapsed_ms,
                num_results=len(hits),
                top_score=top_score,
            ))
        return results

    # ------------------------------------------------------------------
    # Ranking comparison (REQ-B62)
    # ------------------------------------------------------------------

    def compare_rankings(
        self,
        documents: list[dict],
        queries: list[str],
        top_k: int = 5,
    ) -> list[RankingComparison]:
        """
        Compare top-K rankings between custom and sklearn TF-IDF for each query.

        Args:
            documents: Corpus to index.
            queries:   List of query strings.
            top_k:     How many results to compare.

        Returns:
            List of RankingComparison entries.
        """
        pp_stem = make_stemming_preprocessor("english")
        pp_lemma = make_lemmatisation_preprocessor("english")

        engine_custom = TFIDFEngine(pp_stem, use_sklearn=False, tf_scheme="log")
        engine_custom.build_from_documents(documents)

        engine_sklearn = TFIDFEngine(pp_lemma, use_sklearn=True, tf_scheme="log")
        engine_sklearn.build_from_documents(documents)

        comparisons = []
        for q in queries:
            custom_hits  = engine_custom.search(q, top_k=top_k)
            sklearn_hits = engine_sklearn.search(q, top_k=top_k)

            custom_titles  = [r.document.get("title", "")[:60] for r in custom_hits]
            sklearn_titles = [r.document.get("title", "")[:60] for r in sklearn_hits]

            overlap = len(set(custom_titles) & set(sklearn_titles))
            comparisons.append(RankingComparison(
                query=q,
                custom_top5=custom_titles,
                sklearn_top5=sklearn_titles,
                overlap_count=overlap,
            ))

        return comparisons

    # ------------------------------------------------------------------
    # Relevance evaluation  (REQ-B61, B62)
    # ------------------------------------------------------------------

    def evaluate_relevance(
        self,
        engine: TFIDFEngine,
        qrels: dict[str, list[str]],
        method_label: str = "custom",
        k: int = 10,
    ) -> list[RelevanceMetrics]:
        """
        Compute P@K, R@K and Average Precision for each query in *qrels*.

        Relevance is determined by substring matching: a retrieved document
        is considered relevant if its title contains any of the strings listed
        for that query in *qrels*.

        Parameters
        ----------
        engine:       A built TFIDFEngine instance.
        qrels:        Dict mapping query string → list of relevant title substrings.
        method_label: Label for this engine (e.g. \"custom\", \"sklearn\").
        k:            Cutoff for P@K and R@K.

        Returns
        -------
        List of RelevanceMetrics, one per query.
        """
        results = []
        for query, relevant_substrings in qrels.items():
            hits = engine.search(query, top_k=k)
            retrieved_titles = [
                h.document.get("title", "").lower() for h in hits
            ]

            def is_relevant(title: str) -> bool:
                return any(sub.lower() in title for sub in relevant_substrings)

            relevance_flags = [is_relevant(t) for t in retrieved_titles]

            # P@K
            n_relevant_retrieved = sum(relevance_flags)
            precision_at_k = n_relevant_retrieved / k if k > 0 else 0.0

            # R@K — estimated: treat total relevant as unique substrings count
            total_relevant = len(relevant_substrings)
            recall_at_k = n_relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

            # Average Precision
            ap = 0.0
            running_relevant = 0
            for i, flag in enumerate(relevance_flags, start=1):
                if flag:
                    running_relevant += 1
                    ap += running_relevant / i
            avg_precision = ap / total_relevant if total_relevant > 0 else 0.0

            results.append(RelevanceMetrics(
                query=query,
                method=method_label,
                precision_at_k=round(precision_at_k, 4),
                recall_at_k=round(recall_at_k, 4),
                avg_precision=round(avg_precision, 4),
                k=k,
            ))
            logger.info(
                "[%s] %r → P@%d=%.3f R@%d=%.3f AP=%.3f",
                method_label, query, k, precision_at_k, k, recall_at_k, avg_precision,
            )

        return results

    # ------------------------------------------------------------------
    # Full report (REQ-B56 to B62)
    # ------------------------------------------------------------------

    def full_report(
        self,
        documents: list[dict],
        queries: Optional[list[str]] = None,
        qrels: Optional[dict[str, list[str]]] = None,
    ) -> PerformanceReport:
        """
        Run all benchmarks and return a consolidated PerformanceReport.

        Parameters
        ----------
        documents: Corpus to use for indexing and search.
        queries:   Optional list of queries for response-time and
                   ranking-comparison benchmarks.
        qrels:     Optional relevance judgements for P@K/R@K/MAP evaluation.
                   Defaults to DEFAULT_QRELS if not provided.

        Returns
        -------
        PerformanceReport with all metrics.
        """
        if queries is None:
            queries = list(DEFAULT_QRELS.keys())

        if qrels is None:
            qrels = DEFAULT_QRELS

        report = PerformanceReport()

        # 1. Indexing benchmarks (inverted index)
        report.indexing = self.benchmark_indexing(documents)

        # 2. TF-IDF indexing benchmarks
        report.indexing += self.benchmark_tfidf_indexing(documents)

        # 3. Query response time benchmarks
        pp_stem = make_stemming_preprocessor("english")
        engine_custom = TFIDFEngine(pp_stem, use_sklearn=False, tf_scheme="log")
        engine_custom.build_from_documents(documents)
        report.queries += self.benchmark_queries(engine_custom, queries, "custom")

        pp_lemma = make_lemmatisation_preprocessor("english")
        engine_sklearn = TFIDFEngine(pp_lemma, use_sklearn=True, tf_scheme="log")
        engine_sklearn.build_from_documents(documents)
        report.queries += self.benchmark_queries(engine_sklearn, queries, "sklearn")

        # 4. Ranking comparison
        report.ranking_comparisons = self.compare_rankings(documents, queries)

        # 5. Relevance evaluation — P@K, R@K, MAP  (REQ-B61, B62)
        report.relevance += self.evaluate_relevance(engine_custom, qrels, "custom")
        report.relevance += self.evaluate_relevance(engine_sklearn, qrels, "sklearn")

        logger.info("Performance report complete.")
        return report


# ---------------------------------------------------------------------------
# Batch processing helper (REQ-B59)
# ---------------------------------------------------------------------------

def build_index_in_batches(
    documents: list[dict],
    batch_size: int = 100,
    language: str = "english",
) -> InvertedIndex:
    """
    Build an inverted index incrementally in batches (REQ-B59).

    Useful for large corpora where building the full index at once would
    consume too much memory.

    Args:
        documents:  Full document list.
        batch_size: Number of documents to process per batch.
        language:   Language for the preprocessor.

    Returns:
        Fully built InvertedIndex.
    """
    pp = make_stemming_preprocessor(language)
    idx = InvertedIndex(pp)

    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        for doc in batch:
            idx.add_document(doc)
        logger.info(
            "Batch processed: %d/%d documents (vocab=%d)",
            min(start + batch_size, total), total, idx.vocabulary_size,
        )

    return idx