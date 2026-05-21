"""
classifier.py — Multinomial Naïve Bayes document classifier for the IR search engine.

### 4.1 Document Classification
- **REQ-B41**: Implement Multinomial Naïve Bayes classifier
- **REQ-B42**: Train classifier on research publication categories
- **REQ-B43**: Categorize documents into subject areas automatically
- **REQ-B44**: Evaluate classification performance metrics

"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

from src.search.preprocessor import Preprocessor, make_stemming_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category definitions with keyword signals
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, list[str]] = {
    "computer_science": [
        "algorithm", "software", "network", "machine learning", "deep learning",
        "neural", "artificial intelligence", "data", "database", "programming",
        "computer", "computing", "information retrieval", "natural language",
        "image processing", "computer vision", "cryptography", "security",
        "distributed", "cloud", "internet", "web", "mobile", "robot",
        "simulation", "optimization", "graph", "semantic", "ontology",
        "retrieval", "indexing", "classification", "clustering", "nlp",
        "transformer", "embedding", "training", "model", "architecture",
        "system", "framework", "platform", "interface", "user experience",
    ],
    "health_medicine": [
        "health", "medical", "clinical", "patient", "disease", "diagnosis",
        "treatment", "therapy", "cancer", "drug", "pharmaceutical", "hospital",
        "nursing", "surgery", "biology", "bioinformatics", "genomics", "dna",
        "protein", "cell", "tissue", "brain", "cognitive", "mental", "diabetes",
        "cardiovascular", "epidemiology", "public health", "pandemic", "virus",
        "immune", "vaccine", "rehabilitation", "nursing", "biomedical",
        "molecular", "genetics", "physiology", "anatomy", "pharmacology",
    ],
    "engineering": [
        "engineering", "mechanical", "electrical", "civil", "structural",
        "materials", "manufacturing", "fabrication", "design", "prototype",
        "sensor", "actuator", "control", "automation", "robot", "energy",
        "renewable", "solar", "thermal", "fluid", "hydraulic", "aerodynamics",
        "antenna", "signal", "embedded", "hardware", "circuit", "power",
        "construction", "building", "infrastructure", "transport", "vehicle",
        "aerospace", "composite", "polymer", "metal", "alloy",
    ],
    "social_sciences": [
        "social", "society", "education", "pedagogy", "teaching", "learning",
        "student", "school", "university", "curriculum", "policy", "government",
        "law", "legal", "justice", "economics", "economy", "market", "finance",
        "management", "organisation", "leadership", "human resources",
        "history", "cultural", "language", "linguistics", "literature",
        "psychology", "behavior", "communication", "media", "journalism",
        "philosophy", "ethics", "political", "sociology", "demography",
    ],
    "mathematics": [
        "mathematics", "mathematical", "statistic", "probability", "algebra",
        "calculus", "geometry", "topology", "analysis", "differential",
        "equation", "matrix", "linear", "numerical", "computation",
        "theorem", "proof", "combinatorics", "graph theory", "stochastic",
        "bayesian", "regression", "estimation", "inference", "distribution",
    ],
}

# Minimum keyword matches to assign a category (vs. "other")
_MIN_MATCH_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Result of classifying a single document.

    Attributes:
        predicted_category: The top predicted research area.
        confidence:         Probability of the top category (0.0–1.0).
        probabilities:      Dict mapping each category to its probability.
        text_used:          The text that was classified (title + abstract).
    """
    predicted_category: str
    confidence: float
    probabilities: dict[str, float]
    text_used: str = ""


@dataclass
class TrainingReport:
    """
    Summary of classifier training and evaluation.

    Attributes:
        num_documents:       Total documents used for training.
        category_counts:     How many documents per category.
        cross_val_accuracy:  Mean accuracy from 3-fold cross-validation (if run).
        vocabulary_size:     Number of TF-IDF features used.
        categories:          List of category labels.
        per_category_metrics: Precision, recall and F1 per category (REQ-B44).
        macro_avg:           Macro-averaged precision, recall, F1.
        weighted_avg:        Weighted-averaged precision, recall, F1.
    """
    num_documents: int
    category_counts: dict[str, int]
    cross_val_accuracy: float
    vocabulary_size: int
    categories: list[str] = field(default_factory=list)
    per_category_metrics: dict[str, dict] = field(default_factory=dict)
    macro_avg: dict[str, float] = field(default_factory=dict)
    weighted_avg: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Label derivation helper
# ---------------------------------------------------------------------------

def derive_label(text: str) -> str:
    """
    Assign a category label to a document by counting keyword matches.

    For each category, counts how many of its keywords appear in the
    lowercased text. Returns the category with the most matches,
    or "other" if no category exceeds the threshold.

    Args:
        text: Raw document text (title + abstract).

    Returns:
        Category string (one of the keys in CATEGORIES, or "other").
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in CATEGORIES.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)

    best_category = max(scores, key=lambda c: scores[c])
    if scores[best_category] < _MIN_MATCH_THRESHOLD:
        return "other"
    return best_category


# ---------------------------------------------------------------------------
# DocumentClassifier
# ---------------------------------------------------------------------------

class DocumentClassifier:
    """
    Multinomial Naïve Bayes classifier for research publication categories.

    Workflow:
        1. Call ``train(documents)`` to fit the classifier on a corpus.
        2. Call ``classify(text)`` to predict the category of a new text.
        3. Call ``classify_document(doc)`` to classify a document dict.
        4. Call ``classify_batch(documents)`` to label an entire corpus.
        5. Inspect ``training_report_`` for accuracy and coverage metrics.

    Parameters
    ----------
    preprocessor : Preprocessor
        Text preprocessor (stemming / lemmatisation).
    alpha : float
        Laplace smoothing parameter for Naïve Bayes (default 1.0).
    max_features : int
        Maximum TF-IDF vocabulary size (default 5000).
    fields : list[str]
        Document fields to use as text. Default: title + abstract.
    run_cross_validation : bool
        If True, perform 3-fold cross-validation and report mean accuracy.
        Disabled by default when the corpus is small (< 30 docs).
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        alpha: float = 1.0,
        max_features: int = 5000,
        fields: Optional[list[str]] = None,
        run_cross_validation: bool = False,
    ):
        self.preprocessor = preprocessor or make_stemming_preprocessor("english")
        self.alpha = alpha
        self.max_features = max_features
        self.fields = fields or ["title", "abstract"]
        self.run_cross_validation = run_cross_validation

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._model: Optional[MultinomialNB] = None
        self._label_encoder = LabelEncoder()
        self._is_trained: bool = False
        self.training_report_: Optional[TrainingReport] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, documents: list[dict]) -> TrainingReport:
        """
        Derive labels and train the Naïve Bayes classifier.

        Steps:
            1. Extract text (title + abstract) from each document.
            2. Derive a category label for each document via keyword matching.
            3. Vectorise texts with TF-IDF.
            4. Fit MultinomialNB on the TF-IDF matrix.
            5. Optionally run 3-fold cross-validation.

        Args:
            documents: List of document dicts (from the scraper).

        Returns:
            TrainingReport with coverage and accuracy metrics.
        """
        if not documents:
            raise ValueError("Cannot train on an empty document list.")

        logger.info("Deriving labels for %d documents …", len(documents))

        texts: list[str] = []
        labels: list[str] = []

        for doc in documents:
            text = self._extract_text(doc)
            label = derive_label(text)
            texts.append(text)
            labels.append(label)

        # Count categories
        category_counts: dict[str, int] = {}
        for label in labels:
            category_counts[label] = category_counts.get(label, 0) + 1

        logger.info("Category distribution: %s", category_counts)

        # Vectorise
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"\S+",
        )

        # Pre-process texts before vectorising
        processed_texts = [
            " ".join(self.preprocessor.process(t)) for t in texts
        ]
        X = self._vectorizer.fit_transform(processed_texts)

        # Encode labels
        y = self._label_encoder.fit_transform(labels)

        # Fit classifier
        self._model = MultinomialNB(alpha=self.alpha)
        self._model.fit(X, y)
        self._is_trained = True

        # Per-category metrics: precision, recall, F1  (REQ-B44)
        y_pred = self._model.predict(X)
        report_dict = classification_report(
            y, y_pred,
            target_names=self._label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        per_category: dict[str, dict] = {}
        for cat in self._label_encoder.classes_:
            entry = report_dict.get(cat, {})
            per_category[cat] = {
                "precision": round(entry.get("precision", 0.0), 4),
                "recall":    round(entry.get("recall",    0.0), 4),
                "f1":        round(entry.get("f1-score",  0.0), 4),
                "support":   int(entry.get("support",     0)),
            }
        macro    = report_dict.get("macro avg", {})
        weighted = report_dict.get("weighted avg", {})

        # Cross-validation (only when corpus is large enough)
        cv_accuracy = 0.0
        if self.run_cross_validation and len(documents) >= 10:
            n_folds = min(3, len(set(labels)))  # can't have more folds than classes
            if n_folds >= 2:
                scores = cross_val_score(self._model, X, y, cv=n_folds, scoring="accuracy")
                cv_accuracy = float(scores.mean())
                logger.info(
                    "%d-fold CV accuracy: %.3f (±%.3f)", n_folds, cv_accuracy, scores.std()
                )

        self.training_report_ = TrainingReport(
            num_documents=len(documents),
            category_counts=category_counts,
            cross_val_accuracy=cv_accuracy,
            vocabulary_size=len(self._vectorizer.vocabulary_),
            categories=list(self._label_encoder.classes_),
            per_category_metrics=per_category,
            macro_avg={
                "precision": round(macro.get("precision", 0.0), 4),
                "recall":    round(macro.get("recall",    0.0), 4),
                "f1":        round(macro.get("f1-score",  0.0), 4),
            },
            weighted_avg={
                "precision": round(weighted.get("precision", 0.0), 4),
                "recall":    round(weighted.get("recall",    0.0), 4),
                "f1":        round(weighted.get("f1-score",  0.0), 4),
            },
        )

        logger.info(
            "Classifier trained: %d docs | %d categories | vocab=%d",
            len(documents),
            len(category_counts),
            len(self._vectorizer.vocabulary_),
        )

        return self.training_report_

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        """
        Predict the research category for a raw text string.

        Args:
            text: Free-form text (e.g. title + abstract of a new document).

        Returns:
            ClassificationResult with the predicted category and probabilities.

        Raises:
            RuntimeError: If the classifier has not been trained yet.
        """
        self._check_trained()

        processed = " ".join(self.preprocessor.process(text))
        X = self._vectorizer.transform([processed])
        proba = self._model.predict_proba(X)[0]  # shape: (n_classes,)

        categories = list(self._label_encoder.classes_)
        probabilities = {cat: round(float(p), 4) for cat, p in zip(categories, proba)}

        best_idx = int(np.argmax(proba))
        predicted = categories[best_idx]
        confidence = round(float(proba[best_idx]), 4)

        return ClassificationResult(
            predicted_category=predicted,
            confidence=confidence,
            probabilities=probabilities,
            text_used=text[:200] + ("…" if len(text) > 200 else ""),
        )

    def classify_document(self, doc: dict) -> ClassificationResult:
        """
        Classify a document dict by extracting its text fields.

        Args:
            doc: Document dict with at least title and/or abstract.

        Returns:
            ClassificationResult.
        """
        text = self._extract_text(doc)
        return self.classify(text)

    def classify_batch(self, documents: list[dict]) -> list[ClassificationResult]:
        """
        Classify a list of documents in one vectorisation call (efficient).

        Args:
            documents: List of document dicts.

        Returns:
            List of ClassificationResult, one per document.
        """
        self._check_trained()

        texts = [self._extract_text(doc) for doc in documents]
        processed = [" ".join(self.preprocessor.process(t)) for t in texts]
        X = self._vectorizer.transform(processed)
        probas = self._model.predict_proba(X)

        categories = list(self._label_encoder.classes_)
        results = []
        for i, proba in enumerate(probas):
            best_idx = int(np.argmax(proba))
            results.append(ClassificationResult(
                predicted_category=categories[best_idx],
                confidence=round(float(proba[best_idx]), 4),
                probabilities={cat: round(float(p), 4) for cat, p in zip(categories, proba)},
                text_used=texts[i][:200],
            ))
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_top_features(self, category: str, n: int = 10) -> list[tuple[str, float]]:
        """
        Return the top N TF-IDF feature weights for a given category.

        Useful for explaining WHY a document was classified into a category.

        Args:
            category: Category label string.
            n:        Number of top features to return.

        Returns:
            List of (term, weight) tuples sorted by weight descending.
        """
        self._check_trained()

        if category not in self._label_encoder.classes_:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Available: {list(self._label_encoder.classes_)}"
            )

        class_idx = list(self._label_encoder.classes_).index(category)
        feature_log_probs = self._model.feature_log_prob_[class_idx]
        feature_names = self._vectorizer.get_feature_names_out()

        top_indices = np.argsort(feature_log_probs)[::-1][:n]
        return [(feature_names[i], round(float(feature_log_probs[i]), 4)) for i in top_indices]

    @property
    def is_trained(self) -> bool:
        """True if the classifier has been trained."""
        return self._is_trained

    @property
    def categories(self) -> list[str]:
        """List of known category labels (empty if not trained)."""
        if not self._is_trained:
            return []
        return list(self._label_encoder.classes_)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text(self, doc: dict) -> str:
        """Concatenate the configured fields into a single text string."""
        parts = []
        for field_name in self.fields:
            value = doc.get(field_name, "")
            if isinstance(value, list):
                value = " ".join(value)
            if isinstance(value, str) and value.strip() and value != "N/A":
                parts.append(value.strip())
        return " ".join(parts)

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Classifier has not been trained yet. Call train(documents) first."
            )

    def __repr__(self) -> str:
        status = f"trained, {len(self.categories)} categories" if self._is_trained else "untrained"
        return f"DocumentClassifier({status}, alpha={self.alpha})"