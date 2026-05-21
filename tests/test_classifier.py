"""
tests/test_classifier.py — Unit tests for the DocumentClassifier.

Run with:
    pytest tests/test_classifier.py -v
"""

import pytest

from src.search.preprocessor import make_stemming_preprocessor
from src.search.classifier import (
    DocumentClassifier,
    ClassificationResult,
    TrainingReport,
    derive_label,
    CATEGORIES,
)


# ---------------------------------------------------------------------------
# Sample corpus — balanced across categories
# ---------------------------------------------------------------------------

DOCS = [
    # computer_science (4)
    {
        "title": "Deep Learning for Image Recognition",
        "abstract": "We propose a neural network architecture for image classification using convolutional layers and machine learning.",
        "authors": ["Alice"], "year": "2022", "doi": "10.1/cs1", "document_link": "",
    },
    {
        "title": "Information Retrieval with Inverted Indexes",
        "abstract": "This paper describes a software system for document retrieval using inverted indexes and TF-IDF ranking algorithms.",
        "authors": ["Bob"], "year": "2021", "doi": "10.1/cs2", "document_link": "",
    },
    {
        "title": "Natural Language Processing with Transformers",
        "abstract": "We apply transformer-based neural models for natural language understanding and text classification.",
        "authors": ["Carlos"], "year": "2023", "doi": "10.1/cs3", "document_link": "",
    },
    {
        "title": "Graph Algorithms for Network Analysis",
        "abstract": "Graph-based algorithms for distributed computing and network security analysis.",
        "authors": ["Diana"], "year": "2022", "doi": "10.1/cs4", "document_link": "",
    },
    # health_medicine (4)
    {
        "title": "Clinical Diagnosis of Diabetes Using Biomarkers",
        "abstract": "This clinical study analyses patient biomarkers for the early diagnosis and treatment of diabetes.",
        "authors": ["Eva"], "year": "2022", "doi": "10.1/hm1", "document_link": "",
    },
    {
        "title": "Cancer Detection with Medical Imaging",
        "abstract": "We propose a method for cancer detection using medical imaging and clinical data from hospital patients.",
        "authors": ["Frank"], "year": "2021", "doi": "10.1/hm2", "document_link": "",
    },
    {
        "title": "Genomics and Personalised Medicine",
        "abstract": "Genomic sequencing techniques for personalised therapy and drug design in clinical settings.",
        "authors": ["Grace"], "year": "2023", "doi": "10.1/hm3", "document_link": "",
    },
    {
        "title": "Public Health Surveillance Systems",
        "abstract": "Epidemiology and public health monitoring systems for disease outbreak detection and vaccine coverage.",
        "authors": ["Hugo"], "year": "2020", "doi": "10.1/hm4", "document_link": "",
    },
    # engineering (4)
    {
        "title": "Structural Analysis of Composite Materials",
        "abstract": "Mechanical testing and structural analysis of composite materials for aerospace engineering applications.",
        "authors": ["Irene"], "year": "2022", "doi": "10.1/en1", "document_link": "",
    },
    {
        "title": "Renewable Energy Systems Design",
        "abstract": "Design and optimisation of solar and wind energy systems for sustainable electrical engineering.",
        "authors": ["João"], "year": "2023", "doi": "10.1/en2", "document_link": "",
    },
    {
        "title": "Embedded Systems for Automation",
        "abstract": "Hardware design and control of embedded systems for industrial automation and sensor networks.",
        "authors": ["Karim"], "year": "2021", "doi": "10.1/en3", "document_link": "",
    },
    {
        "title": "Civil Infrastructure and Construction Methods",
        "abstract": "Structural engineering methods for civil infrastructure construction and transport networks.",
        "authors": ["Lara"], "year": "2022", "doi": "10.1/en4", "document_link": "",
    },
    # social_sciences (4)
    {
        "title": "Higher Education Policy and Student Outcomes",
        "abstract": "Analysis of education policy and teaching methods and their impact on university student learning outcomes.",
        "authors": ["Marco"], "year": "2022", "doi": "10.1/ss1", "document_link": "",
    },
    {
        "title": "Labour Market Economics and Policy",
        "abstract": "The economics of labour markets and the impact of government policy on employment and social inequality.",
        "authors": ["Nadia"], "year": "2021", "doi": "10.1/ss2", "document_link": "",
    },
    {
        "title": "Digital Media and Political Communication",
        "abstract": "How social media and journalism shape political communication and public opinion in modern society.",
        "authors": ["Oscar"], "year": "2023", "doi": "10.1/ss3", "document_link": "",
    },
    {
        "title": "Legal Frameworks for Human Rights",
        "abstract": "An analysis of legal systems and justice frameworks for the protection of human rights and ethics.",
        "authors": ["Paula"], "year": "2022", "doi": "10.1/ss4", "document_link": "",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pp():
    return make_stemming_preprocessor("english")


@pytest.fixture(scope="module")
def trained_clf(pp):
    clf = DocumentClassifier(pp, run_cross_validation=False)
    clf.train(DOCS)
    return clf


# ---------------------------------------------------------------------------
# derive_label (label derivation heuristic)
# ---------------------------------------------------------------------------

class TestDeriveLabel:
    def test_cs_text(self):
        text = "machine learning neural network deep learning classification algorithm"
        assert derive_label(text) == "computer_science"

    def test_health_text(self):
        text = "clinical patient cancer diagnosis treatment hospital disease"
        assert derive_label(text) == "health_medicine"

    def test_engineering_text(self):
        text = "mechanical engineering structural materials design manufacturing"
        assert derive_label(text) == "engineering"

    def test_social_text(self):
        text = "education policy government social society university student"
        assert derive_label(text) == "social_sciences"

    def test_math_text(self):
        text = "statistics probability algebra calculus mathematical theorem"
        assert derive_label(text) == "mathematics"

    def test_empty_returns_other(self):
        assert derive_label("") == "other"

    def test_gibberish_returns_other(self):
        assert derive_label("asdfqwerty zxcvbnm gibberish nothing") == "other"

    def test_returns_string(self):
        result = derive_label("some text about computer science and algorithms")
        assert isinstance(result, str)

    def test_result_is_known_category(self):
        known = set(CATEGORIES.keys()) | {"other"}
        result = derive_label("this is a document about medical treatment and disease")
        assert result in known


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_returns_training_report(self, pp):
        clf = DocumentClassifier(pp)
        report = clf.train(DOCS)
        assert isinstance(report, TrainingReport)

    def test_report_num_documents(self, trained_clf):
        assert trained_clf.training_report_.num_documents == len(DOCS)

    def test_report_vocabulary_size_positive(self, trained_clf):
        assert trained_clf.training_report_.vocabulary_size > 0

    def test_report_category_counts_nonempty(self, trained_clf):
        assert len(trained_clf.training_report_.category_counts) > 0

    def test_report_categories_list(self, trained_clf):
        assert isinstance(trained_clf.training_report_.categories, list)
        assert len(trained_clf.training_report_.categories) > 0

    def test_is_trained_after_train(self, trained_clf):
        assert trained_clf.is_trained is True

    def test_untrained_is_not_trained(self, pp):
        clf = DocumentClassifier(pp)
        assert clf.is_trained is False

    def test_empty_corpus_raises(self, pp):
        clf = DocumentClassifier(pp)
        with pytest.raises(ValueError):
            clf.train([])

    def test_retrain_replaces_model(self, pp):
        clf = DocumentClassifier(pp)
        clf.train(DOCS[:8])
        assert clf.training_report_.num_documents == 8
        clf.train(DOCS)
        assert clf.training_report_.num_documents == len(DOCS)

    def test_categories_property(self, trained_clf):
        assert len(trained_clf.categories) > 0
        for cat in trained_clf.categories:
            assert isinstance(cat, str)

    def test_untrained_categories_empty(self, pp):
        clf = DocumentClassifier(pp)
        assert clf.categories == []


# ---------------------------------------------------------------------------
# classify (text input)
# ---------------------------------------------------------------------------

class TestClassify:
    def test_returns_classification_result(self, trained_clf):
        result = trained_clf.classify("machine learning for data classification")
        assert isinstance(result, ClassificationResult)

    def test_predicted_category_is_string(self, trained_clf):
        result = trained_clf.classify("deep neural network training")
        assert isinstance(result.predicted_category, str)

    def test_predicted_category_is_known(self, trained_clf):
        result = trained_clf.classify("deep neural network training")
        assert result.predicted_category in trained_clf.categories

    def test_confidence_between_0_and_1(self, trained_clf):
        result = trained_clf.classify("information retrieval algorithms")
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_1(self, trained_clf):
        result = trained_clf.classify("clinical patient diagnosis treatment")
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_probabilities_all_nonnegative(self, trained_clf):
        result = trained_clf.classify("solar energy renewable engineering")
        for cat, prob in result.probabilities.items():
            assert prob >= 0.0, f"Negative probability for '{cat}': {prob}"

    def test_top_probability_matches_predicted(self, trained_clf):
        result = trained_clf.classify("student education university policy teaching")
        best = max(result.probabilities, key=lambda c: result.probabilities[c])
        assert best == result.predicted_category

    def test_confidence_matches_predicted_probability(self, trained_clf):
        result = trained_clf.classify("genomics cancer biology clinical trial")
        expected = result.probabilities[result.predicted_category]
        assert abs(result.confidence - expected) < 1e-4

    def test_text_used_truncated(self, trained_clf):
        long_text = "neural " * 200
        result = trained_clf.classify(long_text)
        assert len(result.text_used) <= 203  # 200 chars + "…"

    def test_untrained_raises_runtime_error(self, pp):
        clf = DocumentClassifier(pp)
        with pytest.raises(RuntimeError):
            clf.classify("some text")

    def test_cs_document_classified_correctly(self, trained_clf):
        result = trained_clf.classify(
            "deep learning neural network algorithm software classification"
        )
        assert result.predicted_category == "computer_science"

    def test_health_document_classified_correctly(self, trained_clf):
        result = trained_clf.classify(
            "clinical patient cancer diagnosis treatment hospital disease"
        )
        assert result.predicted_category == "health_medicine"

    def test_engineering_document_classified_correctly(self, trained_clf):
        result = trained_clf.classify(
            "mechanical engineering structural materials design manufacturing"
        )
        assert result.predicted_category == "engineering"

    def test_social_document_classified_correctly(self, trained_clf):
        result = trained_clf.classify(
            "education university student teaching policy social society"
        )
        assert result.predicted_category == "social_sciences"


# ---------------------------------------------------------------------------
# classify_document (dict input)
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def test_returns_result(self, trained_clf):
        doc = {
            "title": "Convolutional Neural Networks for Image Classification",
            "abstract": "We propose a deep learning model for image recognition.",
        }
        result = trained_clf.classify_document(doc)
        assert isinstance(result, ClassificationResult)

    def test_uses_title_and_abstract(self, trained_clf):
        doc = {
            "title": "Clinical Study on Diabetes",
            "abstract": "Patient diagnosis and treatment of diabetes in a hospital setting.",
        }
        result = trained_clf.classify_document(doc)
        assert result.predicted_category == "health_medicine"

    def test_handles_na_fields(self, trained_clf):
        doc = {"title": "N/A", "abstract": "N/A"}
        result = trained_clf.classify_document(doc)
        assert isinstance(result.predicted_category, str)

    def test_handles_missing_fields(self, trained_clf):
        doc = {"title": "Machine Learning Study"}
        result = trained_clf.classify_document(doc)
        assert isinstance(result, ClassificationResult)

    def test_handles_list_authors_ignored(self, trained_clf):
        doc = {
            "title": "Solar Energy System Design",
            "abstract": "Renewable engineering for energy systems.",
            "authors": ["Alice", "Bob"],
        }
        result = trained_clf.classify_document(doc)
        assert isinstance(result, ClassificationResult)


# ---------------------------------------------------------------------------
# classify_batch
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    def test_returns_list(self, trained_clf):
        results = trained_clf.classify_batch(DOCS[:4])
        assert isinstance(results, list)

    def test_one_result_per_doc(self, trained_clf):
        results = trained_clf.classify_batch(DOCS)
        assert len(results) == len(DOCS)

    def test_all_results_are_classification_result(self, trained_clf):
        results = trained_clf.classify_batch(DOCS[:4])
        for r in results:
            assert isinstance(r, ClassificationResult)

    def test_empty_batch_returns_empty(self, trained_clf):
        results = trained_clf.classify_batch([])
        assert results == []

    def test_batch_matches_individual(self, trained_clf):
        doc = DOCS[0]
        batch_result = trained_clf.classify_batch([doc])[0]
        single_result = trained_clf.classify_document(doc)
        assert batch_result.predicted_category == single_result.predicted_category
        assert abs(batch_result.confidence - single_result.confidence) < 1e-4

    def test_untrained_raises(self, pp):
        clf = DocumentClassifier(pp)
        with pytest.raises(RuntimeError):
            clf.classify_batch(DOCS[:2])


# ---------------------------------------------------------------------------
# get_top_features
# ---------------------------------------------------------------------------

class TestGetTopFeatures:
    def test_returns_list(self, trained_clf):
        features = trained_clf.get_top_features("computer_science")
        assert isinstance(features, list)

    def test_returns_n_features(self, trained_clf):
        features = trained_clf.get_top_features("computer_science", n=5)
        assert len(features) == 5

    def test_features_are_tuples(self, trained_clf):
        features = trained_clf.get_top_features("health_medicine", n=3)
        for term, weight in features:
            assert isinstance(term, str)
            assert isinstance(weight, float)

    def test_unknown_category_raises(self, trained_clf):
        with pytest.raises(ValueError):
            trained_clf.get_top_features("nonexistent_category")

    def test_weights_are_negative_log_probs(self, trained_clf):
        features = trained_clf.get_top_features("engineering", n=10)
        for _, weight in features:
            assert weight <= 0.0  # log probabilities are non-positive

    def test_untrained_raises(self, pp):
        clf = DocumentClassifier(pp)
        with pytest.raises(RuntimeError):
            clf.get_top_features("computer_science")


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_untrained(self, pp):
        clf = DocumentClassifier(pp)
        assert "untrained" in repr(clf)

    def test_repr_trained(self, trained_clf):
        r = repr(trained_clf)
        assert "trained" in r
        assert "categories" in r