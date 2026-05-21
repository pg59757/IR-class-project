"""
nlp.py — Convenience wrappers for NLP utilities.

This module re-exports the main components of the preprocessor for
convenience, and provides utility functions used elsewhere in the project.
"""

from src.search.preprocessor import (
    Preprocessor,
    PreprocessorConfig,
    make_stemming_preprocessor,
    make_lemmatisation_preprocessor,
    make_bare_preprocessor,
    Language,
)

__all__ = [
    "Preprocessor",
    "PreprocessorConfig",
    "make_stemming_preprocessor",
    "make_lemmatisation_preprocessor",
    "make_bare_preprocessor",
    "Language",
    "get_preprocessor",
]


def get_preprocessor(
    mode: str = "stemming",
    language: Language = "english",
    remove_stopwords: bool = True,
) -> Preprocessor:
    """
    Factory function to get a configured Preprocessor.

    Args:
        mode: "stemming", "lemma", or "bare"
        language: "english" or "portuguese"
        remove_stopwords: Whether to remove stop words.

    Returns:
        Configured Preprocessor instance.

    Example:
        >>> pp = get_preprocessor("lemma", "portuguese")
        >>> tokens = pp.process("Recuperação de informação em sistemas distribuídos")
    """
    if mode == "stemming":
        pp = make_stemming_preprocessor(language)
        pp.config.remove_stopwords = remove_stopwords
        return pp
    elif mode == "lemma":
        pp = make_lemmatisation_preprocessor(language)
        pp.config.remove_stopwords = remove_stopwords
        return pp
    else:
        return make_bare_preprocessor(language)