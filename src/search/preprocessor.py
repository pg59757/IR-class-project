"""
2. Text Processing & Natural Language Processing
 
2.1 NLTK Implementation
- **REQ-B13**: Integrate NLTK for text preprocessing
- **REQ-B14**: Implement text tokenization and sentence segmentation
- **REQ-B15**: Handle multiple languages (Portuguese/English)

### 2.2 Stemming and Lemmatization
- **REQ-B16**: Implement stemming algorithms (Porter Stemmer recommended)
- **REQ-B17**: Implement lemmatization using NLTK WordNet
- **REQ-B18**: Allow system configuration to choose between stems/lemmas
- **REQ-B19**: Compare performance between stemming vs lemmatization strategies

### 2.3 Stop Words Processing
- **REQ-B20**: Implement configurable stop words filtering
- **REQ-B21**: Allow inclusion/exclusion of stop words in term dictionary
- **REQ-B22**: Support Portuguese and English stop word lists
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------

_NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def _download_nltk_resources() -> None:
    """Download any missing NLTK resources (silent if already present)."""
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            pass  # will be downloaded below

    for pkg in _NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


_download_nltk_resources()

# ---------------------------------------------------------------------------
# Language type alias
# ---------------------------------------------------------------------------

Language = Literal["english", "portuguese"]

# ---------------------------------------------------------------------------
# POS tag mapping for WordNet lemmatiser
# ---------------------------------------------------------------------------

def _pos_to_wordnet(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the corresponding WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """
    Configuration for the Preprocessor pipeline.

    Attributes:
        language:          Primary language for stop words and stemming.
        use_stemming:      Apply Porter/Snowball stemming when True.
        use_lemmatisation: Apply WordNet lemmatisation when True.
                           If both stemming and lemmatisation are True,
                           lemmatisation runs first, then stemming.
        remove_stopwords:  Filter out stop words when True.
        extra_stopwords:   Additional domain-specific stop words to remove.
        min_token_length:  Discard tokens shorter than this value.
        lowercase:         Convert all tokens to lowercase.
        remove_punctuation:Strip punctuation tokens.
        remove_numbers:    Strip purely numeric tokens.
    """
    language: Language = "english"
    use_stemming: bool = False
    use_lemmatisation: bool = True
    remove_stopwords: bool = True
    extra_stopwords: list[str] = field(default_factory=list)
    min_token_length: int = 2
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False


# ---------------------------------------------------------------------------
# Core Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Flexible NLP preprocessing pipeline.

    Parameters mirror :class:`PreprocessorConfig`; they can also be passed
    directly as keyword arguments for convenience.

    Example
    -------
    >>> pp = Preprocessor(language="portuguese", use_stemming=True)
    >>> pp.process("Os sistemas de recuperação de informação são úteis.")
    ['sistem', 'recuper', 'inform', 'útei']
    """

    def __init__(self, config: PreprocessorConfig | None = None, **kwargs):
        if config is None:
            config = PreprocessorConfig(**kwargs)
        self.config = config

        # --- stop words ---
        self._stopwords: set[str] = set()
        if config.remove_stopwords:
            self._stopwords = self._build_stopwords(config.language)
        if config.extra_stopwords:
            self._stopwords.update(w.lower() for w in config.extra_stopwords)

        # --- stemmer ---
        self._stemmer = None
        if config.use_stemming:
            if config.language == "portuguese":
                self._stemmer = SnowballStemmer("portuguese")
            else:
                self._stemmer = PorterStemmer()

        # --- lemmatiser ---
        self._lemmatiser: WordNetLemmatizer | None = None
        if config.use_lemmatisation:
            self._lemmatiser = WordNetLemmatizer()

        logger.info(
            "Preprocessor ready — lang=%s | stemming=%s | lemmatisation=%s | stopwords=%s",
            config.language,
            config.use_stemming,
            config.use_lemmatisation,
            config.remove_stopwords,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> list[str]:
        """
        Run the full preprocessing pipeline on *text*.

        Steps (in order):
            1. Normalise unicode (remove accents for stemming compatibility)
            2. Tokenise
            3. Lowercase  (if configured)
            4. Remove punctuation / numbers  (if configured)
            5. Remove stop words  (if configured)
            6. Lemmatise  (if configured)
            7. Stem  (if configured)
            8. Filter by minimum length

        Args:
            text: Raw input string.

        Returns:
            List of processed tokens.
        """
        if not text or not text.strip():
            return []

        tokens = self._tokenise(text)

        if self.config.lowercase:
            tokens = [t.lower() for t in tokens]

        if self.config.remove_punctuation:
            tokens = [t for t in tokens if re.search(r"\w", t)]

        if self.config.remove_numbers:
            tokens = [t for t in tokens if not t.isdigit()]

        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self._lemmatiser:
            tokens = self._lemmatise(tokens)

        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        return tokens

    def process_document(self, doc: dict, fields: list[str] | None = None) -> list[str]:
        """
        Process a publication dict and return a combined token list.

        Args:
            doc:    Dict with publication metadata (as produced by the scraper).
            fields: Which fields to process. Defaults to title + abstract.

        Returns:
            Flat list of tokens from all requested fields.
        """
        if fields is None:
            fields = ["title", "abstract"]

        tokens: list[str] = []
        for field_name in fields:
            value = doc.get(field_name, "")
            if isinstance(value, list):          # e.g. authors list
                value = " ".join(value)
            if isinstance(value, str) and value.strip() and value != "N/A":
                tokens.extend(self.process(value))
        return tokens

    def segment_sentences(self, text: str) -> list[str]:
        """
        Split *text* into a list of sentences using NLTK's sentence tokeniser.

        Args:
            text: Raw input string.

        Returns:
            List of sentence strings.
        """
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return sent_tokenize(text, language=lang)

    def get_stopwords(self) -> set[str]:
        """Return the current set of stop words in use."""
        return set(self._stopwords)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Tokenise text using NLTK word_tokenize."""
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return word_tokenize(text, language=lang)

    def _lemmatise(self, tokens: list[str]) -> list[str]:
        """POS-aware lemmatisation using WordNet."""
        # POS tagging only works well for English; for Portuguese we skip POS
        if self.config.language == "english":
            pos_tags = nltk.pos_tag(tokens)
            return [
                self._lemmatiser.lemmatize(token, _pos_to_wordnet(tag))
                for token, tag in pos_tags
            ]
        # Portuguese: lemmatise as nouns (best available without external model)
        return [self._lemmatiser.lemmatize(t) for t in tokens]

    @staticmethod
    def _build_stopwords(language: Language) -> set[str]:
        """Build a combined stop word set (English + Portuguese always included)."""
        words: set[str] = set()
        for lang in ("english", "portuguese"):
            try:
                words.update(stopwords.words(lang))
            except OSError:
                logger.warning("Stop words for '%s' not available.", lang)
        return words

    @staticmethod
    def normalise_unicode(text: str) -> str:
        """
        Decompose unicode characters and strip combining marks (accents).
        Useful before stemming to normalise accented characters.

        Example: "recuperação" → "recuperacao"
        """
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )


# ---------------------------------------------------------------------------
# Factory helpers  (convenient one-liners for common configurations)
# ---------------------------------------------------------------------------

def make_stemming_preprocessor(language: Language = "english") -> Preprocessor:
    """Return a Preprocessor configured for stemming (no lemmatisation)."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=True,
            use_lemmatisation=False,
            remove_stopwords=True,
        )
    )


def make_lemmatisation_preprocessor(language: Language = "english") -> Preprocessor:
    """Return a Preprocessor configured for lemmatisation (no stemming)."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=False,
            use_lemmatisation=True,
            remove_stopwords=True,
        )
    )


def make_bare_preprocessor(language: Language = "english") -> Preprocessor:
    """Return a Preprocessor that only tokenises and lowercases (no filtering)."""
    return Preprocessor(
        PreprocessorConfig(
            language=language,
            use_stemming=False,
            use_lemmatisation=False,
            remove_stopwords=False,
        )
    )