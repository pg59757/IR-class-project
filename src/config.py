"""
config.py — Centralised configuration for the IR search engine (REQ-B67).

All tuneable parameters live here.  Override any value via environment
variables (the name is the key in UPPER_SNAKE_CASE prefixed with IR_).

Examples
--------
    # Use default config
    from src.config import settings
    print(settings.TOP_K_DEFAULT)

    # Override at runtime (e.g. in tests)
    settings.TOP_K_DEFAULT = 5
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default):
    """Read an environment variable, converting to the same type as *default*."""
    raw = os.environ.get(f"IR_{key}")
    if raw is None:
        return default
    try:
        return type(default)(raw)
    except (ValueError, TypeError):
        return default


@dataclass
class Settings:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_FILE: Path = BASE_DIR / "src" / "scraper" / "scraper_results.json"
    DB_PATH: str = "data/repositorium.db"
    INDEX_CACHE_DIR: str = "data/index_cache"

    # ------------------------------------------------------------------
    # Scraper (REQ-B07, B08)
    # ------------------------------------------------------------------
    SCRAPER_MAX_ITEMS: int = 500
    SCRAPER_DEFAULT_AREA: str = "computer_science"
    SCRAPER_COLLECTIONS: list = field(default_factory=lambda: [
        "1822/21293",  # Informatics / Computer Science
        "1822/68418",  # Health Sciences
    ])

    # ------------------------------------------------------------------
    # Preprocessing (REQ-B18, B20)
    # ------------------------------------------------------------------
    # "stemming" | "lemmatisation" | "bare"
    DEFAULT_PREPROCESSING: str = "stemming"
    # "english" | "portuguese"
    DEFAULT_LANGUAGE: str = "english"
    REMOVE_STOPWORDS: bool = True
    MIN_TOKEN_LENGTH: int = 2

    # ------------------------------------------------------------------
    # TF-IDF (REQ-B32–B36)
    # ------------------------------------------------------------------
    # "log" | "raw" | "boolean"
    TF_SCHEME: str = "log"
    # "custom" | "sklearn"
    DEFAULT_TFIDF_BACKEND: str = "custom"
    TFIDF_FIELDS: list = field(default_factory=lambda: ["title", "abstract"])

    # ------------------------------------------------------------------
    # Search API (REQ-B49)
    # ------------------------------------------------------------------
    TOP_K_DEFAULT: int = 20
    TOP_K_MAX: int = 100
    SNIPPET_WINDOW_CHARS: int = 120
    SNIPPET_MAX_LENGTH: int = 300
    QUERY_EXPANSION_MAX_SYNONYMS: int = 2
    PHRASE_QUERY_PROXIMITY_WINDOW: int = 3

    # ------------------------------------------------------------------
    # Classifier (REQ-B41–B43)
    # ------------------------------------------------------------------
    CLASSIFIER_MAX_FEATURES: int = 5000
    CLASSIFIER_ALPHA: float = 1.0
    CLASSIFIER_RUN_CROSS_VALIDATION: bool = False

    # ------------------------------------------------------------------
    # Performance / batch processing (REQ-B59)
    # ------------------------------------------------------------------
    BATCH_SIZE: int = 100

    # ------------------------------------------------------------------
    # API (REQ-B63–B66)
    # ------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    CORS_ORIGINS: list = field(default_factory=lambda: ["*"])
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Apply environment-variable overrides."""
        self.TOP_K_DEFAULT  = _env("TOP_K_DEFAULT",  self.TOP_K_DEFAULT)
        self.TOP_K_MAX      = _env("TOP_K_MAX",       self.TOP_K_MAX)
        self.API_PORT       = _env("API_PORT",        self.API_PORT)
        self.API_HOST       = _env("API_HOST",        self.API_HOST)
        self.API_RELOAD     = _env("API_RELOAD",      self.API_RELOAD)
        self.DB_PATH        = _env("DB_PATH",         self.DB_PATH)
        self.LOG_LEVEL      = _env("LOG_LEVEL",       self.LOG_LEVEL)
        self.BATCH_SIZE     = _env("BATCH_SIZE",      self.BATCH_SIZE)
        self.DEFAULT_PREPROCESSING = _env("DEFAULT_PREPROCESSING", self.DEFAULT_PREPROCESSING)
        self.REMOVE_STOPWORDS      = _env("REMOVE_STOPWORDS",      self.REMOVE_STOPWORDS)
        self.TF_SCHEME             = _env("TF_SCHEME",             self.TF_SCHEME)
        self.DEFAULT_TFIDF_BACKEND = _env("DEFAULT_TFIDF_BACKEND", self.DEFAULT_TFIDF_BACKEND)
        self.SNIPPET_WINDOW_CHARS  = _env("SNIPPET_WINDOW_CHARS",  self.SNIPPET_WINDOW_CHARS)
        self.QUERY_EXPANSION_MAX_SYNONYMS = _env(
            "QUERY_EXPANSION_MAX_SYNONYMS", self.QUERY_EXPANSION_MAX_SYNONYMS
        )


# Singleton — import this everywhere
settings = Settings()