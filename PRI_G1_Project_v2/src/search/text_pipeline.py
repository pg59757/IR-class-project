"""
text_pipeline.py
----------------
Pipeline completo de pré-processamento de texto NLP.

Inclui:
  - Preprocessor      : pipeline configurável (tokenização, stopwords, stemming, lematização)
  - PreprocessorConfig: dataclass de configuração
  - Funções factory   : make_stemming_preprocessor, make_lemmatisation_preprocessor, make_bare_preprocessor
  - process_texts()   : processa uma pasta de .txt e guarda tokens em .json

Uso direto na linha de comandos:
    python text_pipeline.py --input pasta/txts --output pasta/tokens
    python text_pipeline.py --input pasta/txts --output pasta/tokens --lang portuguese --mode stemming --summary
"""

# ===========================================================================
# Imports
# ===========================================================================

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. NLTK — garantir recursos disponíveis
# ===========================================================================

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
    """Descarrega recursos NLTK em falta (silencioso se já existirem)."""
    for pkg in _NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


_download_nltk_resources()


# ===========================================================================
# 2. Tipos e utilitários
# ===========================================================================

Language = Literal["english", "portuguese"]


def _pos_to_wordnet(treebank_tag: str) -> str:
    """Converte uma tag Penn Treebank para a constante POS do WordNet."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ===========================================================================
# 3. Configuração
# ===========================================================================

@dataclass
class PreprocessorConfig:
    """
    Configuração do pipeline de pré-processamento.

    Atributos:
        language:           
        use_stemming:       
        use_lemmatisation:  
                            
        remove_stopwords:   
        extra_stopwords:    
        min_token_length:   
        lowercase:          
        remove_punctuation: 
        remove_numbers:     
    """
    language: Language = "english"      #Idioma principal para stopwords e stemming.
    use_stemming: bool = False          #Aplica stemming Porter/Snowball.
    use_lemmatisation: bool = True      #Aplica lematização WordNet. Se ambos True, lematização corre antes do stemming.
    remove_stopwords: bool = True       #Remove stopwords.
    extra_stopwords: list[str] = field(default_factory=list)    #Stopwords adicionais de domínio.
    min_token_length: int = 2           #Descarta tokens mais curtos que este valor.
    lowercase: bool = True              #Converte para minúsculas.
    remove_punctuation: bool = True     #Remove tokens de pontuação.
    remove_numbers: bool = False        #Remove tokens numéricos.#


# ===========================================================================
# 4. Preprocessor
# ===========================================================================

class Preprocessor:
    """
    Pipeline flexível de pré-processamento NLP.
    """

    def __init__(self, config: PreprocessorConfig | None = None, **kwargs):
        if config is None:
            config = PreprocessorConfig(**kwargs)
        self.config = config

        # stopwords
        self._stopwords: set[str] = set()
        if config.remove_stopwords:
            self._stopwords = self._build_stopwords(config.language)
        if config.extra_stopwords:
            self._stopwords.update(w.lower() for w in config.extra_stopwords)

        # stemmer
        self._stemmer = None
        if config.use_stemming:
            if config.language == "portuguese":
                self._stemmer = SnowballStemmer("portuguese")
            else:
                self._stemmer = PorterStemmer()

        # lemmatiser
        self._lemmatiser: WordNetLemmatizer | None = None
        if config.use_lemmatisation:
            self._lemmatiser = WordNetLemmatizer()

        logger.info(
            "Preprocessor pronto — lang=%s | stemming=%s | lematização=%s | stopwords=%s",
            config.language, config.use_stemming,
            config.use_lemmatisation, config.remove_stopwords,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process(self, text: str) -> list[str]:
        """
        Executa o pipeline completo sobre `text`.

        Passos (por ordem):
            1. Tokenização
            2. Lowercase
            3. Remoção de pontuação / números
            4. Remoção de stopwords
            5. Lematização
            6. Stemming
            7. Filtro de comprimento mínimo
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
        Processa um dicionário de publicação (como os produzidos pelo scraper)
        e devolve uma lista combinada de tokens.
        """
        if fields is None:
            fields = ["title", "abstract"]

        tokens: list[str] = []
        for field_name in fields:
            value = doc.get(field_name, "")
            if isinstance(value, list):
                value = " ".join(value)
            if isinstance(value, str) and value.strip() and value != "N/A":
                tokens.extend(self.process(value))
        return tokens

    def segment_sentences(self, text: str) -> list[str]:
        """Divide `text` em frases usando o tokenizador NLTK."""
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return sent_tokenize(text, language=lang)

    def get_stopwords(self) -> set[str]:
        """Devolve o conjunto de stopwords em uso."""
        return set(self._stopwords)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        lang = "portuguese" if self.config.language == "portuguese" else "english"
        return word_tokenize(text, language=lang)

    def _lemmatise(self, tokens: list[str]) -> list[str]:
        if self.config.language == "english":
            pos_tags = nltk.pos_tag(tokens)
            return [
                self._lemmatiser.lemmatize(token, _pos_to_wordnet(tag))
                for token, tag in pos_tags
            ]
        return [self._lemmatiser.lemmatize(t) for t in tokens]

    @staticmethod
    def _build_stopwords(language: Language) -> set[str]:
        """Constrói stopwords combinadas (inglês + português sempre incluídos)."""
        words: set[str] = set()
        for lang in ("english", "portuguese"):
            try:
                words.update(stopwords.words(lang))
            except OSError:
                logger.warning("Stopwords para '%s' não disponíveis.", lang)
        return words

    @staticmethod
    def normalise_unicode(text: str) -> str:
        """Remove acentos: 'recuperação' → 'recuperacao'."""
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )


# ===========================================================================
# 5. Factories
# ===========================================================================

def make_stemming_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor configurado para stemming (sem lematização)."""
    return Preprocessor(PreprocessorConfig(
        language=language,
        use_stemming=True,
        use_lemmatisation=False,
        remove_stopwords=True,
    ))


def make_lemmatisation_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor configurado para lematização (sem stemming)."""
    return Preprocessor(PreprocessorConfig(
        language=language,
        use_stemming=False,
        use_lemmatisation=True,
        remove_stopwords=True,
    ))


def make_bare_preprocessor(language: Language = "english") -> Preprocessor:
    """Preprocessor que só tokeniza e converte para lowercase."""
    return Preprocessor(PreprocessorConfig(
        language=language,
        use_stemming=False,
        use_lemmatisation=False,
        remove_stopwords=False,
    ))


# ===========================================================================
# 6. Processamento de pasta de ficheiros .txt
# ===========================================================================

def _build_preprocessor(lang: str, mode: str) -> Preprocessor:
    """Constrói o Preprocessor com base no modo escolhido."""
    if mode == "stemming":
        return make_stemming_preprocessor(language=lang)
    if mode == "lemmatisation":
        return make_lemmatisation_preprocessor(language=lang)
    if mode == "bare":
        return make_bare_preprocessor(language=lang)
    # full (default) — lematização + stemming
    return Preprocessor(PreprocessorConfig(
        language=lang,
        use_stemming=True,
        use_lemmatisation=True,
        remove_stopwords=True,
        min_token_length=2,
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
    ))


def _process_file(txt_path: Path, preprocessor: Preprocessor) -> dict:
    """Lê um .txt e devolve dict com tokens e metadados."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")

    tokens = preprocessor.process(text)
    sentences = preprocessor.segment_sentences(text)
    sentences_tokens = [preprocessor.process(s) for s in sentences]

    return {
        "filename": txt_path.name,
        "num_sentences": len(sentences),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "sentences": [
            {"sentence": s, "tokens": t}
            for s, t in zip(sentences, sentences_tokens)
        ],
    }


def process_texts(
    input_dir: str | Path,      #Pasta com os ficheiros .txt.
    output_dir: str | Path,     #Pasta de destino para os .json.
    lang: str = "english",      #'english' ou 'portuguese'.
    mode: str = "full",         #'full' | 'stemming' | 'lemmatisation' | 'bare'.
    summary: bool = False,      #Se True, gera também um summary.json.
) -> None:
    """
    Processa todos os .txt em `input_dir` e guarda os tokens em `output_dir`.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        logger.error("Pasta de input não existe: %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("Nenhum .txt encontrado em: %s", input_dir)
        return

    logger.info("Encontrados %d ficheiros .txt", len(txt_files))
    logger.info("Idioma: %s | Modo: %s", lang, mode)

    preprocessor = _build_preprocessor(lang, mode)
    summary_data = []

    for txt_path in txt_files:
        logger.info("A processar: %s", txt_path.name)
        result = _process_file(txt_path, preprocessor)

        out_path = output_dir / (txt_path.stem + ".json")
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(
            "  → %d frases | %d tokens → %s",
            result["num_sentences"], result["num_tokens"], out_path.name,
        )

        if summary:
            summary_data.append({
                "filename": result["filename"],
                "num_sentences": result["num_sentences"],
                "num_tokens": result["num_tokens"],
            })

    if summary:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info("Sumário guardado em: %s", summary_path)

    logger.info("Concluído. %d ficheiros processados.", len(txt_files))


# ===========================================================================
# 7. Entrada pela linha de comandos
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline NLP: pré-processa .txt e guarda tokens em .json"
    )
    parser.add_argument("--input",   "-i", required=True, help="Pasta com os .txt")
    parser.add_argument("--output",  "-o", required=True, help="Pasta de saída para os .json")
    parser.add_argument("--lang",    "-l", default="english",
                        choices=["english", "portuguese"], help="Idioma (default: english)")
    parser.add_argument("--mode",    "-m", default="full",
                        choices=["full", "stemming", "lemmatisation", "bare"],
                        help="Modo de pré-processamento (default: full)")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Gera summary.json com estatísticas globais")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_texts(
        input_dir=args.input,
        output_dir=args.output,
        lang=args.lang,
        mode=args.mode,
        summary=args.summary,
    )


    