"""
populate_db.py — Script de integração: popula a base de dados a partir dos
dados existentes no scraper e nos motores de pesquisa.

Executar:
    python -m src.storage.populate_db

Ou com caminho personalizado:
    python -m src.storage.populate_db --db data/repositorium.db
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(db_path: str = "data/repositorium.db") -> None:
    # Importações locais para evitar problemas de path ao executar directamente
    from src.storage.database import DocumentStore
    from src.search.preprocessor import make_stemming_preprocessor
    from src.search.inverted_index import InvertedIndex

    # ------------------------------------------------------------------
    # 1. Carregar documentos do scraper
    # ------------------------------------------------------------------
    data_file = Path(__file__).parent.parent / "scraper" / "scraper_results.json"
    if not data_file.exists():
        logger.error("Ficheiro de dados não encontrado: %s", data_file)
        sys.exit(1)

    with open(data_file, encoding="utf-8") as fh:
        documents = json.load(fh)

    logger.info("Carregados %d documentos do scraper.", len(documents))

    # ------------------------------------------------------------------
    # 2. Pré-processar conteúdo (REQ-B10)
    # ------------------------------------------------------------------
    pp = make_stemming_preprocessor("english")
    processed_texts: dict[int, list[str]] = {}
    for doc_id, doc in enumerate(documents):
        tokens = pp.process_document(doc, fields=["title", "abstract"])
        processed_texts[doc_id] = tokens

    logger.info("Pré-processamento concluído para %d documentos.", len(processed_texts))

    # ------------------------------------------------------------------
    # 3. Inicializar a base de dados
    # ------------------------------------------------------------------
    store = DocumentStore(db_path)
    store.init_schema()

    # ------------------------------------------------------------------
    # 4. Guardar documentos, autores e metadados (REQ-B09, B10, B11)
    # ------------------------------------------------------------------
    saved = store.save_documents(documents, processed_texts)
    logger.info("Guardados %d documentos na base de dados.", saved)

    # ------------------------------------------------------------------
    # 5. Construir e guardar o índice invertido (REQ-B12)
    # ------------------------------------------------------------------
    inv_index = InvertedIndex(pp)
    inv_index.build_from_documents(documents)
    store.save_inverted_index(inv_index)

    # ------------------------------------------------------------------
    # 6. Estatísticas finais
    # ------------------------------------------------------------------
    stats = store.get_stats()
    logger.info("=== Base de dados populada com sucesso ===")
    for table, count in stats.items():
        logger.info("  %-30s %d registos", table, count)

    store.close()
    logger.info("Ligação fechada. Base de dados: %s", Path(db_path).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popula a base de dados IR.")
    parser.add_argument(
        "--db",
        default="data/repositorium.db",
        help="Caminho para o ficheiro SQLite (default: data/repositorium.db)",
    )
    args = parser.parse_args()
    main(args.db)