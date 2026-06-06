"""
document_loader.py — Carregamento e fusão de documentos para o motor IR.

Combina duas fontes de dados:
    1. scraper_results.json  — metadados recolhidos do RepositóriUM (título,
                               autores, ano, DOI, abstract, link para PDF).
                               Cobre todos os documentos (ex: 78 entradas).

    2. jsons/*.json          — texto completo extraído dos PDFs via pdftotext
                               e processado pelo text_pipeline.py.
                               Cobre os primeiros N documentos (ex: 20 entradas),
                               pela mesma ordem que o scraper.

Estratégia de fusão:
    Para cada documento do scraper, adiciona-se "full_text" quando existe um JSON de PDF
    correspondente (texto reconstruído a partir das frases).
    Se não houver PDF, "full_text" fica vazio. Este campo pode ser incluído nos campos de
    indexação (ex.: ["title", "abstract", "full_text"]) para que o texto integral dos PDFs
    contribua para o índice invertido e para os vetores TF‑IDF.

Formato do documento resultante:
    {
        "title":         str,
        "year":          str,
        "doi":           str,
        "abstract":      str,
        "authors":       list[str],
        "document_link": str,
        "full_text":     str,   # "" se não houver PDF correspondente
        "has_full_text": bool,
        "pdf_num_tokens": int,  # 0 se não houver PDF
    }
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_documents(
    scraper_path: str | Path,           #Caminho para o ficheiro scraper_results.json.
    jsons_dir: str | Path,              #Pasta com os ficheiros pdfNN.json gerados pelo text_pipeline.py.
    fields_for_full_text: bool = True,  #Se True, reconstrói o texto completo do PDF a partir das frases
                                        #e guarda-o no campo "full_text" de cada documento.
) -> list[dict]:
    """
    Carrega e funde os metadados do scraper com o texto completo dos PDFs.

    Retorna
        list[dict]
            Lista de documentos prontos para passar ao SearchEngine.build().
            Ordenados da mesma forma que o scraper_results.json.

    Raises
        FileNotFoundError
            Se o scraper_results.json não existir.
    """
    scraper_path = Path(scraper_path)
    jsons_dir    = Path(jsons_dir)

    # ── 1. Carrega metadados do scraper ──────────────────────────────────────
    if not scraper_path.exists():
        raise FileNotFoundError(
            f"Ficheiro do scraper não encontrado: {scraper_path}\n"
            "Corre o scraper primeiro: python src/scraper/main.py"
        )

    with open(scraper_path, encoding="utf-8") as f:
        scraper_docs: list[dict] = json.load(f)

    logger.info("Scraper: %d documentos carregados de '%s'", len(scraper_docs), scraper_path)

    # ── 2. Carrega JSONs de texto completo dos PDFs ──────────────────────────
    pdf_texts: dict[int, dict] = {}   # índice (0-based) → dados do JSON

    if not jsons_dir.exists():
        logger.warning(
            "Pasta de JSONs não encontrada: '%s'. "
            "A continuar só com metadados do scraper.", jsons_dir
        )
    else:
        pdf_json_files = sorted(jsons_dir.glob("pdf*.json"))
        if not pdf_json_files:
            logger.warning("Nenhum ficheiro pdf*.json encontrado em '%s'.", jsons_dir)
        else:
            for json_file in pdf_json_files:
                # Extrai o índice numérico do nome: pdf01.json → 0, pdf20.json → 19
                stem = json_file.stem          # "pdf01"
                num_str = stem.replace("pdf", "").lstrip("0") or "0"
                try:
                    idx = int(num_str) - 1     # converte para 0-based
                except ValueError:
                    logger.warning("Nome de ficheiro inesperado ignorado: %s", json_file.name)
                    continue

                if idx < 0:
                    logger.warning("Índice inválido para '%s', ignorado.", json_file.name)
                    continue

                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    pdf_texts[idx] = data
                    logger.debug("PDF JSON carregado: %s → índice %d", json_file.name, idx)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.error("Erro ao ler '%s': %s", json_file.name, exc)

            logger.info(
                "PDFs: %d ficheiros JSON carregados de '%s'",
                len(pdf_texts), jsons_dir
            )

    # ── 3. Fusão ─────────────────────────────────────────────────────────────
    merged: list[dict] = []

    for idx, scraper_doc in enumerate(scraper_docs):
        doc = dict(scraper_doc)   # cópia para não mutar o original

        # Normaliza "authors" para list[str] (o scraper por vezes devolve string)
        authors = doc.get("authors", [])
        if isinstance(authors, str):
            doc["authors"] = [a.strip() for a in authors.split(";") if a.strip()]

        # Valores por omissão para os campos de texto completo
        doc["full_text"]      = ""
        doc["has_full_text"]  = False
        doc["pdf_num_tokens"] = 0

        # Se existir PDF JSON para este índice, extrai o texto completo
        if idx in pdf_texts:
            pdf_data = pdf_texts[idx]

            if fields_for_full_text:
                # Reconstrói o texto completo a partir das frases originais
                # (não dos tokens já processados, para permitir re-processamento
                #  com diferentes configurações de pré-processamento)
                sentences = pdf_data.get("sentences", [])
                full_text_parts = [s.get("sentence", "") for s in sentences if s.get("sentence")]
                doc["full_text"] = " ".join(full_text_parts)

            doc["has_full_text"]  = True
            doc["pdf_num_tokens"] = pdf_data.get("num_tokens", 0)

            logger.debug(
                "Documento %d ('%s…'): fusão com PDF (%d tokens).",
                idx,
                doc.get("title", "")[:40],
                doc["pdf_num_tokens"],
            )

        merged.append(doc)

    # ── 4. Sumário ───────────────────────────────────────────────────────────
    n_with_pdf = sum(1 for d in merged if d["has_full_text"])
    n_without  = len(merged) - n_with_pdf

    logger.info(
        "Fusão concluída: %d documentos totais | %d com texto completo (PDF) | %d só com metadados",
        len(merged), n_with_pdf, n_without,
    )
    print(
        f"[document_loader] {len(merged)} documentos carregados: "
        f"{n_with_pdf} com texto completo (PDF) + {n_without} só com metadados."
    )

    return merged
