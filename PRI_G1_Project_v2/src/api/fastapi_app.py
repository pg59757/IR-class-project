import math
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.search.search_engine import SearchEngine
from src.config import Settings
from src.document_loader import load_documents

# ── configuração ──────────────────────────────────────────────────────────────
_settings = Settings()
DATA_PATH = _settings.DATA_FILE
JSONS_DIR = Path(_settings.BASE_DIR) / "jsons"

app = FastAPI(
    title="IR Search Engine",
    description="Motor de Recuperação de Informação — RepositóriUM / UMinho",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve o frontend em /ui
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


# ── estado global ─────────────────────────────────────────────────────────────
documents: list[dict] = []
_engine_custom:  Optional[SearchEngine] = None
_engine_sklearn: Optional[SearchEngine] = None


def _get_engine(algorithm: str) -> SearchEngine:
    """Devolve o motor correto com base no parâmetro 'algorithm'."""
    return _engine_sklearn if algorithm == "sklearn" else _engine_custom


def ensure_loaded() -> None:
    """
    Carrega documentos e constrói os índices na primeira chamada.

    Fontes de dados:
        1. scraper_results.json — metadados (título, autores, ano, DOI, abstract).
        2. jsons/*.json         — texto completo extraído dos PDFs via pdftotext,
                                  adicionado ao campo 'full_text' de cada documento.
    """
    global documents, _engine_custom, _engine_sklearn
    if _engine_custom is not None and _engine_sklearn is not None:
        return

    scraper_path = Path(DATA_PATH)
    if not scraper_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ficheiro de dados não encontrado: {DATA_PATH}. "
                "Corre o scraper primeiro: python src/scraper/main.py"
            ),
        )

    try:
        documents = load_documents(scraper_path=scraper_path, jsons_dir=JSONS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar documentos: {exc}")

    index_fields = ["title", "abstract", "full_text"]

    _engine_custom = SearchEngine(
        language="english",
        preprocess_mode="stemming",
        tfidf_backend="custom",
        fields=index_fields,
    )
    _engine_sklearn = SearchEngine(
        language="english",
        preprocess_mode="stemming",
        tfidf_backend="sklearn",
        fields=index_fields,
    )
    _engine_custom.build(documents)
    _engine_sklearn.build(documents)


# ── helper: normalizar scores (min-max relativo ao top resultado) ─────────────
def _normalize_scores(results: list) -> list:
    """
    Normaliza os scores para que o melhor fique com 1.0 e os restantes sejam proporcionais, tornando
    a escala mais intuitiva sem alterar a ordenação.
    """
    if not results:
        return results
    max_score = max(float(r.score) for r in results)
    if max_score == 0:
        return results
    for r in results:
        r.score = round(float(r.score) / max_score, 4)
    return results


# ── helper: serializar um resultado ──────────────────────────────────────────
def _fmt(r) -> dict:
    doc = r.document or {}
    return {
        "doc_id":   r.doc_id,
        "score":    round(float(r.score), 4),
        "title":    doc.get("title", "N/A"),
        "authors":  doc.get("authors", []),
        "year":     doc.get("year", "N/A"),
        "abstract": doc.get("abstract", ""),
        "doi":      doc.get("doi", ""),
        "url":      doc.get("document_link", ""),
    }


# ── helper: filtrar por ano ───────────────────────────────────────────────────
def _filter_year(results, year_from, year_to):
    if not year_from and not year_to:
        return results
    filtered = []
    for r in results:
        try:
            y = int((r.document or {}).get("year", 0))
        except (ValueError, TypeError):
            continue
        if year_from and y < year_from:
            continue
        if year_to and y > year_to:
            continue
        filtered.append(r)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# RAIZ → redireciona para /ui
# Resolve o problema de http://127.0.0.1:8000 devolver "Not Found"
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


# ─────────────────────────────────────────────────────────────────────────────
# /search  — pesquisa TF-IDF (texto livre)
#
# Parâmetros aceites (todos os que o frontend envia):
#   q                 — query obrigatória
#   algorithm         — "custom" | "sklearn"   (alias: backend)
#   use_stemming      — "true"/"false" (aceite mas o motor já usa stemming;
#                        reservado para extensão futura)
#   remove_stopwords  — "true"/"false" (idem)
#   field             — "all" | "title" | "abstract"
#   year_from / year_to — filtro de anos
#   top_k             — número de resultados
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/search", summary="Pesquisa TF-IDF (texto livre)")
def search(
    q:                str           = Query(...,       description="Query de pesquisa"),
    algorithm:        str           = Query("custom",  description="Backend TF-IDF: 'custom' ou 'sklearn'"),
    backend:          Optional[str] = Query(None,      description="Alias de 'algorithm' (compatibilidade)"),
    use_stemming:     Optional[str] = Query(None,      description="Aceite para compatibilidade com o frontend"),
    remove_stopwords: Optional[str] = Query(None,      description="Aceite para compatibilidade com o frontend"),
    field:            str           = Query("all",     description="Campo: 'all' | 'title' | 'abstract'"),
    year_from:        Optional[int] = Query(None,      description="Ano mínimo"),
    year_to:          Optional[int] = Query(None,      description="Ano máximo"),
    top_k:            int           = Query(20,        description="Número de resultados"),
):
    ensure_loaded()

    # 'backend' é alias de 'algorithm' (compatibilidade com chamadas antigas)
    algo = backend if backend else algorithm

    engine = _get_engine(algo)
    raw    = engine.tfidf_search(q, top_k=top_k * 3)  # margem para filtro de ano
    raw    = _filter_year(raw, year_from, year_to)

    # Filtro por campo: se pedido, recalcula apenas sobre o campo escolhido
    if field == "title":
        raw = [r for r in raw if q.lower() in (r.document or {}).get("title", "").lower()]
    elif field == "abstract":
        raw = [r for r in raw if q.lower() in (r.document or {}).get("abstract", "").lower()]

    raw = _normalize_scores(raw[:top_k])

    return {
        "query":   q,
        "mode":    "tfidf",
        "backend": algo,
        "field":   field,
        "total":   len(raw),
        "results": [_fmt(r) for r in raw],
    }


# ─────────────────────────────────────────────────────────────────────────────
# /search/boolean  — pesquisa booleana (AND / OR / NOT)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/search/boolean", summary="Pesquisa booleana (AND / OR / NOT)")
def search_boolean(
    q:        str           = Query(...,  description="Query booleana"),
    year_from: Optional[int] = Query(None, description="Ano mínimo"),
    year_to:   Optional[int] = Query(None, description="Ano máximo"),
    top_k:    int           = Query(100,  description="Número máximo de resultados"),
):
    ensure_loaded()

    raw = _engine_custom.boolean_search(q)
    raw = _filter_year(raw, year_from, year_to)
    raw = _normalize_scores(raw[:top_k])

    return {
        "query":   q,
        "mode":    "boolean",
        "total":   len(raw),
        "results": [_fmt(r) for r in raw],
    }


# ─────────────────────────────────────────────────────────────────────────────
# /search/author  — pesquisa por nome de autor
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/search/author", summary="Pesquisa por autor")
def search_author(
    name:  str           = Query(...,  description="Nome (parcial ou completo) do autor"),
    top_k: int           = Query(100,  description="Número máximo de resultados"),
):
    ensure_loaded()

    raw = _engine_custom.author_search(name)
    raw = _normalize_scores(raw[:top_k])

    return {
        "query":   name,
        "mode":    "author",
        "total":   len(raw),
        "results": [_fmt(r) for r in raw],
    }


# ─────────────────────────────────────────────────────────────────────────────
# /explain  — detalhes TF-IDF de uma query (painel educativo no frontend)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/explain", summary="Detalhes TF-IDF de uma query (uso educativo)")
def explain(q: str = Query(..., description="Query a explicar")):
    ensure_loaded()

    engine    = _engine_custom
    tfidf_eng = engine.tfidf_engine
    tokens    = engine.preprocessor.process(q)

    term_details = []
    for tok in tokens:
        idf = tfidf_eng.get_term_idf(tok)
        term_details.append({
            "original_term":   tok,
            "processed_term":  tok,
            "idf":             round(idf, 4),
        })

    # Top 5 documentos por TF-IDF
    top_docs = engine.tfidf_search(q, top_k=5)

    return {
        "query":        q,
        "tokens":       tokens,
        "term_details": term_details,
        "top_docs":     [_fmt(r) for r in top_docs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints de inspeção do índice (mantidos do original)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/index/term/{term}", summary="Informação de um termo no índice invertido")
def index_term(term: str):
    ensure_loaded()
    idx_obj  = _engine_custom.inverted_index
    processed = idx_obj.preprocessor.process(term)

    if not processed:
        return {"term": term, "processed": None, "document_frequency": 0, "postings": []}

    proc_term = processed[0]
    plist     = idx_obj._index.get(proc_term)

    if not plist:
        return {"term": term, "processed": proc_term, "document_frequency": 0, "postings": []}

    num_docs = idx_obj.num_documents
    idf      = 1.0 + math.log(num_docs / (1 + plist.df)) if num_docs > 0 else 0.0

    return {
        "term":               term,
        "processed":          proc_term,
        "document_frequency": plist.df,
        "idf":                round(idf, 4),
        "postings":           [{"doc_id": p.doc_id, "tf": p.tf} for p in plist.postings],
    }


@app.get("/index/intersect", summary="Interseção de dois termos (skip pointers)")
def intersect_terms(
    t1: str = Query(..., description="Primeiro termo"),
    t2: str = Query(..., description="Segundo termo"),
):
    ensure_loaded()
    docs = _engine_custom.intersect(t1, t2)
    return {
        "term1":     t1,
        "term2":     t2,
        "total":     len(docs),
        "documents": [
            {"title": d.get("title", "N/A"), "year": d.get("year", "N/A"),
             "authors": d.get("authors", [])}
            for d in docs
        ],
    }


@app.get("/matrix/stats", summary="Estatísticas da matriz termo-documento")
def matrix_stats():
    ensure_loaded()
    if _engine_custom.boolean_engine is not None:
        return _engine_custom.boolean_engine.matrix_stats()
    return _engine_custom.tdm.stats()


@app.get("/matrix/term/{term}", summary="Vetor de um termo na matriz termo-documento")
def matrix_term_vector(term: str):
    ensure_loaded()
    vec = None
    if _engine_custom.boolean_engine is not None:
        vec = _engine_custom.boolean_engine.get_term_vector(term)
    elif _engine_custom.tdm is not None:
        vec = _engine_custom.tdm.get_term_vector(term)

    if vec is None:
        raise HTTPException(status_code=404, detail="Termo não encontrado no vocabulário da matriz.")

    return {"term": term, "vector": vec.tolist()}


@app.get("/stats", summary="Estatísticas gerais do motor")
def stats():
    ensure_loaded()
    return _engine_custom.stats()
