from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
import re

from src.search.tfidf import TFIDFEngine
from src.search.preprocessor import make_stemming_preprocessor
from src.search.inverted_index import InvertedIndex
from src.search.boolean_search import BooleanSearchEngine
from src.search.classifier import DocumentClassifier
from src.config import Settings

# -------------------------------------------------
# DATA PATH
# -------------------------------------------------
DATA_PATH = Settings().DATA_FILE

app = FastAPI(title="IR Search Engine")

# CORS — permite que o frontend (ficheiro local ou outro porto) aceda à API
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

documents = []

# Dois engines TF-IDF: custom e sklearn
_engine_custom = TFIDFEngine(
    preprocessor=make_stemming_preprocessor("english"),
    use_sklearn=False,
)
_engine_sklearn = TFIDFEngine(
    preprocessor=make_stemming_preprocessor("english"),
    use_sklearn=True,
)

# Engine padrão (custom)
engine = _engine_custom

# Índice invertido + motor booleano (partilham o mesmo preprocessor)
_inverted_index = InvertedIndex(
    preprocessor=make_stemming_preprocessor("english"),
)
_boolean_engine = BooleanSearchEngine(_inverted_index)

# Classificador Naïve Bayes
_classifier = DocumentClassifier(
    preprocessor=make_stemming_preprocessor("english"),
)


# -------------------------------------------------
# LOAD DATA  — usa DATA_PATH no momento da chamada
# -------------------------------------------------
def load_documents():
    global documents
    path = Path(DATA_PATH)
    if not path.exists():
        raise RuntimeError(f"Data file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    _engine_custom.build_from_documents(documents)
    _engine_sklearn.build_from_documents(documents)
    _inverted_index.build_from_documents(documents)
    _classifier.train(documents)


# Carrega ao arrancar; nos testes o fixture muda DATA_PATH e chama
# /stats ou qualquer endpoint que force reload — mas como o fixture
# usa TestClient como context manager o app já está inicializado.
# A solução: usar lifespan ou detetar mudança de DATA_PATH.
# Alternativa mais simples: carregar dentro do próprio TestClient via
# evento de startup. Usamos @app.on_event para garantir reload.

_loaded_path = None


def ensure_loaded():
    """Recarrega se DATA_PATH mudou desde o último load."""
    global _loaded_path
    current = str(DATA_PATH)
    if _loaded_path != current:
        load_documents()
        _loaded_path = current


# -------------------------------------------------
# MODELS
# -------------------------------------------------
class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[dict]


# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    ensure_loaded()
    return {"status": "ok"}


# -------------------------------------------------
# SEARCH FREETEXT
# -------------------------------------------------
@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(...),
    top_k: int = Settings().TOP_K_DEFAULT,
    algorithm: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
):
    ensure_loaded()

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Seleciona engine consoante algoritmo pedido
    active_engine = _engine_sklearn if algorithm == "sklearn" else _engine_custom

    results = active_engine.search(q, top_k=top_k)

    output = []
    for r in results:
        doc = r.document

        # Filtro de ano
        if year_from or year_to:
            try:
                year = int(doc.get("year", 0))
            except (ValueError, TypeError):
                year = 0
            if year_from and year < year_from:
                continue
            if year_to and year > year_to:
                continue

        output.append({
            "doc_id": r.doc_id,
            "title": doc.get("title"),
            "abstract": doc.get("abstract"),
            "authors": doc.get("authors", []),
            "year": doc.get("year"),
            "doi": doc.get("doi"),
            "document_link": doc.get("document_link"),
            "score": r.score,
        })

    return {"query": q, "total": len(output), "results": output}


@app.get("/search/boolean", response_model=SearchResponse)
def search_boolean(q: str = Query(...)):
    ensure_loaded()

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    results = _boolean_engine.search(q)

    output = [
        {
            "doc_id": r.doc_id,
            "title": r.document.get("title"),
            "abstract": r.document.get("abstract"),
            "authors": r.document.get("authors", []),
            "year": r.document.get("year"),
            "doi": r.document.get("doi"),
            "document_link": r.document.get("document_link"),
            "score": r.score,
        }
        for r in results
    ]

    return {"query": q, "total": len(output), "results": output}


# -------------------------------------------------
# SEARCH AUTHOR
# -------------------------------------------------
@app.get("/search/author", response_model=SearchResponse)
def search_author(name: str = Query(...)):
    ensure_loaded()

    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Empty name")

    name_lower = name.lower()
    output = []

    for idx, doc in enumerate(documents):
        authors = doc.get("authors", [])
        if any(name_lower in author.lower() for author in authors):
            output.append({
                "doc_id": idx,
                "title": doc.get("title"),
                "abstract": doc.get("abstract"),
                "authors": authors,
                "year": doc.get("year"),
                "doi": doc.get("doi"),
                "document_link": doc.get("document_link"),
                "score": 1.0,
            })

    return {"query": name, "total": len(output), "results": output}


# -------------------------------------------------
# GET DOCUMENT  — com doc_id no body
# -------------------------------------------------
@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    ensure_loaded()

    if doc_id < 0 or doc_id >= len(documents):
        raise HTTPException(status_code=404, detail="Document not found")

    doc = dict(documents[doc_id])   # cópia para não mutar o original
    doc["doc_id"] = doc_id          # campo exigido pelo teste
    return doc


# -------------------------------------------------
# SIMILAR DOCUMENTS
# -------------------------------------------------
@app.get("/similar/{doc_id}")
def similar(doc_id: int, top_k: int = Query(default=5)):
    """Devolve os top_k documentos mais similares ao documento dado (cosine TF-IDF)."""
    ensure_loaded()

    if _engine_custom.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    results = _engine_custom.similar_to(doc_id, top_k=top_k)

    return {
        "doc_id": doc_id,
        "total": len(results),
        "results": [
            {
                "doc_id": r.doc_id,
                "title": r.document.get("title"),
                "authors": r.document.get("authors", []),
                "year": r.document.get("year"),
                "similarity": r.similarity,
            }
            for r in results
        ],
    }


# -------------------------------------------------
# CLASSIFY DOCUMENT
# -------------------------------------------------
@app.get("/classify/{doc_id}")
def classify_document(doc_id: int):
    """Classifica um documento por área temática usando Naïve Bayes."""
    ensure_loaded()

    doc = _engine_custom.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    result = _classifier.classify_document(doc)

    return {
        "doc_id": doc_id,
        "title": doc.get("title"),
        "predicted_category": result.predicted_category,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }


# -------------------------------------------------
# EXPLAIN  — elementos educativos
# -------------------------------------------------
@app.get("/explain")
def explain(q: str = Query(...)):
    """
    Devolve informação pedagógica sobre o processamento de uma query:
    tokens após pré-processamento, IDF de cada termo, e top-5 postings.
    """
    ensure_loaded()

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    preprocessor = _engine_custom.preprocessor
    tokens = preprocessor.process(q)

    term_info = []
    for token in dict.fromkeys(tokens):   # unique, preserving order
        idf = _engine_custom.get_term_idf(token)
        postings = _inverted_index.get_postings(token)
        term_info.append({
            "original_term": token,
            "idf": round(idf, 4),
            "doc_frequency": len(postings),
            "sample_doc_ids": [p.doc_id for p in postings[:5]],
        })

    return {
        "query": q,
        "tokens": tokens,
        "term_details": term_info,
        "total_documents": len(documents),
    }


# -------------------------------------------------
# STATS
# -------------------------------------------------
@app.get("/stats")
def stats():
    ensure_loaded()
    return _engine_custom.stats()


# -------------------------------------------------
# IDF  — query param ?term=
# -------------------------------------------------
@app.get("/tfidf/idf")
def get_idf(term: str = Query(...)):
    ensure_loaded()
    value = _engine_custom.get_term_idf(term)
    return {"term": term, "idf": value}