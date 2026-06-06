"""
run_engine.py — Script principal do Motor de Recuperação de Informação.

Carrega ficheiros .txt de uma pasta, constrói todos os índices e permite
executar pesquisas booleanas e TF-IDF interativamente ou via argumentos.

Uso
---
    # Modo interativo (menu no terminal)
    python run_engine.py --input pasta/docs

    # Pesquisa booleana direta
    python run_engine.py --input pasta/docs --query "information AND retrieval" --mode boolean

    # Pesquisa TF-IDF
    python run_engine.py --input pasta/docs --query "neural network" --mode tfidf

    # Escolher idioma, pré-processamento e backend TF-IDF
    python run_engine.py --input pasta/docs --lang portuguese --preprocess stemming --backend sklearn

    # Mostrar estatísticas dos índices
    python run_engine.py --input pasta/docs --stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carregamento de ficheiros .txt
# ---------------------------------------------------------------------------

def load_txt_documents(input_dir: str | Path) -> list[dict]:
    """
    Lê todos os ficheiros .txt de uma pasta e converte-os em dicts
    compatíveis com o SearchEngine.

    Formato produzido:
        {
            "title":    "<nome do ficheiro sem extensão>",
            "abstract": "<conteúdo completo do ficheiro>",
            "authors":  [],
            "year":     "N/A",
            "doi":      "N/A",
            "filename": "<nome do ficheiro>",
        }

    Se o ficheiro tiver uma primeira linha curta (≤120 chars) seguida de
    linha vazia, essa linha é usada como título e o resto como abstract.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"[ERRO] Pasta não encontrada: {input_dir}")
        sys.exit(1)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"[AVISO] Nenhum ficheiro .txt encontrado em: {input_dir}")
        sys.exit(1)

    documents = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        lines = text.splitlines()

        # Heurística: primeira linha curta → título
        if lines and len(lines[0]) <= 120 and (len(lines) == 1 or not lines[1].strip()):
            title = lines[0].strip()
            abstract = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        else:
            title = path.stem.replace("_", " ").replace("-", " ").title()
            abstract = text

        documents.append({
            "title":    title,
            "abstract": abstract,
            "authors":  [],
            "year":     "N/A",
            "doi":      "N/A",
            "filename": path.name,
        })
        print(f"  ✓ Carregado: {path.name}  ({len(abstract.split())} palavras)")

    print(f"\n{len(documents)} documento(s) carregado(s) de '{input_dir}'\n")
    return documents


# ---------------------------------------------------------------------------
# Formatação de resultados
# ---------------------------------------------------------------------------

def print_results(results, search_type: str, query: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {search_type.upper()}  |  query: \"{query}\"")
    print(f"  {len(results)} resultado(s) encontrado(s)")
    print(f"{'─'*60}")
    if not results:
        print("  (sem resultados)")
        return
    for i, r in enumerate(results, 1):
        title   = r.document.get("title", "N/A")
        fname   = r.document.get("filename", "")
        score   = r.score
        snippet = r.document.get("abstract", "")[:120].replace("\n", " ")
        print(f"\n  [{i}] {title}  ({fname})")
        print(f"      Score: {score:.4f}")
        if snippet:
            print(f"      …{snippet}…")
    print()


def print_stats(stats: dict) -> None:
    print(f"\n{'═'*60}")
    print("  ESTATÍSTICAS DOS ÍNDICES")
    print(f"{'═'*60}")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    print()


# ---------------------------------------------------------------------------
# Menu interativo
# ---------------------------------------------------------------------------

MENU = """
╔══════════════════════════════════════╗
║   Motor de Recuperação de Informação  ║
╠══════════════════════════════════════╣
║  1  Pesquisa Booleana (AND/OR/NOT)   ║
║  2  Pesquisa TF-IDF (relevância)     ║
║  3  Pesquisa por Autor               ║
║  4  Interseção de Termos (índice)    ║
║  5  Documentos Similares             ║
║  6  Estatísticas dos Índices         ║
║  7  Mudar backend TF-IDF             ║
║  0  Sair                             ║
╚══════════════════════════════════════╝
"""

def interactive_menu(engine, documents: list[dict]) -> None:
    print(MENU)
    while True:
        try:
            choice = input("Opção: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA sair…")
            break

        if choice == "0":
            print("Até logo!")
            break

        elif choice == "1":
            q = input("  Query booleana (ex: neural AND network NOT survey): ").strip()
            if q:
                results = engine.boolean_search(q)
                print_results(results, "BOOLEANA", q)

        elif choice == "2":
            q = input("  Query TF-IDF (linguagem natural): ").strip()
            k = input("  Top-K resultados [10]: ").strip()
            top_k = int(k) if k.isdigit() else 10
            if q:
                results = engine.tfidf_search(q, top_k=top_k)
                print_results(results, "TF-IDF", q)

        elif choice == "3":
            name = input("  Nome do autor: ").strip()
            if name:
                results = engine.author_search(name)
                print_results(results, "AUTOR", name)

        elif choice == "4":
            t1 = input("  Termo 1: ").strip()
            t2 = input("  Termo 2: ").strip()
            if t1 and t2:
                docs = engine.intersect(t1, t2)
                print(f"\n  Documentos com '{t1}' AND '{t2}': {len(docs)}")
                for d in docs:
                    print(f"   • {d.get('title', 'N/A')}  ({d.get('filename', '')})")

        elif choice == "5":
            id_str = input("  doc_id (inteiro): ").strip()
            k = input("  Top-K [5]: ").strip()
            top_k = int(k) if k.isdigit() else 5
            if id_str.lstrip("-").isdigit():
                results = engine.similar_to(int(id_str), top_k=top_k)
                print_results(results, "SIMILARES", f"doc_id={id_str}")

        elif choice == "6":
            print_stats(engine.stats())

        elif choice == "7":
            current = engine.tfidf_backend
            new_backend = "sklearn" if current == "custom" else "custom"
            print(f"  A mudar de '{current}' para '{new_backend}'…")
            engine.set_tfidf_backend(new_backend, documents)
            print(f"  Backend alterado para: {new_backend}")

        else:
            print("  Opção inválida. Escolhe 0-7.")


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Motor de Recuperação de Informação sobre ficheiros .txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Pasta com os ficheiros .txt a indexar")
    parser.add_argument("--query",  "-q", default=None,
                        help="Query a executar (sem este argumento → modo interativo)")
    parser.add_argument("--mode",   "-m", default="tfidf",
                        choices=["boolean", "tfidf", "author"],
                        help="Tipo de pesquisa (default: tfidf)")
    parser.add_argument("--topk",   "-k", type=int, default=10,
                        help="Número máximo de resultados TF-IDF (default: 10)")
    parser.add_argument("--lang",   "-l", default="english",
                        choices=["english", "portuguese"],
                        help="Idioma dos documentos (default: english)")
    parser.add_argument("--preprocess", "-p", default="stemming",
                        choices=["stemming", "lemmatisation", "bare", "full"],
                        help="Modo de pré-processamento (default: stemming)")
    parser.add_argument("--backend", "-b", default="custom",
                        choices=["custom", "sklearn"],
                        help="Backend TF-IDF (default: custom)")
    parser.add_argument("--tf-scheme", default="log",
                        choices=["raw", "log", "boolean"],
                        help="Esquema de ponderação TF (default: log)")
    parser.add_argument("--fields", nargs="+", default=["title", "abstract"],
                        help="Campos a indexar (default: title abstract)")
    parser.add_argument("--stats",  "-s", action="store_true",
                        help="Mostrar estatísticas dos índices e sair")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar logs detalhados")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # ── Importação aqui para que o logging já esteja configurado ────────────
    from search_engine import SearchEngine

    # ── Carrega documentos ──────────────────────────────────────────────────
    print(f"\nA carregar documentos de: {args.input}")
    documents = load_txt_documents(args.input)

    # ── Constrói o motor ────────────────────────────────────────────────────
    print("A construir índices…  (pode demorar alguns segundos)")
    engine = SearchEngine(
        language=args.lang,
        preprocess_mode=args.preprocess,
        tfidf_backend=args.backend,
        tf_scheme=args.tf_scheme,
        fields=args.fields,
    )
    engine.build(documents)
    print("Índices prontos.\n")

    # ── Modo estatísticas ───────────────────────────────────────────────────
    if args.stats:
        print_stats(engine.stats())
        return

    # ── Modo query direta ───────────────────────────────────────────────────
    if args.query:
        if args.mode == "boolean":
            results = engine.boolean_search(args.query)
            print_results(results, "BOOLEANA", args.query)
        elif args.mode == "tfidf":
            results = engine.tfidf_search(args.query, top_k=args.topk)
            print_results(results, "TF-IDF", args.query)
        elif args.mode == "author":
            results = engine.author_search(args.query)
            print_results(results, "AUTOR", args.query)
        return

    # ── Modo interativo ─────────────────────────────────────────────────────
    interactive_menu(engine, documents)


if __name__ == "__main__":
    main()