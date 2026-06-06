# Motor de Pesquisa de Publicações Científicas

Projeto desenvolvido no âmbito da unidade curricular **Pesquisa e Recuperação de Informação** — Universidade do Minho, 2025/2026.

Sistema de recuperação de informação sobre publicações do [RepositóriUM](https://repositorium.uminho.pt), composto por um scraper, um motor de IR, uma API REST e uma interface web.

---

## Requisitos

- Python 3.10+
- Google Chrome (para o scraper)
- pip

---

## Instalação

```bash
# 1. Clonar o repositório
git clone <url-do-fork>
cd PRI_PROJETO1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Descarregar recursos NLTK (feito automaticamente na primeira execução,
#    mas pode ser feito antecipadamente)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

---

## Recolha de Dados (Scraper)

O scraper recolhe metadados de publicações do RepositóriUM usando Selenium.

```bash
cd src/scraper
python main.py
```

Os resultados são guardados em `src/scraper/scraper_results.json`. O número máximo de documentos a recolher é configurável em `main.py` (parâmetro `max_items`).

**Nota:** é necessário ter o Google Chrome instalado. O scraper detecta automaticamente o executável nas localizações padrão do sistema.

### Texto completo (PDFs)

Para os primeiros 20 documentos da coleção, o texto completo foi obtido manualmente: os PDFs foram descarregados do RepositóriUM, convertidos para texto simples e segmentados em frases, produzindo os ficheiros `jsons/pdf01.json` a `jsons/pdf20.json`. O `document_loader.py` funde automaticamente estes ficheiros com os metadados do scraper antes de indexar.

---

## Iniciar a API

```bash
# A partir da raiz do projeto
uvicorn src.api.fastapi_app:app --reload
```

O servidor fica disponível em `http://localhost:8000`.

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Redireciona para a interface web |
| GET | `/search?q=...` | Pesquisa TF-IDF por texto livre |
| GET | `/search/boolean?q=...` | Pesquisa booleana (AND / OR / NOT) |
| GET | `/search/author?name=...` | Pesquisa por nome de autor |
| GET | `/explain?q=...` | Pipeline de pré-processamento e IDF da query |
| GET | `/index/term/{term}` | Postings list e IDF de um termo do vocabulário |
| GET | `/index/intersect?t1=...&t2=...` | Interseção de dois termos com skip pointers |
| GET | `/matrix/stats` | Estatísticas da matriz termo-documento |
| GET | `/matrix/term/{term}` | Vetor binário de um termo na matriz |
| GET | `/stats` | Estatísticas consolidadas do motor |

### Parâmetros de pesquisa (`/search`)

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `q` | string | — | Query de pesquisa |
| `top_k` | int | 20 | Número máximo de resultados |
| `algorithm` | `custom` \| `sklearn` | `custom` | Backend TF-IDF |
| `field` | `all` \| `title` \| `abstract` | `all` | Campo de pesquisa |
| `year_from` | int | — | Filtro: ano mínimo |
| `year_to` | int | — | Filtro: ano máximo |

### Exemplos de queries booleanas

```
machine AND learning
neural OR deep
machine NOT neural
(neural OR deep) AND learning
```

### Documentação interativa (Swagger)

Disponível em `http://localhost:8000/docs` após iniciar o servidor.

---

## Interface Web

Disponível em `http://localhost:8000/ui` após iniciar o servidor.

Funcionalidades:
- Pesquisa por texto livre, booleana e por autor
- Seleção de algoritmo: TF-IDF personalizado vs. sklearn
- Seleção de pré-processamento: stemming vs. lematização, com/sem stop words
- Filtro por campo (todos os campos, título, abstract) e por intervalo de anos
- Comparação lado a lado dos dois algoritmos (botão "Comparar")
- Painel educativo ("Como funciona") com explicação dos modelos TF-IDF e Booleano, tokens processados e valores IDF em tempo real

---

## Estrutura do Projeto

```
PRI_G1_Project_v2/
├── src/
│   ├── config.py               # Configuração centralizada (env-overridable)
│   ├── document_loader.py      # Fusão de metadados do scraper com texto dos PDFs
│   ├── scraper/
│   │   ├── scraper.py          # Scraper Selenium para DSpace 8
│   │   ├── main.py             # Ponto de entrada do scraper
│   │   └── scraper_results.json
│   ├── search/
│   │   ├── text_pipeline.py    # Pipeline NLP (tokenização, stemming, lematização)
│   │   ├── inverted_index.py   # Índice invertido com skip pointers
│   │   ├── boolean_search.py   # Motor booleano AND/OR/NOT com parser recursivo
│   │   ├── term_document_matrix.py  # Matriz termo-documento (binary/tf/tfidf)
│   │   ├── tfidf.py            # TF-IDF custom + sklearn, similaridade do cosseno
│   │   ├── search_engine.py    # Orquestrador unificado do motor de IR
│   │   ├── classifier.py       # Classificador Naïve Bayes por área temática
│   │   ├── query_utils.py      # Utilitários de parsing de queries
│   │   ├── evaluation.py       # Métricas de avaliação
│   │   └── run_engine.py       # Ponto de entrada CLI para o motor de IR
│   ├── api/
│   │   └── fastapi_app.py      # API REST (FastAPI)
│   ├── frontend/
│   │   └── index.html          # Interface web (single-page application)
│   └── storage/
│       ├── database.py         # Camada de persistência SQLite
│       └── populate_db.py      # Script de população da base de dados
├── docs/
│   └── PRI_G1_Report_v2.pdf    # Relatório técnico
├── jsons/                      # Texto completo dos PDFs (pdf01.json … pdf20.json)
├── pdfs/
│   ├── pdfs/                   # Ficheiros PDF originais (pdf01.pdf … pdf20.pdf)
│   └── pdftotext/              # Extração em texto simples (pdf01.txt … pdf20.txt)
├── tests/
│   ├── preprocessor/
│   │   └── test_preprocessor.py
│   ├── test_api.py
│   ├── test_tfidf.py
│   ├── test_scraper.py
│   ├── test_index_search.py
│   ├── test_classifier.py
│   ├── test_database.py
│   └── test_query_utils.py
├── data/
│   └── repositorium.db         # Base de dados SQLite
├── conftest.py
├── requirements.txt
└── README.md
```

---

## Testes

```bash
# Correr todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Apenas a API
pytest tests/test_api.py -v
```

---

## Configuração por Variáveis de Ambiente

As principais configurações podem ser sobrescritas via variáveis de ambiente com o prefixo `IR_`:

```bash
export IR_API_PORT=9000                  # Porta da API (padrão: 8000)
export IR_TOP_K_DEFAULT=10               # Resultados por defeito (padrão: 20)
export IR_TF_SCHEME=raw                  # Esquema TF: raw | log | boolean (padrão: log)
export IR_DEFAULT_TFIDF_BACKEND=sklearn  # Backend padrão (padrão: custom)
export IR_REMOVE_STOPWORDS=false         # Remover stop words (padrão: true)
```

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|------------|------------|
| Scraper | Python, Selenium, Chrome |
| NLP | NLTK (tokenização, stemming Porter/Snowball, lematização WordNet) |
| IR | Índice invertido com skip pointers, matriz termo-documento, TF-IDF, similaridade do cosseno |
| ML | scikit-learn (TF-IDF sklearn) |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Persistência | SQLite |
| Testes | pytest, pytest-cov |