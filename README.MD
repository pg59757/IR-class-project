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
| GET | `/` | Estado da API |
| GET | `/search?q=...` | Pesquisa TF-IDF por texto livre |
| GET | `/search/boolean?q=...` | Pesquisa booleana (AND / OR / NOT) |
| GET | `/search/author?name=...` | Pesquisa por nome de autor |
| GET | `/documents/{id}` | Metadados de um documento |
| GET | `/similar/{id}` | Documentos mais similares (cosine TF-IDF) |
| GET | `/classify/{id}` | Classificação temática (Naïve Bayes) |
| GET | `/explain?q=...` | Pipeline de pré-processamento e IDF da query |
| GET | `/stats` | Estatísticas do índice |
| GET | `/tfidf/idf?term=...` | Valor IDF de um termo |

### Parâmetros de pesquisa (`/search`)

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `q` | string | — | Query de pesquisa |
| `top_k` | int | 20 | Número máximo de resultados |
| `algorithm` | `custom` \| `sklearn` | `custom` | Backend TF-IDF |
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
- Comparação lado a lado dos dois algoritmos
- Painel educativo com dados reais: tokens processados, IDF por termo, postings do índice

---

## Estrutura do Projeto

```
PRI_PROJETO1/
├── src/
│   ├── scraper/
│   │   ├── scraper.py          # Scraper Selenium para DSpace 8
│   │   ├── main.py             # Ponto de entrada do scraper
│   │   └── scraper_results.json
│   ├── search/
│   │   ├── preprocessor.py     # Tokenização, stemming, lematização, stop words
│   │   ├── inverted_index.py   # Índice invertido com skip pointers
│   │   ├── boolean_search.py   # Motor booleano AND/OR/NOT com parser completo
│   │   ├── tfidf.py            # TF-IDF custom + sklearn, similaridade do cosseno
│   │   ├── classifier.py       # Classificador Naïve Bayes por área temática
│   │   ├── term_document_matrix.py
│   │   ├── query_utils.py
│   │   ├── nlp.py
│   │   └── evaluation.py
│   ├── api/
│   │   └── fastapi_app.py      # API REST (FastAPI)
│   ├── frontend/
│   │   └── index.html          # Interface web
│   └── storage/
│       ├── database.py
│       └── populate_db.py
├── tests/
│   ├── test_api.py
│   ├── test_tfidf.py
│   ├── test_scraper.py
│   ├── test_index_search.py
│   ├── test_classifier.py
│   ├── test_database.py
│   └── test_query_utils.py
├── data/
│   └── repositorium.db
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
export IR_API_PORT=9000          # Porta da API (padrão: 8000)
export IR_TOP_K_DEFAULT=10       # Resultados por defeito (padrão: 20)
export IR_TF_SCHEME=raw          # Esquema TF: raw | log | boolean (padrão: log)
export IR_DEFAULT_TFIDF_BACKEND=sklearn  # Backend padrão (padrão: custom)
export IR_REMOVE_STOPWORDS=false # Remover stop words (padrão: true)
```

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|------------|------------|
| Scraper | Python, Selenium, Chrome |
| NLP | NLTK (tokenização, stemming Porter/Snowball, lematização WordNet) |
| IR | Índice invertido, TF-IDF, similaridade do cosseno |
| ML | scikit-learn (TF-IDF sklearn, Naïve Bayes) |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Testes | pytest, pytest-cov |

---

## Referências

- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval* ([disponível online](https://nlp.stanford.edu/IR-book/))
- [Documentação NLTK](https://www.nltk.org/)
- [Documentação scikit-learn](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RepositóriUM — UMinho](https://repositorium.uminho.pt)