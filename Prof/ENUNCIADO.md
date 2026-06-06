# Enunciado do Projeto — Motor de Pesquisa de Publicações Científicas

**Unidade Curricular:** Pesquisa e Recuperação de Informação  
**Instituição:** Universidade do Minho  
**Ano letivo:** 2025/2026

---

## 1. Contexto e Motivação

A Universidade do Minho disponibiliza, através do [RepositóriUM](https://repositorium.uminho.pt/), um vasto acervo de publicações científicas — teses, dissertações, artigos e outros documentos produzidos pela sua comunidade académica.

O objetivo deste projeto é construir um **motor de pesquisa** sobre essa coleção, aplicando as técnicas de Recuperação de Informação (IR) lecionadas ao longo da unidade curricular. O sistema deverá permitir que qualquer utilizador pesquise publicações de forma eficiente e relevante, combinando recolha automática de dados, processamento de linguagem natural e uma interface web intuitiva.

---

## 2. Objetivos de Aprendizagem

No final deste projeto, cada estudante deverá ser capaz de:

- Implementar um módulo de *web scraping* para recolha automatizada de metadados;
- Construir um índice invertido e efetuar pesquisas booleanas com listas de postings;
- Aplicar técnicas de pré-processamento de texto (tokenização, *stemming*, lematização, remoção de *stop words*);
- Calcular pesos TF-IDF e ordenar resultados por similaridade do cosseno;
- Desenvolver uma API REST e uma interface web para exposição do sistema;
- Trabalhar colaborativamente com Git/GitHub seguindo boas práticas de desenvolvimento de software.

---

## 3. Descrição do Projeto

O projeto divide-se em três módulos interdependentes que deverão ser integrados numa aplicação coesa:

### 3.1 Módulo de Recolha de Dados (*Scraper*)

Deverá ser desenvolvido um *scraper* que recolha automaticamente metadados de publicações do RepositóriUM. Para cada publicação, devem ser extraídos, no mínimo:

| Campo | Descrição |
|---|---|
| Título | Título da publicação |
| Autores | Lista de autores |
| Resumo (*abstract*) | Texto resumo do documento |
| Data de publicação | Ano de publicação |
| DOI | Identificador do documento |
| Ligação ao PDF | URL de acesso ao documento completo |

**Requisitos mínimos:**
- Suporte a paginação para recolha de grandes coleções;
- Parâmetro configurável para o número máximo de documentos a recolher;
- Tratamento de erros de rede e *timeouts*;
- Exportação dos dados para formato JSON.

> **Ponto de partida:** O ficheiro [`scraper.py`](scraper.py) e [`main.py`](main.py) contêm já uma implementação base de um *scraper* para o DSpace 8 do RepositóriUM. Leiam e compreendam o código existente antes de o estender.

### 3.2 Módulo de *Backend* — Motor de Recuperação de Informação

O *backend* é o núcleo do sistema e deve implementar os seguintes componentes:

#### 3.2.1 Pré-processamento de Texto
- Tokenização e segmentação de frases;
- Remoção de *stop words* em português e inglês (configurável);
- *Stemming* (algoritmo de Porter) **e** lematização (WordNet via NLTK);
- Suporte a múltiplos idiomas (português/inglês).

#### 3.2.2 Modelo Booleano
- Construção de uma matriz termo-documento;
- Suporte a operadores booleanos: `AND`, `OR`, `NOT` com precedência correta;
- Interpretação de termos separados por espaço como `AND` implícito.

#### 3.2.3 Índice Invertido
- Construção de um índice invertido com listas de *postings*;
- Armazenamento de frequências de termos e de documentos;
- Otimização da interseção de *postings* com *skip pointers*;
- Suporte a atualizações incrementais do índice.

#### 3.2.4 TF-IDF e Similaridade
- Implementação própria do cálculo TF-IDF;
- Integração da implementação do `sklearn` para comparação;
- Cálculo de similaridade do cosseno para ordenação de resultados;
- Possibilidade de escolher entre as duas implementações.

#### 3.2.5 API REST
- Endpoints para pesquisa por texto livre, pesquisa booleana e pesquisa por autor;
- Suporte a filtros (data, tipo de documento, área de investigação);
- Retorno de resultados ordenados por relevância com pontuação (*score*);
- Documentação com OpenAPI/Swagger.

### 3.3 Módulo de *Frontend* — Interface Web

Deve ser desenvolvida uma interface web que permita:

- Caixa de pesquisa principal com suporte a consultas booleanas;
- Seleção do modo de processamento (stemming vs. lematização, com/sem *stop words*);
- Seleção do algoritmo de ordenação (TF-IDF personalizado vs. sklearn vs. booleano);
- Visualização dos resultados com título, autores, resumo, pontuação de relevância e ligação ao documento;
- Paginação e filtros de refinamento;
- Pesquisa de autores e página de perfil;
- Elementos educativos que demonstrem o funcionamento dos algoritmos de IR.

---

## 4. Organização das Equipas

O projeto é desenvolvido em equipas de **3 a 5 elementos**, organizados em grupos de trabalho especializados:

| Grupo | Foco principal |
|---|---|
| **Backend** | Scraper, processamento de texto, índice invertido, TF-IDF, API |
| **Frontend** | Interface web, integração com API, experiência do utilizador |
| **DevOps** | Containerização (Docker), CI/CD, testes automatizados, documentação |

> **Nota:** É esperada colaboração transversal entre equipas, especialmente na integração dos módulos. A avaliação contempla contribuições cruzadas entre grupos.

Consulte o ficheiro [`COMTRIBUTING.md`](COMTRIBUTING.md) para instruções detalhadas sobre o fluxo de trabalho com Git e GitHub.

---

## 5. Requisitos Detalhados

Os requisitos técnicos detalhados para cada módulo encontram-se nos seguintes documentos:

- 📄 **Backend:** [`docs/internal/Backend Team Requirements.md`](docs/internal/Backend%20Team%20Requirements.md)
- 📄 **Frontend:** [`docs/internal/Frontend Team Requirements.md`](docs/internal/Frontend%20Team%20Requirements.md)
- 📄 **Avaliação:** [`docs/internal/Student Evaluation Framework.md`](docs/internal/Student%20Evaluation%20Framework.md)

---

## 6. Entrega e Submissão

### 6.1 Como Submeter

1. **Faça *fork*** do repositório do docente para a conta da vossa equipa (ver [`COMTRIBUTING.md`](COMTRIBUTING.md));
2. Trabalhem **exclusivamente no *fork* da equipa** — não submetam *Pull Requests* para o repositório do docente;
3. No final de cada *sprint*, criem uma ***release*** no GitHub com a *tag* correspondente (ex.: `v1.0-sprint1`);
4. Partilhem o link do *fork* com o docente quando solicitado.

### 6.2 Estrutura de Entrega Esperada

```
IR-class-project/
├── src/
│   ├── scraper/          # Módulo de recolha de dados
│   ├── search/           # Motor de IR (índice, TF-IDF, booleano)
│   ├── api/              # API REST (ex.: FastAPI)
│   └── frontend/         # Interface web
├── tests/                # Testes unitários e de integração
├── docs/                 # Documentação técnica
├── docker/               # Ficheiros Docker
├── requirements.txt      # Dependências Python
└── README.md             # Instruções de instalação e uso
```

### 6.3 Prazos (*Sprints*)

| Sprint | Entrega | Objetivos |
|---|---|---|
| **Sprint 1** | A definir pelo docente | Configuração do repositório, *scraper* funcional, recolha de dados |
| **Sprint 2** | A definir pelo docente | Pré-processamento, índice invertido, pesquisa booleana |
| **Sprint 3** | A definir pelo docente | TF-IDF, API REST, integração frontend-backend |
| **Sprint 4** | A definir pelo docente | Refinamento, testes, Docker, documentação, apresentação final |

> Os prazos exatos serão anunciados pelo docente no início de cada sprint.

---

## 7. Critérios de Avaliação

A avaliação final de cada estudante é composta por três componentes:

| Componente | Peso | Descrição |
|---|---|---|
| **A — Sucesso global do projeto** | 40% | Qualidade e funcionalidade do produto final integrado |
| **B — Contribuição técnica individual** | 40% | Qualidade dos *commits*, *pull requests*, revisões de código e gestão de tarefas |
| **C — Avaliação por pares e entre equipas** | 20% | Avaliação anónima por colegas da mesma equipa (e de outras equipas?) |

> Consulte o documento [`docs/internal/Student Evaluation Framework.md`](docs/internal/Student%20Evaluation%20Framework.md) para o detalhe completo dos critérios.

### O que será avaliado no projeto final:

- ✅ Funcionalidade e correta integração dos módulos;
- ✅ Qualidade e organização do código;
- ✅ Testes automatizados (*unit tests* e de integração);
- ✅ Documentação técnica (README, comentários, API docs);
- ✅ Utilização correta do Git (mensagens de *commit*, PRs, revisões);
- ✅ Apresentação final e demonstração ao vivo do sistema.

---

## 8. Recursos e Tecnologias Sugeridas

### Linguagens e Frameworks

| Área | Tecnologias sugeridas |
|---|---|
| *Backend* | Python 3.10+, FastAPI, NLTK, scikit-learn |
| *Frontend* | React / Vue / Angular, CSS (Sass/Less) |
| Base de dados | JSON, SQLite ou PostgreSQL |
| Containerização | Docker, Docker Compose |
| Testes | pytest, pytest-cov |

### Referências Bibliográficas

- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval* ([disponível online](https://nlp.stanford.edu/IR-book/))
- [Documentação NLTK](https://www.nltk.org/)
- [Documentação scikit-learn](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RepositóriUM — UMinho](https://repositorium.uminho.pt/)

### Recursos de Aprendizagem

- [Pro Git Book](https://git-scm.com/book/en/v2) — fluxo de trabalho com Git
- [GitHub Guides](https://guides.github.com/) — GitHub e colaboração
- [Interactive Git Tutorial](https://learngitbranching.js.org/) — prática interativa

---

## 9. Boas Práticas a Seguir

### ✅ Faça

- Crie *branches* para cada funcionalidade (`feature/nome-da-funcionalidade`);
- Escreva mensagens de *commit* claras e descritivas;
- Reveja o código dos colegas através de *Pull Requests*;
- Documente o código com docstrings e comentários relevantes;
- Atualize o repositório com frequência a partir do repositório do docente;
- Escreva testes antes ou durante o desenvolvimento das funcionalidades.

## 10. Dúvidas e Suporte

1. **Leia a documentação** — consulte o [`COMTRIBUTING.md`](COMTRIBUTING.md) e a pasta `docs/`;
2. **Pesquise nas Internet** — a sua dúvida pode já ter resposta;
3. **Pergunte à equipa** — colabore com os colegas;
4. **GitHub Discussions** — use para questões gerais e anúncios;
5. **Aulas e horas de atendimento** — para questões mais complexas, fale com o docente.

---

<div align="center">

**Bom trabalho a todos! 🚀**

*"A melhor forma de aprender Recuperação de Informação é construindo um motor de pesquisa."*

**Universidade do Minho — Escola de Engenharia**

</div>
