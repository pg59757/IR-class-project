# Research Publication Search System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive web application for searching and discovering research publications and authors affiliated with University of Minho. This project combines web scraping, natural language processing, and modern web technologies to provide an intuitive search experience for academic content.

Project inspired by the [Coventry-PureHub-Search-Engine](https://github.com/maladeep/Coventry-PureHub-Search-Engine/tree/main?tab=readme-ov-file#features)

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Technology Stack](#technology-stack)
- [Team Organization](#team-organization)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Testing](#testing)
- [License](#license)

---

## 🎯 Overview

This project is a collaborative educational initiative that implements a full-stack research publication search system. The system scrapes academic publications from [UMinho repositories](https://repositorium.uminho.pt/), processes them using NLP techniques, and provides a user-friendly web interface for searching and exploring the content.

**Key Capabilities:**
- 🔍 Advanced search with natural language processing
- 📊 TF-IDF-based ranking for relevant results
- 👤 Author profiles and publication tracking
- 🚀 Fast inverse indexer for quick searches
- 📱 Responsive web interface
- 🐳 Containerized deployment

---

## ✨ Features

### Data Collection
- **Web Scraper**: Automated collection from [UMinho repositories](https://repositorium.uminho.pt/)
- **Metadata Extraction**: Title, authors, abstract, DOI, publication date
- **Pagination Handling**: Efficient navigation through large collections
- **Error Recovery**: Robust handling of network issues and timeouts

### Search & Processing
- **Natural Language Processing**:
  - Text stemming and lemmatization
  - Stop word removal
  - TF-IDF vectorization
- **Inverse Indexer**: Fast keyword-based search
- **Ranking Algorithm**: Relevance-based result ordering
- **Fuzzy Matching**: Handles typos and variations

### User Interface
- **Search Interface**: Clean, intuitive search experience
- **Filters**: Year, author, department, publication type
- **Author Profiles**: Detailed author information and statistics
- **Publication Details**: Full metadata display
- **Responsive Design**: Works on desktop, tablet, and mobile

### DevOps & Infrastructure
- **Docker Support**: Containerized application
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Logging and performance metrics
- **Documentation**: Comprehensive guides and API docs

---

## 📁 Project Structure (EXAMPLE)

```
UMinho-scraper/
├── src/                        # Source code
│   ├── scraper/               # Web scraping module
│   │   ├── scraper.py        # Main scraper class
│   │   └── utils.py          # Helper functions
│   ├── search/               # Search engine
│   │   ├── indexer.py       # Inverse indexer
│   │   ├── tfidf.py         # TF-IDF implementation
│   │   └── nlp.py           # NLP utilities
│   ├── api/                 # Backend API
│   │   ├── app.py           # FastAPI application
│   │   └── routes/          # API endpoints
│   └── frontend/            # Frontend
│       ├── src/
│       └── public/
│
├── tests/                    # Test suite
│   ├── test_scraper.py
│   ├── test_search.py
│   └── test_api.py
│
├── docs/                     # Documentation
│   ├── architecture.md
│   ├── api-reference.md
│   └── deployment.md
│
├── docker/                   # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/                  # GitHub configurations
│   ├── workflows/           # CI/CD pipelines
│   └── PULL_REQUEST_TEMPLATE.md
│
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Development dependencies
├── .gitignore
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Chrome/Chromium** browser (for Selenium)
- **Docker** (optional, for containerized deployment)

### Quick Start

#### 1. Fork the Repository

**Team leads:** Follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md) to fork this repository to your team's account.

#### 2. Clone Your Fork

```bash
# Clone your team's fork
git clone https://github.com/YOUR_TEAM/IR-class-project.git
cd IR-class-project
```

#### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 4. Run the Scraper

```bash
# Run scraper with default settings
python main.py
```

#### 6. Run Tests  (TODO)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_scraper.py -v
```
---

## 🤝 Contributing

We welcome contributions from all team members! Please read our detailed contribution guidelines:

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

---

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory..

---

## 🎓 Learning Resources

### Git & GitHub
- [Pro Git Book](https://git-scm.com/book/en/v2) (Free)
- [GitHub Guides](https://guides.github.com/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

### Python
- [Python Official Docs](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Testing with pytest](https://docs.pytest.org/)

### Web Development
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Docker Documentation](https://docs.docker.com/)

### NLP & Search
- [NLTK Book](https://www.nltk.org/book/)
- [Elasticsearch Guide](https://www.elastic.co/guide/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## 📞 Getting Help

1. **Read the docs** - Check [CONTRIBUTING.md](CONTRIBUTING.md) and `docs/` folder
2. **Search Issues** - Your question may already be answered
3. **Ask Your Team** - Collaborate with teammates first
4. **GitHub Discussions** - Use for general questions
5. **Office Hours** - Attend instructor's office hours for complex issues


---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ by University of Minho Students**

</div>
