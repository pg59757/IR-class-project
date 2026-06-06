# Backend Requirements - University Research Publication Search Engine

## Project Overview
Develop a backend system for searching and analyzing research publications from University of Minho's document repository. 
The system must implement core Information Retrieval techniques for educational purposes.

## 1. Data Collection & Web Scraping

### 1.1 Document Scraper
- **REQ-B01**: Implement web scraper to extract data from university's document pages
- **REQ-B02**: Extract document metadata: title, authors, publication date, keywords
- **REQ-B03**: Extract and store document abstracts
- **REQ-B04**: Capture PDF download links for full documents
- **REQ-B05**: Store author information and affiliations
- **REQ-B06**: Handle pagination and sequential page processing
- **REQ-B07**: Implement configurable collection size parameter (minimum N documents)
- **REQ-B08**: Focus collection on specific research areas (e.g., health, engineering)

### 1.2 Data Storage
- **REQ-B09**: Design database schema for documents, authors, and metadata
- **REQ-B10**: Store raw text content and processed versions
- **REQ-B11**: Maintain document-author relationships
- **REQ-B12**: Store indexing data structures efficiently

## 2. Text Processing & Natural Language Processing

### 2.1 NLTK Implementation
- **REQ-B13**: Integrate NLTK for text preprocessing
- **REQ-B14**: Implement text tokenization and sentence segmentation
- **REQ-B15**: Handle multiple languages (Portuguese/English)

### 2.2 Stemming and Lemmatization
- **REQ-B16**: Implement stemming algorithms (Porter Stemmer recommended)
- **REQ-B17**: Implement lemmatization using NLTK WordNet
- **REQ-B18**: Allow system configuration to choose between stems/lemmas
- **REQ-B19**: Compare performance between stemming vs lemmatization strategies

### 2.3 Stop Words Processing
- **REQ-B20**: Implement configurable stop words filtering
- **REQ-B21**: Allow inclusion/exclusion of stop words in term dictionary
- **REQ-B22**: Support Portuguese and English stop word lists

## 3. Information Retrieval Core Components

### 3.1 Boolean Model & Term-Document Matrix
- **REQ-B23**: Implement Boolean operators (AND, OR, NOT) with precedence rules
- **REQ-B24**: Create term-document incidence matrix
- **REQ-B25**: Process AND operations considering term frequency optimization
- **REQ-B26**: Support space-separated terms as implicit AND operations

### 3.2 Inverted Index
- **REQ-B27**: Build inverted index data structure
- **REQ-B28**: Implement postings lists for each term
- **REQ-B29**: Optimize postings list intersection with skip pointers
- **REQ-B30**: Store term frequencies and document frequencies
- **REQ-B31**: Support incremental index updates

### 3.3 TF-IDF Implementation
- **REQ-B32**: Calculate Term Frequency (TF) scores
- **REQ-B33**: Calculate Inverse Document Frequency (IDF) scores
- **REQ-B34**: Implement custom TF-IDF calculation function
- **REQ-B35**: Integrate sklearn TF-IDF for comparison
- **REQ-B36**: Allow user selection between custom and sklearn implementations

### 3.4 Similarity and Ranking
- **REQ-B37**: Implement cosine similarity calculation
- **REQ-B38**: Rank search results by relevance scores
- **REQ-B39**: Support ranking by different weighting schemes
- **REQ-B40**: Generate document similarity matrices

## 4. Machine Learning Components (TODO)

### 4.1 Document Classification
- **REQ-B41**: Implement Multinomial Naïve Bayes classifier
- **REQ-B42**: Train classifier on research publication categories
- **REQ-B43**: Categorize documents into subject areas automatically
- **REQ-B44**: Evaluate classification performance metrics

## 5. Search API & Query Processing

### 5.1 Query Processing
- **REQ-B45**: Parse complex Boolean queries with operator precedence
- **REQ-B46**: Support search across titles, abstracts, and full documents
- **REQ-B47**: Implement query expansion techniques
- **REQ-B48**: Handle phrase queries and proximity searches

### 5.2 Search Results
- **REQ-B49**: Return ranked results with relevance scores
- **REQ-B50**: Generate text snippets containing query terms
- **REQ-B51**: Provide document access links
- **REQ-B52**: Support different result formats (JSON/XML)

### 5.3 Author Search
- **REQ-B53**: Enable author-specific searches
- **REQ-B54**: Support partial author name matching
- **REQ-B55**: Link authors to their publications

## 6. Performance & Evaluation

### 6.1 Indexing Performance
- **REQ-B56**: Measure and log indexing time performance
- **REQ-B57**: Compare indexing speed: stems vs lemmas
- **REQ-B58**: Monitor memory usage during indexing
- **REQ-B59**: Implement batch processing for large collections

### 6.2 Search Performance
- **REQ-B60**: Measure query response times
- **REQ-B61**: Evaluate search result relevance
- **REQ-B62**: Compare ranking effectiveness across different methods

## 7. Technical Implementation Requirements

### 7.1 API Development
- **REQ-B63**: Design RESTful API endpoints
- **REQ-B64**: Implement proper error handling and status codes
- **REQ-B65**: Add API documentation (OpenAPI/Swagger)
- **REQ-B66**: Include request validation and sanitization

### 7.2 Configuration & Deployment
- **REQ-B67**: Create configuration files for system parameters
- **REQ-B68**: Support different deployment environments
- **REQ-B69**: Implement logging and monitoring capabilities
- **REQ-B70**: Add unit tests for core IR components

## 8. Deliverables

### 8.1 Code Components
- Web scraping module with robust error handling
- Text processing pipeline with NLTK integration
- Inverted index implementation with skip pointers
- TF-IDF calculation (custom + sklearn comparison)
- Boolean query parser and processor
- Cosine similarity ranking system
- Naïve Bayes classification module
- RESTful API with comprehensive endpoints

### 8.2 Analysis Reports
- Performance comparison: stems vs lemmas impact
- Ranking effectiveness analysis across weighting schemes
- Classification accuracy evaluation
- System scalability assessment

### 8.3 Technical Documentation
- API documentation with usage examples
- Database schema documentation
- Algorithm implementation explanations
- Performance optimization recommendations
