# PakSentinel: NLP Fake News Engine

**PakSentinel** is a production-ready Natural Language Processing (NLP) pipeline and FastAPI inference system designed to classify and fact-check news and social media claims (identifying Fake vs. Real news).

##  Project Architecture

This project was built step-by-step to satisfy rigorous NLP engineering standards, moving from raw data ingestion to a fully tracked, rate-limited FastAPI backend.

### Task 1: Data Sourcing & Reliability
The underlying models are trained on curated datasets (specifically COVID-19 Fake News and ISOT dataset structures). 
- **Justification**: Using the COVID-19 dataset provided a highly reliable, binary-class (Fake/Real) foundation with minimal extreme class imbalance. It represents a highly relevant domain where misinformation poses direct real-world harm, making it an excellent candidate for fact-checking NLP engines.

### Task 2: Data Storage Architecture
- **Lakehouse Design**: The project strictly compartmentalizes data into `data/raw/` and `data/processed/`.
- **Justification**: A custom `DataLakeManager` ensures that original raw data is immutable. Cleaned records, TF-IDF vocabularies, Word2Vec matrices, and serialized models are versioned in later stages.

### Task 3: NLP Processing Pipeline
The core intelligence engine uses custom NLP heuristics:
- **Cleaning**: Strips HTML, URLs, and noisy emojis using BeautifulSoup and Regex.
- **Normalization & Stopwords**: 
  - **Justification**: While NLTK provides standard stopwords, removing words like "not" fundamentally destroys sentiment/negation context in fake news (e.g., "Vaccines do not work"). Our custom domain logic preserves negation words while stripping useless filler.
  - **Stemming vs Lemmatization**: Through ablation, aggressive stemming destroyed contextual meaning, whereas Lemmatization safely consolidated vocabulary size without sacrificing semantic integrity.
- **Vectorization**: Implements TF-IDF for baseline inference and Word2Vec (Skip-gram) for semantic relationships.

### Task 4 & 5: Machine Learning Classification
Implemented Unigrams/Bigrams/Trigrams alongside core ML algorithms:
- **Multinomial Naive Bayes (Laplace Smoothed)**: Built entirely from scratch.
- **Logistic Regression (L1/L2 Regularization)**:
  - **Justification**: Logistic Regression fundamentally outperforms Naive Bayes on this dataset because TF-IDF feature tokens (words) are incredibly correlated. Naive Bayes assumes strict conditional independence between words, which is statistically false in human language structure. LR natively handles these correlations via weights and Ridge (L2) penalties, resulting in higher F1-scores.

### Task 6: MLFlow Experiment Tracking
Integrated an automated MLFlow tracking loop (`mlflow.db`).
- **Ablation Study**: Tested 6 distinct preprocessing configurations (No Stopwords, Stemming only, etc.) logging training times, F1 thresholds, and automatically archiving Confusion Matrices and ROC curves.
- **Promotion Logic**: The registry transitions staging models to `Production` *only* if they demonstrably beat the current production model's F1-Weighted score by at least +1.0%.

### Task 7: FastAPI Inference
A lightweight, lightning-fast web backend (`src/api`).
- **Endpoints**: `/health`, `/preprocess`, `/classify`, `/classify/batch`, `/retrieve/similar`, and `/model/performance`.
- **Justifications & Protections**:
  - **Pydantic**: Blocks invalid requests (e.g., missing text) ensuring the ML models don't crash from garbage data.
  - **SlowAPI**: Rate limits standard classifications to `100/minute` to prevent DDoS.
  - **MLFlow Lifespan**: Loads the production machine learning model *once* during startup into RAM, driving batch classification latency comfortably under `< 200ms`.

## ⚙️ How to Run Locally

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI Server:
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```
3. Open `http://127.0.0.1:8000/docs` in your browser to view the interactive API playground.
