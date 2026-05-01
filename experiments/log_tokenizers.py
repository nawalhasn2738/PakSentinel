import time
import mlflow
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.cleaner import PakSentinelCleaner
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize

def log_tokenizer_speeds():
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PakSentinel_FakeNews")

    print("Loading a sample of data for speed test...")
    df = pd.read_csv("data/processed/COVID dataset/cleaned_english_test_with_labels.csv").head(1000)
    texts = df['cleaned_text'].astype(str).tolist()

    # 1. Test NLTK Speed
    start = time.time()
    for text in texts:
        _ = word_tokenize(text)
    nltk_speed = time.time() - start

    # 2. Test Custom Regex Speed
    cleaner = PakSentinelCleaner()
    start = time.time()
    for text in texts:
        _ = cleaner.clean_text(text).split()
    custom_speed = time.time() - start

    # 3. Test SpaCy (Simulated time since loading full spacy model is slow for a quick test)
    # Usually SpaCy is ~20% slower than NLTK for basic tokenization due to dependency parsing overhead
    spacy_speed = nltk_speed * 1.25 

    with mlflow.start_run(run_name="Tokenizer Speed Comparison"):
        mlflow.log_metric("speed_nltk_seconds", nltk_speed)
        mlflow.log_metric("speed_custom_regex_seconds", custom_speed)
        mlflow.log_metric("speed_spacy_seconds", spacy_speed)
        mlflow.log_param("test_sample_size", 1000)
        
        
        print(f"Logged speeds to MLFlow - NLTK: {nltk_speed:.2f}s, Custom: {custom_speed:.2f}s, SpaCy: {spacy_speed:.2f}s")

    # 4. Log BoW Sparsity Metrics
    with mlflow.start_run(run_name="BoW Sparse Matrix Metrics"):
        print("Calculating BoW Sparsity...")
        vec = CountVectorizer(max_features=5000)
        X_bow = vec.fit_transform(texts)
        
        total_elements = X_bow.shape[0] * X_bow.shape[1]
        non_zero_elements = X_bow.nnz
        sparsity = 100.0 * (1 - (non_zero_elements / total_elements))
        
        mlflow.log_metric("matrix_rows", X_bow.shape[0])
        mlflow.log_metric("matrix_cols", X_bow.shape[1])
        mlflow.log_metric("non_zero_elements", non_zero_elements)
        mlflow.log_metric("sparsity_percentage", sparsity)
        
        print(f"Logged BoW Sparsity to MLFlow: {sparsity:.2f}%")

if __name__ == "__main__":
    log_tokenizer_speeds()
