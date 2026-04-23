import os
import time
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder
import sys

# Ensure parent directory is in path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.logistic_regression import PakSentinelLogReg

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def ensure_nltk():
    for req in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{req}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{req}')
            except LookupError:
                nltk.download(req, quiet=True)

def apply_preprocessing(text, remove_stopwords=False, stem=False, lemma=False, min_len=0):
    """Task 6: Dynamic Preprocessing for Ablation"""
    if not isinstance(text, str): return ""
    tokens = word_tokenize(text.lower())
    
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stops]
        
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    elif lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
    if min_len > 0:
        tokens = [t for t in tokens if len(t) >= min_len]
        
    return " ".join(tokens)

def run_ablation():
    ensure_nltk()
    
    # 1. Setup MLFlow Tracking Environment
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PakSentinel_FakeNews")
    
    print("Loading datasets...")
    df = pd.read_csv("data/processed/COVID dataset/cleaned_english_test_with_labels.csv")
    df = df.dropna(subset=['cleaned_text'])
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['label']) # Encode real vs fake
    
    # Task 6: 6 Ablation Configurations
    configs = [
        {"name": "Config 1 - Base", "remove_stopwords": False, "stem": False, "lemma": False, "min_len": 0, "max_features": 5000},
        {"name": "Config 2 - No Stopwords", "remove_stopwords": True, "stem": False, "lemma": False, "min_len": 0, "max_features": 5000},
        {"name": "Config 3 - Stemming", "remove_stopwords": True, "stem": True, "lemma": False, "min_len": 0, "max_features": 5000},
        {"name": "Config 4 - Lemmatization", "remove_stopwords": True, "stem": False, "lemma": True, "min_len": 0, "max_features": 5000},
        {"name": "Config 5 - TF-IDF Max 1000", "remove_stopwords": True, "stem": False, "lemma": True, "min_len": 0, "max_features": 1000},
        {"name": "Config 6 - Min Token Len = 3", "remove_stopwords": True, "stem": False, "lemma": True, "min_len": 3, "max_features": 5000},
    ]

    for config in configs:
        with mlflow.start_run(run_name=config["name"]):
            print(f"\n--- Running: {config['name']} ---")
            
            # Apply text transformation
            start_time = time.time()
            df['processed_text'] = df['cleaned_text'].apply(
                lambda x: apply_preprocessing(x, config['remove_stopwords'], config['stem'], config['lemma'], config['min_len'])
            )
            
            # Vectorization
            vec = TfidfVectorizer(max_features=config['max_features'])
            X = vec.fit_transform(df['processed_text'])
            y = df['target'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Logistic Regression (Chosen over NaiveBayes due to native sklearn MLflow support)
            model_wrapper = PakSentinelLogReg(C=1.0)
            model_wrapper.train_variants(X_train, y_train)
            model = model_wrapper.models['l2'] # We focus on Ridge
            
            train_time = time.time() - start_time
            
            # Evaluation
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            roc_auc = auc(*roc_curve(y_test, y_proba)[:2])
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

            print(f"Accuracy: {acc:.4f} | F1-Weighted: {f1_weighted:.4f}")
            
            # MLFlow Parameter Logging
            mlflow.log_param("dataset", "COVID_english_test")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("remove_stopwords", config["remove_stopwords"])
            mlflow.log_param("stemming", config["stem"])
            mlflow.log_param("lemmatization", config["lemma"])
            mlflow.log_param("min_len", config["min_len"])
            mlflow.log_param("vectorizer_max_features", config["max_features"])
            mlflow.log_param("model_type", "LogisticRegression_L2")
            
            # MLFlow Metrics Logging
            mlflow.log_metric("training_time_seconds", train_time)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1_weighted)
            mlflow.log_metric("roc_auc", roc_auc)
            
            for class_name in le.classes_:
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
                
            # MLFlow Artifact Logging (Plots and Reports)
            os.makedirs("temp_artifacts", exist_ok=True)
            
            with open("temp_artifacts/classification_report.txt", "w") as f:
                f.write(classification_report(y_test, y_pred, target_names=le.classes_))
            mlflow.log_artifact("temp_artifacts/classification_report.txt")
            
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{config['name']} Confusion Matrix")
            plt.savefig("temp_artifacts/confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("temp_artifacts/confusion_matrix.png")
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig("temp_artifacts/roc_curve.png")
            plt.close()
            mlflow.log_artifact("temp_artifacts/roc_curve.png")
            
            pd.DataFrame(list(vec.vocabulary_.keys()), columns=['word']).to_csv("temp_artifacts/tfidf_vocab.csv", index=False)
            mlflow.log_artifact("temp_artifacts/tfidf_vocab.csv")
            
            # Registering the model in MLFlow formats
            mlflow.sklearn.log_model(model, "model")

    print("\n--- Ablation Study Complete. Run 'mlflow ui' to view tracking dashboard ---")

if __name__ == "__main__":
    run_ablation()
