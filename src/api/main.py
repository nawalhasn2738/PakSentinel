import os
import datetime
import mlflow
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from src.api.routes import router
from src.api.utils import setup_logging, limiter

class AppState:
    model = None
    model_name = ""
    model_version = ""
    model_stage = ""
    model_f1 = 0.0
    load_time = ""
    vectorizer = None
    claims_db = None
    tfidf_matrix = None

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing PakSentinel Lifespan context...")
    AppState.load_time = datetime.datetime.now().isoformat()
    
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        client = mlflow.tracking.MlflowClient()
        
        model_name = "PakSentinel_Best_LogReg"
        prod_versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == "Production"]
        
        if prod_versions:
            prod_version = prod_versions[0]
            AppState.model_name = model_name
            AppState.model_version = prod_version.version
            AppState.model_stage = "Production"
            
            run = client.get_run(prod_version.run_id)
            AppState.model_f1 = run.data.metrics.get("f1_weighted", 0.0)
            
            logger.info(f"Loading Production Model {model_name} (v{prod_version.version})")
            AppState.model = mlflow.sklearn.load_model(f"runs:/{prod_version.run_id}/model")
            
            logger.info("Initializing Vectorizer and Fact-Check Database...")
            df = pd.read_csv("data/processed/COVID dataset/cleaned_english_test_with_labels.csv")
            df = df.dropna(subset=['cleaned_text'])
            
            AppState.vectorizer = TfidfVectorizer(max_features=5000)
            AppState.tfidf_matrix = AppState.vectorizer.fit_transform(df['cleaned_text'])
            AppState.claims_db = df
            
        else:
            logger.warning("No Production model found in MLFlow Registry!")
            
    except Exception as e:
        logger.error(f"Lifespan initialization failed: {e}")
        
    yield
    logger.info("Shutting down PakSentinel...")

app = FastAPI(title="PakSentinel Inference API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.state.model_config = AppState

app.include_router(router)
