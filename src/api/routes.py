import time
import mlflow
from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from src.api.schemas import PreprocessRequest, ClassifyRequest, BatchClassifyRequest, RetrieveRequest
from src.core.cleaner import PakSentinelCleaner
from src.api.utils import limiter, setup_logging

router = APIRouter()
logger = setup_logging()

@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    state = request.app.state.model_config
    return {
        "status": "healthy",
        "model_name": state.model_name,
        "version": state.model_version,
        "stage": state.model_stage,
        "f1_score": state.model_f1,
        "load_timestamp": state.load_time
    }

@router.post("/preprocess")
async def preprocess_text(req: PreprocessRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    cleaner = PakSentinelCleaner()
    cleaned = cleaner.clean_text(req.text)

    from experiments.ablation_study import apply_preprocessing

    # Use the same preprocessing configuration as the production model
    # Based on ablation study results, Config 4 (Lemmatization) typically performs best
    final_processed = apply_preprocessing(
        cleaned,
        remove_stopwords=True,  # Remove stopwords
        stem=False,            # No stemming
        lemma=True,            # Use lemmatization
        min_len=0              # No minimum length filter
    )

    runtime = time.perf_counter() - start

    return {
        "original": req.text,
        "tokens": final_processed.split(),
        "processing_time_ms": runtime * 1000
    }

@router.post("/classify")
@limiter.limit("100/minute")
async def classify_text(request: Request, req: ClassifyRequest) -> Dict[str, Any]:
    state = request.app.state.model_config
    if not state.model or not state.vectorizer:
        raise HTTPException(status_code=503, detail="Model clearly not loaded yet")

    # Apply consistent preprocessing
    cleaner = PakSentinelCleaner()
    cleaned = cleaner.clean_text(req.text)

    from experiments.ablation_study import apply_preprocessing
    processed_text = apply_preprocessing(
        cleaned,
        remove_stopwords=True,  # Match training preprocessing
        stem=False,
        lemma=True,
        min_len=0
    )
    
    vec_input = state.vectorizer.transform([processed_text])
    
    # Predict
    if hasattr(state.model, 'predict_proba'):
        probs = state.model.predict_proba(vec_input)[0]
        pred = state.model.predict(vec_input)[0]
    else:
        pred = state.model.predict(vec_input)[0]
        probs = [0.0, 0.0]
        probs[pred] = 1.0

    # Handle feature importance for different model types
    feature_impacts = []
    if hasattr(state.model, 'coef_'):  # sklearn-style models
        coeffs = state.model.coef_[0]
        words = state.vectorizer.get_feature_names_out()
        feature_impacts = [(words[i], coeffs[i]) for i in vec_input.nonzero()[1]]
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    else:  # Custom models like PakSentinelNaiveBayes
        # For custom models, we can't easily extract feature importance
        # Return top TF-IDF features from the input text
        words = state.vectorizer.get_feature_names_out()
        tfidf_scores = vec_input.toarray()[0]
        top_indices = tfidf_scores.argsort()[-5:][::-1]  # Top 5 features
        feature_impacts = [(words[i], float(tfidf_scores[i])) for i in top_indices]

    label_map = {0: "fake", 1: "real"}

    logger.info(f"Classified text as {label_map.get(pred, str(pred))}")
    return {
        "text": req.text,
        "prediction": label_map.get(pred, str(pred)),
        "confidence": float(max(probs)),
        "class_probabilities": {"fake": float(probs[0]), "real": float(probs[1])},
        "top_contributing_features": feature_impacts[:5]
    }

@router.post("/classify/batch")
@limiter.limit("10/minute")
async def classify_batch(request: Request, req: BatchClassifyRequest) -> Dict[str, Any]:
    state = request.app.state.model_config
    if not state.model:
        raise HTTPException(status_code=503, detail="Model unavailable")

    # Apply consistent preprocessing to all texts
    cleaner = PakSentinelCleaner()
    from experiments.ablation_study import apply_preprocessing

    processed_texts = []
    for text in req.texts:
        cleaned = cleaner.clean_text(text)
        processed = apply_preprocessing(
            cleaned,
            remove_stopwords=True,  # Match training preprocessing
            stem=False,
            lemma=True,
            min_len=0
        )
        processed_texts.append(processed)

    vec_input = state.vectorizer.transform(processed_texts)
    preds = state.model.predict(vec_input)

    label_map = {0: "fake", 1: "real"}
    results = [{"text": t[:50]+"...", "prediction": label_map.get(p, str(p))} for t, p in zip(req.texts, preds)]

    return {"batch_size": len(req.texts), "results": results}

@router.post("/retrieve/similar")
async def retrieve_similar(request: Request, req: RetrieveRequest) -> Dict[str, Any]:
    state = request.app.state.model_config
    if state.tfidf_matrix is None or state.claims_db is None:
        raise HTTPException(status_code=503, detail="Database uninitialized")
        
    cleaner = PakSentinelCleaner()
    cleaned = cleaner.clean_text(req.text)
    
    from experiments.ablation_study import apply_preprocessing
    processed_text = apply_preprocessing(
        cleaned,
        remove_stopwords=True,
        stem=False,
        lemma=True,
        min_len=0
    )
    
    vec_input = state.vectorizer.transform([processed_text])
    sim_scores = cosine_similarity(vec_input, state.tfidf_matrix).flatten()
    
    top_indices = sim_scores.argsort()[-req.top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": str(state.claims_db.iloc[idx]['tweet']),
            "label": str(state.claims_db.iloc[idx]['label']),
            "similarity_score": float(sim_scores[idx])
        })
        
    return {"query": req.text, "top_claims": results}

@router.get("/model/performance")
async def model_performance() -> Dict[str, Any]:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    
    try:
        model_name = "PakSentinel_Best_LogReg"
        versions = client.search_model_versions(f"name='{model_name}'")
        
        history = []
        for v in versions:
            run = client.get_run(v.run_id)
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "f1_weighted": run.data.metrics.get("f1_weighted", 0.0),
                "accuracy": run.data.metrics.get("accuracy", 0.0)
            })
            
        return {"model": model_name, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
