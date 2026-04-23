import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Create a fixture that provides the test client, which handles the lifespan startup/shutdown
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "version" in data
    assert "stage" in data
    assert "f1_score" in data

def test_preprocess(client):
    payload = {
        "text": "This is a REALLY bad FAKE news tweet! http://fake.com",
        "remove_stopwords": True,
        "stem": True,
        "min_len": 3
    }
    response = client.post("/preprocess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original" in data
    assert "tokens" in data
    assert "processing_time_ms" in data
    assert len(data["tokens"]) > 0

def test_classify_latency_and_response(client):
    payload = {"text": "URGENT! Drinking bleached water cures COVID globally!"}
    
    start_time = time.perf_counter()
    response = client.post("/classify", json=payload)
    runtime = (time.perf_counter() - start_time) * 1000  # ms
    
    assert response.status_code == 200
    assert runtime < 100.0, f"Latency {runtime:.2f}ms exceeds 100ms limit"
    
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "class_probabilities" in data
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) <= 5

def test_classify_edge_case_validation(client):
    # Too short (<10 chars)
    response = client.post("/classify", json={"text": "Short"})
    assert response.status_code == 422

def test_classify_batch_latency(client):
    # Create batch of 10 texts
    batch_payload = {"texts": [f"This is a relatively safe claim number {i} for testing" for i in range(10)]}
    
    start_time = time.perf_counter()
    response = client.post("/classify/batch", json=batch_payload)
    runtime = (time.perf_counter() - start_time) * 1000
    
    assert response.status_code == 200
    assert runtime < 200.0, f"Batch Latency {runtime:.2f}ms exceeds 200ms limit"
    
    data = response.json()
    assert data["batch_size"] == 10
    assert len(data["results"]) == 10

def test_retrieve_similar(client):
    payload = {
        "text": "COVID vaccine causes mutations",
        "top_k": 3
    }
    response = client.post("/retrieve/similar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_claims" in data
    assert len(data["top_claims"]) == 3
    assert "similarity_score" in data["top_claims"][0]

def test_retrieve_validation(client):
    # top_k>20 should fail
    payload = {"text": "valid text", "top_k": 25}
    response = client.post("/retrieve/similar", json=payload)
    assert response.status_code == 422
    
def test_model_performance(client):
    response = client.get("/model/performance")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "history" in data
    assert isinstance(data["history"], list)
