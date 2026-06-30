# api.py
import contextlib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

class TransactionPayload(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

ml_models_registry = {}

# Clean Lifespan context manager instead of @app.on_event
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Warming up local model binaries into RAM cache...")
    from model_io import ALGORITHMS, STRATEGIES, get_model_path
    for algo in ALGORITHMS:
        ml_models_registry[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if path.exists():
                try:
                    ml_models_registry[algo][strategy] = joblib.load(path)
                except Exception as e:
                    print(f"⚠️ Failed to load model file at {path}: {e}")
    yield
    print("🛑 Draining memory caches and shutting down API routing...")


app = FastAPI(title="Fraud Inference Engine", version="1.1.0", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Credit Card Fraud Detection API is Live & Operational!",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    count = sum(len(ml_models_registry[a]) for a in ml_models_registry)
    return {"status": "Healthy", "models_cached_count": count}


@app.post("/predict")
def predict_transaction(
    payload: TransactionPayload,
    algo: str = Query("Random Forest"),
    strategy: str = Query("None"),
    threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    if algo not in ml_models_registry or strategy not in ml_models_registry[algo]:
        raise HTTPException(status_code=404, detail=f"Requested setup '{algo} ({strategy})' not cached.")
        
    cached = ml_models_registry[algo][strategy]
    
    # Run rapid in-memory scoring without nested network bloat
    input_df = pd.DataFrame([payload.model_dump()])
    
    # Safeguard format differences between bare models and structure dictionaries
    model_obj = cached["model"] if isinstance(cached, dict) and "model" in cached else cached
    
    try:
        fraud_prob = float(model_obj.predict_proba(input_df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")
    
    return {
        "prediction": "Fraud" if fraud_prob >= threshold else "Legit",
        "fraud_probability": fraud_prob,
        "applied_threshold": threshold,
        "model_used": f"{algo} ({strategy})"
    }


@app.get("/models")
def get_leaderboard_matrix():
    leaderboard = []
    for algo in ml_models_registry:
        for strategy, data in ml_models_registry[algo].items():
            # Gracefully handle missing evaluation data to avoid 500 runtime crashes
            metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
            leaderboard.append({
                "algorithm": algo,
                "strategy": strategy,
                "f1_score_fraud": metrics.get("f1_fraud", metrics.get("f1_score_fraud", 0.0)),
                "train_rows": metrics.get("train_rows", "N/A"),
                "test_rows": metrics.get("test_rows", "N/A")
            })
    return sorted(leaderboard, key=lambda x: x.get("f1_score_fraud", 0.0), reverse=True)