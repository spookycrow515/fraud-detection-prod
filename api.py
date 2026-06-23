# api.py
import contextlib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
import mlflow

# Define schema contract
class TransactionPayload(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float

# Global production memory state
ml_models_registry = {}

# 1. NEW: Replace deprecated on_event with enterprise lifespan management
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup model deserialization and connection pool bootstrapping.
    Warms up cache layers inside system memory RAM for rapid inference execution.
    """
    print("⏳ Initializing FastAPI production infrastructure layers...")
    
    # Connect and lock down your MLflow baseline tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Bootstrap your models from your local workspace files
    # (Maintains fallback data pipelines cleanly)
    from model import ALGORITHMS, STRATEGIES, get_model_path
    
    for algo in ALGORITHMS:
        ml_models_registry[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if path.exists():
                ml_models_registry[algo][strategy] = joblib.load(path)
                print(f"📦 Successfully warmed up: {algo} ({strategy}) into RAM.")
            else:
                print(f"⚠️ Warning: Pre-trained model matrix file missing at {path}")
                
    yield # Execution boundary partition (Server handles traffic here)
    
    # Cleanup tasks on shutdown (if any) can go here
    print("🛑 Draining connections and shutting down microservice routing...")

# Initialize the primary application app shell utilizing our lifespan manager
app = FastAPI(
    title="Enterprise Fraud Inference Microservice", 
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/health", tags=["Telemetry"])
def health_check():
    """Confirms network health and checks internal memory footprint registers."""
    return {
        "status": "ok", 
        "models_cached_count": sum(len(ml_models_registry[a]) for a in ml_models_registry)
    }

@app.post("/predict", tags=["Inference Engine"])
def predict_transaction(
    payload: TransactionPayload,
    algo: str = Query("Random Forest", description="Algorithm selection"),
    strategy: str = Query("None", description="Balancing methodology option"),
    threshold: float = Query(0.50, ge=0.0, le=1.0, description="Dynamic cutoff boundary criteria")
):
    """
    Intercepts standard JSON payment frames and runs validation.
    Applies custom dynamic user threshold controls to calculate predictions.
    """
    if algo not in ml_models_registry or strategy not in ml_models_registry[algo]:
        raise HTTPException(status_code=404, detail="Requested pipeline variant not registered.")
        
    cached_package = ml_models_registry[algo][strategy]
    trained_model = cached_package["model"]
    
    # Reshape the Pydantic dictionary fields into a single DataFrame row array
    input_data = pd.DataFrame([payload.model_dump()])
    
    # Execute inference using warmed up weights in RAM
    try:
        fraud_probability = float(trained_model.predict_proba(input_data)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model scoring fault: {str(e)}")
        
    # Apply dynamic decision boundary threshold overriding
    prediction_label = "Fraud" if fraud_probability >= threshold else "Legit"
    
    # 🚀 OPTIONAL: Log this validation check directly to MLflow for absolute audit logs
    try:
        with mlflow.start_run(run_name=f"Inference_Audit_{algo.replace(' ', '_')}", nested=True):
            mlflow.log_param("applied_threshold", threshold)
            mlflow.log_metric("transaction_amount", payload.Amount)
            mlflow.log_metric("fraud_probability", fraud_probability)
    except Exception:
        pass # Gracefully fall back if MLflow server tab isn't processing requests
        
    return {
        "prediction": prediction_label,
        "fraud_probability": fraud_probability,
        "applied_threshold": threshold,
        "model_used": f"{algo} ({strategy})"
    }

@app.get("/models", tags=["Telemetry"])
def get_leaderboard_matrix():
    """Scans cached local artifacts to feed the Streamlit client window."""
    leaderboard = []
    from model import ALGORITHMS, STRATEGIES
    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            if algo in ml_models_registry and strategy in ml_models_registry[algo]:
                metrics = ml_models_registry[algo][strategy]["metrics"]
                leaderboard.append({
                    "algorithm": algo,
                    "strategy": strategy,
                    "f1_score_fraud": metrics["f1_fraud"],
                    "train_rows": metrics["train_rows"],
                    "test_rows": metrics["test_rows"]
                })
    # Sort by F1-Score descending before sending across the wire interface
    return sorted(leaderboard, key=lambda x: x["f1_score_fraud"], reverse=True)