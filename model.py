# model.py
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

from model_io import ALGORITHMS, STRATEGIES, get_model_path, FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = Path(__file__).parent / "creditcard.csv"
TEMP_DATA_PATH = Path(__file__).parent / "temp.csv"


def train_single_pipeline(X_train, X_test, y_train, y_test, algo: str, strategy: str):
    if strategy == "SMOTE":
        X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    if algo == "Random Forest":
        cw = "balanced" if strategy == "class_weight" else None
        model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight=cw, random_state=42, n_jobs=-1)
    else: # XGBoost
        spw = (y_train_res == 0).sum() / (y_train_res == 1).sum() if strategy == "class_weight" else None
        model = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=spw, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss")

    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    
    metrics = {
        "f1_fraud": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "precision_fraud": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "recall_fraud": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "train_rows": len(X_train_res),
        "test_rows": len(X_test)
    }
    return model, metrics


def execute_full_pipeline(df: pd.DataFrame):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    use_mlflow = False
    
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Credit_Card_Fraud_Production")
            use_mlflow = True
            print(f"📡 MLflow Tracking enabled pointing to: {tracking_uri}")
        except Exception as e:
            print(f"⚠️ Failed to connect to MLflow URI. Running pipeline offline. Error: {e}")
    else:
        print("⚠️ No MLFLOW_TRACKING_URI found in environment variables. Running pipeline offline.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_size=0.3, random_state=42, stratify=df[TARGET_COLUMN]
    )

    best_f1 = -1
    best_run_id = None

    for algo in ALGORITHMS:
        for strategy in STRATEGIES:
            run_name = f"{algo.replace(' ', '_')}_{strategy}"
            
            # Use active tracking session context only if setup succeeded
            if use_mlflow:
                ctx = mlflow.start_run(run_name=run_name)
            else:
                ctx = contextlib.nullcontext() if hasattr(contextlib, 'nullcontext') else contextlib.suppress()

            # Handle dual tracking pipeline branches cleanly
            if use_mlflow:
                with ctx as run:
                    model, metrics = train_single_pipeline(X_train, X_test, y_train, y_test, algo, strategy)
                    mlflow.log_params({"algorithm": algo, "balancing_strategy": strategy, "n_estimators": 100})
                    mlflow.log_metrics({k: v for k, v in metrics.items()})
                    mlflow.sklearn.log_model(model, "model")
                    
                    # Store standard dictionary contract structure for the caching layer
                    joblib.dump({"model": model, "metrics": metrics}, get_model_path(algo, strategy))
                    
                    if metrics["f1_fraud"] > best_f1:
                        best_f1 = metrics["f1_fraud"]
                        best_run_id = run.info.run_id
            else:
                model, metrics = train_single_pipeline(X_train, X_test, y_train, y_test, algo, strategy)
                joblib.dump({"model": model, "metrics": metrics}, get_model_path(algo, strategy))
                if metrics["f1_fraud"] > best_f1:
                    best_f1 = metrics["f1_fraud"]

    # Auto-register champion model weights only if remote logging tracking registry is active
    if use_mlflow and best_run_id:
        try:
            model_uri = f"runs:/{best_run_id}/model"
            model_name = "Credit_Card_Fraud_Classifier"
            reg_version = mlflow.register_model(model_uri, model_name)
            
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", reg_version.version)
            print(f"🏆 Champion model promoted to registry: Version {reg_version.version} with F1: {best_f1:.4f}")
        except Exception as e:
            print(f"⚠️ Registry connection dropped. Champion cached locally but not registered online: {e}")
    else:
        print(f"🏆 Pipeline execution complete. Local performance champion F1-Score: {best_f1:.4f}")


if __name__ == "__main__":
    import contextlib
    if DATA_PATH.exists():
        print(f"📊 Dataset located. Starting model execution pipeline...")
        execute_full_pipeline(pd.read_csv(DATA_PATH))
    else:
        print(f"❌ Error: Raw data file missing at '{DATA_PATH}'. Cannot execute pipeline.")