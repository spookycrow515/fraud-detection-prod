# model.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow

DATA_PATH = Path(__file__).parent / "creditcard.csv"
TEMP_DATA_PATH = Path(__file__).parent / "temp.csv"

# Global configurations mapping both models and strategies
ALGORITHMS = ["Random Forest", "XGBoost"]
STRATEGIES = ["None", "class_weight", "SMOTE"]

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def get_model_path(algo: str, strategy: str) -> Path:
    clean_algo = algo.lower().replace(" ", "_")
    return Path(__file__).parent / f"fraud_model_{clean_algo}_{strategy}.pkl"


def generate_and_save_temp_sample(df: pd.DataFrame) -> None:
    """Samples exactly 1,000 records from the source dataset (90% Legit / 10% Fraud)."""
    legit_df = df[df[TARGET_COLUMN] == 0]
    fraud_df = df[df[TARGET_COLUMN] == 1]
    
    legit_sample_size = min(900, len(legit_df))
    fraud_sample_size = min(100, len(fraud_df))
    
    legit_sample = legit_df.sample(n=legit_sample_size, random_state=42)
    fraud_sample = fraud_df.sample(n=fraud_sample_size, random_state=42)
    
    temp_sample_df = pd.concat([legit_sample, fraud_sample]).sample(frac=1, random_state=42)
    temp_sample_df.to_csv(TEMP_DATA_PATH, index=False)


def train_single_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          algo: str, strategy: str) -> dict:
    """Trains a chosen algorithm (Random Forest or XGBoost) with a specific balancing strategy."""
    
    # 1. Handle Data Balancing (SMOTE)
    if strategy == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # 2. Instantiate Base Classifiers
    if algo == "Random Forest":
        class_weight = "balanced" if strategy == "class_weight" else None
        model = RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight=class_weight, random_state=42, n_jobs=-1
        )
    elif algo == "XGBoost":
        scale_pos_weight = None
        if strategy == "class_weight":
            num_neg = (y_train_resampled == 0).sum()
            num_pos = (y_train_resampled == 1).sum()
            scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
            
        model = XGBClassifier(
            n_estimators=100, max_depth=6, scale_pos_weight=scale_pos_weight,
            learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="logloss"
        )

    # 3. Fit Model
    model.fit(X_train_resampled, y_train_resampled)

    # 4. Gather Metrics
    y_pred = model.predict(X_test)
    fraud_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "train_rows": len(X_train_resampled),
        "test_rows": len(X_test),
        "fraud_rate": float(y_train_resampled.mean()),
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test.to_numpy(),
        "fraud_proba": fraud_proba,
    }


def execute_full_pipeline(df: pd.DataFrame) -> dict:
    """Loops through all algorithms and strategies, trains them, and logs to MLflow cleanly."""
    # Set the permanent experiment name space
    mlflow.set_experiment("Credit_Card_Fraud_Production")
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    all_artifacts = {}

    for algo in ALGORITHMS:
        all_artifacts[algo] = {}
        for strategy in STRATEGIES:
            # Open EXACTLY ONE run per model execution sequence
            run_name = f"{algo.replace(' ', '_')}_{strategy}"
            
            with mlflow.start_run(run_name=run_name):
                # Train the model
                artifacts = train_single_pipeline(X_train, X_test, y_train, y_test, algo, strategy)
                
                # Log metrics/parameters directly to MLflow server registers
                mlflow.log_param("algorithm", algo)
                mlflow.log_param("balancing_strategy", strategy)
                mlflow.log_param("train_samples", artifacts["metrics"]["train_rows"])
                mlflow.log_metric("f1_score_fraud", artifacts["metrics"]["f1_fraud"])
                
                # Package and serialize locally
                save_package = {
                    "model": artifacts["model"],
                    "metrics": artifacts["metrics"],
                    "y_test": artifacts["y_test"],
                    "fraud_proba": artifacts["fraud_proba"]
                }
                joblib.dump(save_package, get_model_path(algo, strategy))
                
                all_artifacts[algo][strategy] = artifacts

    return all_artifacts


def load_all_saved_models() -> dict:
    """Loads all pre-trained pipeline models directly from pkl files."""
    all_artifacts = {}
    for algo in ALGORITHMS:
        all_artifacts[algo] = {}
        for strategy in STRATEGIES:
            path = get_model_path(algo, strategy)
            if not path.exists():
                return {}  # Signal incomplete structure
            all_artifacts[algo][strategy] = joblib.load(path)
    return all_artifacts


def get_or_train_all_models(cleaned_df: pd.DataFrame = None) -> dict:
    """Orchestrates loading cached files or running full active matrix execution builds."""
    if DATA_PATH.exists() and cleaned_df is not None:
        generate_and_save_temp_sample(cleaned_df)
        return execute_full_pipeline(cleaned_df)
    else:
        loaded_models = load_all_saved_models()
        if not loaded_models:
            raise FileNotFoundError("Missing both creditcard_2023.csv and pre-trained pickle models.")
        return loaded_models