"""Multiclass sentiment models and confidence helpers."""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def train_classifier(model_name: str, X_train, y_train, use_smote: bool = False, random_state: int = 42) -> Any:
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=180, max_depth=10, class_weight="balanced", random_state=random_state, n_jobs=2)
    elif model_name == "XGBoost":
        model = XGBClassifier(n_estimators=140, max_depth=5, learning_rate=0.06, eval_metric="mlogloss", random_state=random_state, n_jobs=2)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, use_smote: bool = False, random_state: int = 42) -> Dict[str, Any]:
    return {name: train_classifier(name, X_train, y_train, use_smote, random_state) for name in MODEL_NAMES}


def predict_sentiment_probabilities(model: Any, X) -> np.ndarray:
    return np.clip(model.predict_proba(X), 0.0, 1.0)


def predict_crisis_probabilities(model: Any, X) -> np.ndarray:
    return predict_sentiment_probabilities(model, X)[:, 0]


def sentiment_confidence(probabilities: np.ndarray) -> np.ndarray:
    return np.max(probabilities, axis=1)


def sentiment_index(label: str, confidence: float) -> float:
    polarity = {"Negative": -1, "Neutral": 0, "Positive": 1}[label]
    return round(50 + polarity * confidence * 50, 1)


def compute_risk_score(probability: float) -> float:
    return round(float(probability) * 100, 1)


def assign_risk_tier(risk_score: float):
    return {"tier": "Negative" if risk_score >= 60 else "Neutral" if risk_score >= 30 else "Positive", "status": "Sentiment polarity"}
