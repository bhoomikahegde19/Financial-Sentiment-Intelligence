"""Temporal, class-balanced evaluation for sentiment classification."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.features import FinancialFeaturePipeline
from src.models import predict_sentiment_probabilities, train_classifier


def evaluate_predictions(y_true, probabilities) -> Dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    return {"accuracy": round(float(accuracy_score(y_true, predictions)), 4), "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4), "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted", zero_division=0)), 4), "report": classification_report(y_true, predictions, output_dict=True, zero_division=0)}


def run_out_of_time_evaluation(df: pd.DataFrame, train_years: Tuple[int, ...] = (2022, 2023, 2024), test_years: Tuple[int, ...] = (2025, 2026), model_name: str = "Logistic Regression", use_smote: bool = False) -> Dict[str, Any]:
    train_df = df[df.year.isin(train_years)].copy()
    test_df = df[df.year.isin(test_years)].copy()
    pipeline = FinancialFeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    labels = {label: index for index, label in enumerate(["Negative", "Neutral", "Positive"])}
    y_train = train_df.sentiment_label.map(labels).to_numpy()
    y_test = test_df.sentiment_label.map(labels).to_numpy()
    model = train_classifier(model_name, X_train, y_train, use_smote=use_smote)
    train_probs = predict_sentiment_probabilities(model, X_train)
    test_probs = predict_sentiment_probabilities(model, X_test)
    return {"model_name": model_name, "train_years": list(train_years), "test_years": list(test_years), "n_train": len(train_df), "n_test": len(test_df), "train_metrics": evaluate_predictions(y_train, train_probs), "oot_test_metrics": evaluate_predictions(y_test, test_probs), "fitted_pipeline": pipeline, "fitted_model": model}


def run_stress_test(df_test, pipeline, model, optimal_threshold=None, **kwargs) -> pd.DataFrame:
    probabilities = predict_sentiment_probabilities(model, pipeline.transform(df_test))
    labels = {label: index for index, label in enumerate(["Negative", "Neutral", "Positive"])}
    metrics = evaluate_predictions(df_test.sentiment_label.map(labels).to_numpy(), probabilities)
    return pd.DataFrame([{"noise_level": 0.0, "missing_rate": 0.0, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}])
