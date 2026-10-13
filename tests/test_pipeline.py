"""Focused tests for the financial sentiment intelligence pipeline."""

import numpy as np
import pytest

from src.data_generator import generate_sentiment_dataset
from src.features import FinancialFeaturePipeline
from src.models import predict_sentiment_probabilities, sentiment_confidence, train_all_models
from src.explainability import explain_document
from src.evaluation import run_out_of_time_evaluation


@pytest.fixture
def sample_data():
    return generate_sentiment_dataset(n_documents=360, seed=42)


def test_text_corpus_has_temporal_and_quality_fields(sample_data):
    assert len(sample_data) == 360
    assert {"text", "publication_date", "source", "sentiment_label", "sarcasm_flag", "misinformation_flag"}.issubset(sample_data.columns)
    assert set(sample_data.sentiment_label.unique()) == {"Positive", "Neutral", "Negative"}


def test_feature_pipeline_is_numeric_and_leakage_aware(sample_data):
    train = sample_data[sample_data.year < 2025]
    pipeline = FinancialFeaturePipeline()
    matrix = pipeline.fit_transform(train)
    assert matrix.shape[0] == len(train)
    assert len(pipeline.get_feature_names()) == matrix.shape[1]
    assert not np.isnan(matrix.toarray()).any()


def test_models_return_three_class_probabilities(sample_data):
    pipeline = FinancialFeaturePipeline()
    matrix = pipeline.fit_transform(sample_data)
    labels = sample_data.sentiment_label.map({"Negative": 0, "Neutral": 1, "Positive": 2}).to_numpy()
    for model in train_all_models(matrix, labels, use_smote=False).values():
        probabilities = predict_sentiment_probabilities(model, matrix)
        assert probabilities.shape == (len(sample_data), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert (sentiment_confidence(probabilities) >= 0.0).all()


def test_document_explanation_has_evidence():
    explanation = explain_document("The company reports improved growth but faces delayed projects.", "Neutral", 0.72)
    assert explanation
    assert "narrative" in explanation[0]


def test_temporal_evaluation(sample_data):
    result = run_out_of_time_evaluation(sample_data, model_name="Logistic Regression")
    assert result["train_years"] == [2022, 2023, 2024]
    assert result["oot_test_metrics"]["macro_f1"] >= 0.35
