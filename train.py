"""Generate the financial text corpus and serialize the sentiment system."""

import os
import joblib
from src.data_generator import generate_sentiment_dataset
from src.evaluation import run_out_of_time_evaluation
from src.explainability import get_global_feature_importance


def run_pipeline():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    data_path = os.path.join("data", "synthetic_financial_sentiment.csv")
    df = generate_sentiment_dataset()
    df.to_csv(data_path, index=False)
    result = run_out_of_time_evaluation(df, model_name="Logistic Regression")
    result["dataset_path"] = data_path
    result["global_importance"] = get_global_feature_importance(result["fitted_model"], result["fitted_pipeline"].get_feature_names())
    artifact_path = os.path.join("models", "sentiment_system.joblib")
    joblib.dump(result, artifact_path)
    print(f"Saved {len(df):,} documents to {data_path}")
    print(f"OOT macro F1: {result['oot_test_metrics']['macro_f1']}")
    print(f"Saved trained system to {artifact_path}")


if __name__ == "__main__":
    run_pipeline()
