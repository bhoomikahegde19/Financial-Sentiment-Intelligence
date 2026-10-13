"""Human-readable feature and document explanations for sentiment predictions."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features import FEATURE_HUMAN_NAMES


def get_global_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.mean(np.abs(getattr(model, "coef_", np.ones((1, len(feature_names))))), axis=0)
    frame = pd.DataFrame({"feature": feature_names, "importance_pct": importances[:len(feature_names)]})
    total = frame.importance_pct.sum()
    frame["importance_pct"] = frame.importance_pct / total * 100 if total else 0
    frame["human_name"] = frame.feature.map(lambda name: FEATURE_HUMAN_NAMES.get(name, name.replace("word:", "").replace("meta:", "").replace("_", " ").title()))
    return frame.sort_values("importance_pct", ascending=False).head(top_n).reset_index(drop=True)


def explain_document(text: str, predicted_label: str, confidence: float, top_n: int = 5) -> List[Dict[str, Any]]:
    positive = {"strong", "improved", "growth", "resilient", "raises", "welcomed", "secured", "accelerating"}
    negative = {"weaker", "delayed", "pressure", "fell", "cut", "concerns", "risks", "slowing", "liquidity"}
    results = []
    for token in [token.strip(".,'\"").lower() for token in text.split()]:
        if token in positive or token in negative:
            direction = "supports positive sentiment" if token in positive else "supports negative sentiment"
            results.append({"feature": token, "human_name": "Influential phrase", "direction": direction, "impact_score": round(confidence, 3), "narrative": f"The phrase '{token}' {direction}."})
    if not results:
        results.append({"feature": "context", "human_name": "Document context", "direction": "low evidence", "impact_score": 0.0, "narrative": "No high-signal sentiment phrase was detected; review the source manually."})
    return results[:top_n]


def explain_company_risk(model: Any, X_single, feature_names: List[str], raw_feature_dict: Dict[str, Any], top_n: int = 6):
    return explain_document(str(raw_feature_dict.get("text", "")), str(raw_feature_dict.get("sentiment_label", "Neutral")), 0.5, top_n)
