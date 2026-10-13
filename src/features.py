"""Leakage-aware TF-IDF and metadata feature pipeline for financial text."""

from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

TEXT_FEATURE = "text"
CATEGORICAL_FEATURES = ["source", "topic", "sector"]
NUMERICAL_FEATURES = ["sarcasm_flag", "misinformation_flag"]
FEATURE_HUMAN_NAMES = {"text": "Financial language", "sarcasm_flag": "Sarcasm cue", "misinformation_flag": "Verification warning"}


class FinancialFeaturePipeline:
    """Fit text and metadata transforms on training documents only."""

    def __init__(self, max_features: int = 2500):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "FinancialFeaturePipeline":
        self.vectorizer.fit(df[TEXT_FEATURE].fillna(""))
        self.encoder.fit(df[CATEGORICAL_FEATURES].fillna("Unknown"))
        self.feature_names = [f"word:{name}" for name in self.vectorizer.get_feature_names_out()] + [f"meta:{name}" for name in self.encoder.get_feature_names_out(CATEGORICAL_FEATURES)] + NUMERICAL_FEATURES
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise ValueError("Pipeline has not been fitted yet. Call fit() first.")
        text_matrix = self.vectorizer.transform(df[TEXT_FEATURE].fillna(""))
        metadata = self.encoder.transform(df[CATEGORICAL_FEATURES].fillna("Unknown"))
        numeric = np.asarray(df[NUMERICAL_FEATURES].fillna(0), dtype=float)
        return hstack([text_matrix, metadata, csr_matrix(numeric)], format="csr")

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.feature_names
