"""TF-IDF Models for Single-Response Text Classification.

Architecture:
Response text -> TfidfVectorizer -> Classifier -> KNOWN / AMBIGUOUS / UNKNOWN

Uses sklearn Pipeline to guarantee zero fold leakage.
"""

from typing import Dict, Any, Callable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_tfidf_vectorizer() -> TfidfVectorizer:
    """Canonical TF-IDF vectorizer configuration requested by specification."""
    return TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )

def get_tfidf_pipelines() -> Dict[str, Callable[[], Pipeline]]:
    """Return dictionary of factory functions creating TF-IDF pipelines."""
    return {
        "TFIDF_LogisticRegression": lambda: Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "TFIDF_LinearSVC": lambda: Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0)),
        ]),
        "TFIDF_RandomForest": lambda: Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ]),
        "TFIDF_XGBoost": lambda: Pipeline([
            ("tfidf", get_tfidf_vectorizer()),
            ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="mlogloss")),
        ]),
    }
