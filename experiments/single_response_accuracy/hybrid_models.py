"""Hybrid Single-Response Models (Feature Fusion).

Architecture:
Sentence Embedding (384-d) + Single-Response Linguistic Features -> Scaler -> Classifier
"""

from typing import Dict, Any, Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_hybrid_classifiers() -> Dict[str, Callable[[], Any]]:
    """Return dictionary of factory functions creating classifiers for fused feature matrices."""
    return {
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]),
        "LinearSVC": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0)),
        ]),
        "RandomForest": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ]),
        "XGBoost": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="mlogloss")),
        ]),
    }
