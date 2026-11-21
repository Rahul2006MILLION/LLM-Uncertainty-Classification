"""Configurable model factory supporting baseline and proposed classifiers."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: List[str] = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
]


class MajorityBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Predicts majority class of training data, providing class probability distribution."""

    def __init__(self):
        self.majority_class_: int = 0
        self.class_priors_: np.ndarray = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        self.classes_: np.ndarray = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(y) == 0:
            return self
        vals, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(vals[np.argmax(counts)])
        priors = np.zeros(3, dtype=np.float64)
        for v, c in zip(vals, counts):
            if 0 <= v < 3:
                priors[v] = c / len(y)
        self.class_priors_ = priors
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.class_priors_, (X.shape[0], 1))


class ModelFactory:
    """Instantiates configured models with fixed seeds and isolated preprocessing."""

    @staticmethod
    def create_model(
        model_name: str,
        random_seed: int = 42,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params = dict(hyperparameters or {})
        name = model_name.lower().strip()

        if name == "majority_baseline":
            return MajorityBaselineClassifier()

        elif name == "logistic_regression":
            # Feature scaling fitted ONLY on training fold inside scikit-learn Pipeline
            c_val = params.get("C", 1.0)
            max_iter = params.get("max_iter", 1000)
            solver = params.get("solver", "lbfgs")
            
            lr = LogisticRegression(
                C=c_val,
                max_iter=max_iter,
                solver=solver,
                random_state=random_seed,
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", lr),
            ])

        elif name == "random_forest":
            n_est = params.get("n_estimators", 100)
            max_d = params.get("max_depth", 5)
            min_samples_split = params.get("min_samples_split", 2)
            
            return RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_samples_split,
                random_state=random_seed,
            )

        elif name == "hist_gradient_boosting":
            max_iter = params.get("max_iter", 100)
            lr_val = params.get("learning_rate", 0.05)
            max_d = params.get("max_depth", 4)

            return HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=lr_val,
                max_depth=max_d,
                random_state=random_seed,
            )

        elif name == "xgboost":
            try:
                import xgboost as xgb
                n_est = params.get("n_estimators", 100)
                lr_val = params.get("learning_rate", 0.05)
                max_d = params.get("max_depth", 4)
                subsample = params.get("subsample", 0.8)

                return xgb.XGBClassifier(
                    n_estimators=n_est,
                    learning_rate=lr_val,
                    max_depth=max_d,
                    subsample=subsample,
                    random_state=random_seed,
                    eval_metric="mlogloss",
                    verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost not installed; falling back to HistGradientBoostingClassifier")
                return ModelFactory.create_model("hist_gradient_boosting", random_seed, params)

        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Must be one of {SUPPORTED_MODELS}"
            )
