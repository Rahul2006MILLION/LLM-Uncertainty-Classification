"""Supervised ML Classifier for Uncertainty Classification.

Predicts whether a question's response status is:
- KNOWN
- AMBIGUOUS
- UNKNOWN

Supports XGBoost with fallback to RandomForest. Strictly reports
'CLASSIFIER NOT TRAINED' if no validated trained model exists, preventing
fabricated research results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from configs.config import DEFAULT_PATH_CONFIG
from src.classification.dataset import (
    DatasetExample,
    FEATURE_COLUMNS,
    VALID_LABELS,
)
from src.evaluation.metrics import evaluate_predictions


LABEL_TO_INT = {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


class UncertaintyClassifier:
    """Supervised classifier for predicting uncertainty state from NLP & entropy features."""

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self._is_trained = False
        self.feature_names = list(FEATURE_COLUMNS)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _features_to_matrix(self, examples: List[DatasetExample]) -> np.ndarray:
        matrix = []
        for ex in examples:
            row = [float(ex.features.get(f, 0.0)) for f in self.feature_names]
            matrix.append(row)
        return np.array(matrix, dtype=np.float32)

    def fit(
        self,
        train_examples: List[DatasetExample],
        val_examples: Optional[List[DatasetExample]] = None,
    ) -> Dict[str, Any]:
        """Fit the classifier on labeled training examples."""
        if not train_examples:
            raise ValueError("Cannot train classifier on empty dataset.")

        X_train = self._features_to_matrix(train_examples)
        y_train = np.array([LABEL_TO_INT[ex.label] for ex in train_examples], dtype=int)

        clf = None
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                )
                clf.fit(X_train, y_train)
            except Exception:
                # Fallback to Random Forest
                self.model_type = "random_forest"

        if clf is None:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
            )
            clf.fit(X_train, y_train)

        self.model = clf
        self._is_trained = True

        eval_report = {}
        if val_examples:
            X_val = self._features_to_matrix(val_examples)
            y_val_preds = clf.predict(X_val)
            y_val_true = [ex.label for ex in val_examples]
            y_val_pred_labels = [INT_TO_LABEL[int(p)] for p in y_val_preds]
            eval_report = evaluate_predictions(
                y_true=y_val_true,
                y_pred=y_val_pred_labels,
                classes=sorted(list(VALID_LABELS)),
            )

        return eval_report

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict uncertainty class for a single feature dictionary.

        If the model is not trained on validated data, returns 'CLASSIFIER NOT TRAINED'.
        """
        if not self._is_trained or self.model is None:
            return {
                "prediction": "CLASSIFIER NOT TRAINED",
                "probabilities": {},
                "confidence": "uncalibrated (model not trained on labeled data)",
                "is_trained": False,
            }

        x = np.array([[float(features.get(f, 0.0)) for f in self.feature_names]], dtype=np.float32)
        pred_idx = int(self.model.predict(x)[0])
        pred_label = INT_TO_LABEL[pred_idx]

        probs_dict = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            for idx, p in enumerate(probs):
                if idx in INT_TO_LABEL:
                    probs_dict[INT_TO_LABEL[idx]] = float(p)
            top_prob = float(probs[pred_idx])
            confidence_str = f"{top_prob:.1%} (model predicted probability; uncalibrated)"
        else:
            confidence_str = "uncalibrated"

        return {
            "prediction": pred_label,
            "probabilities": probs_dict,
            "confidence": confidence_str,
            "is_trained": True,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        if not self._is_trained or self.model is None:
            raise ValueError("Cannot save an untrained classifier.")

        if path is None:
            path = DEFAULT_PATH_CONFIG.models_dir / "uncertainty_classifier.joblib"

        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({"model": self.model, "model_type": self.model_type, "feature_names": self.feature_names}, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "UncertaintyClassifier":
        """Load trained model from disk if available."""
        if path is None:
            path = DEFAULT_PATH_CONFIG.models_dir / "uncertainty_classifier.joblib"

        instance = cls()
        if not path.exists():
            return instance

        try:
            import joblib
            data = joblib.load(path)
            instance.model = data["model"]
            instance.model_type = data.get("model_type", "xgboost")
            instance.feature_names = data.get("feature_names", list(FEATURE_COLUMNS))
            instance._is_trained = True
        except Exception:
            instance._is_trained = False

        return instance
