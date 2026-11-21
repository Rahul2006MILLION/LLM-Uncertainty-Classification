"""Uncertainty classifier for KNOWN, AMBIGUOUS, and UNKNOWN categories."""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

# Label mappings
LABEL_TO_ID = {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 2}
ID_TO_LABEL = {0: "KNOWN", 1: "AMBIGUOUS", 2: "UNKNOWN"}

logger = logging.getLogger(__name__)


class UncertaintyClassifier:
    """Supervised classifier predicting query uncertainty type from consistency features."""

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}

    def _init_underlying_model(self):
        """Instantiate XGBoost or scikit-learn fallback model."""
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                    verbosity=0,
                )
            except Exception as e:
                logger.warning("XGBoost instantiation failed (%s), falling back to HistGradientBoosting", e)
                self.model_type = "hist_gb"

        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train the classifier on extracted feature matrix X and numeric labels y."""
        self.feature_names = list(feature_names)
        self.model = self._init_underlying_model()
        self.model.fit(X, y)
        self.is_trained = True
        self.metadata = metadata or {}
        self.metadata.update({
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        })
        return self.metadata

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Predict uncertainty category from a feature dictionary.
        
        If classifier is not trained, returns honest 'CLASSIFIER NOT TRAINED' status.
        """
        if not self.is_trained or self.model is None:
            return {
                "status": "CLASSIFIER NOT TRAINED",
                "predicted_label": None,
                "probabilities": None,
                "confidence": 0.0,
            }

        # Align features according to trained feature_names
        x_vec = np.array([[feature_dict.get(fname, 0.0) for fname in self.feature_names]], dtype=np.float32)

        try:
            pred_id = int(self.model.predict(x_vec)[0])
            probs = self.model.predict_proba(x_vec)[0].tolist()
            predicted_label = ID_TO_LABEL.get(pred_id, "UNKNOWN")
            confidence = float(probs[pred_id])

            prob_dict = {ID_TO_LABEL[i]: float(p) for i, p in enumerate(probs)}

            return {
                "status": "PREDICTED",
                "predicted_label": predicted_label,
                "predicted_class_id": pred_id,
                "probabilities": prob_dict,
                "confidence": confidence,
            }
        except Exception as e:
            logger.error("Inference failed: %s", e)
            return {
                "status": f"ERROR: {str(e)}",
                "predicted_label": None,
                "probabilities": None,
                "confidence": 0.0,
            }

    def save(self, filepath: str) -> None:
        """Serialize model and metadata to disk.
        
        Saves the model artifact (via joblib) and exports a human-readable metadata JSON alongside it.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained classifier.")

        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_dict = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "metadata": self.metadata,
        }
        joblib.dump(save_dict, filepath)
        logger.info("Classifier binary saved to %s", filepath)

        # Export accompanying human-readable JSON metadata file
        meta_path = os.path.splitext(filepath)[0] + "_metadata.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_type": self.model_type,
                    "feature_names": self.feature_names,
                    "is_trained": self.is_trained,
                    "metadata": self.metadata,
                }, f, indent=2)
            logger.info("Human-readable metadata exported to %s", meta_path)
        except Exception as e:
            logger.warning("Could not write metadata JSON: %s", e)

    @classmethod
    def load(cls, filepath: str) -> "UncertaintyClassifier":
        """Load a trained classifier from disk, resolving .joblib and .json paths cleanly."""
        import joblib

        target_path = filepath
        if not os.path.exists(target_path):
            # Check alternative extensions
            base = os.path.splitext(filepath)[0]
            for alt in [f"{base}.joblib", f"{base}.json", f"{base}.pkl"]:
                if os.path.exists(alt):
                    target_path = alt
                    break

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Model file not found at {filepath} (checked alternatives)")

        data = joblib.load(target_path)
        clf = cls(model_type=data.get("model_type", "xgboost"))
        clf.model = data["model"]
        clf.feature_names = data.get("feature_names", [])
        clf.is_trained = data.get("is_trained", True)
        clf.metadata = data.get("metadata", {})
        return clf
