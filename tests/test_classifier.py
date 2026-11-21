"""Unit tests for UncertaintyClassifier, training, serialization, and untrained status."""

import os
import tempfile
import pytest
import numpy as np
from src.classification.classifier import UncertaintyClassifier, LABEL_TO_ID, ID_TO_LABEL
from src.features.feature_extractor import FEATURE_NAMES


class TestUncertaintyClassifier:
    """Tests model training, serialization, inference, and honest untrained reporting."""

    def test_untrained_status(self):
        """Untrained classifier must report 'CLASSIFIER NOT TRAINED' honestly."""
        clf = UncertaintyClassifier()
        pred = clf.predict({"semantic_entropy": 0.5})
        assert pred["status"] == "CLASSIFIER NOT TRAINED"
        assert pred["predicted_label"] is None
        assert pred["confidence"] == 0.0

    def test_training_and_inference_xgboost(self):
        """Train classifier with synthetic fixtures for testing infrastructure."""
        clf = UncertaintyClassifier(model_type="xgboost", random_state=42)
        n_samples = 30
        n_feats = len(FEATURE_NAMES)
        np.random.seed(42)
        X = np.random.randn(n_samples, n_feats).astype(np.float32)
        y = np.random.choice([0, 1, 2], size=n_samples)

        metadata = clf.train(X, y, feature_names=FEATURE_NAMES)
        assert clf.is_trained
        assert metadata["n_samples"] == n_samples

        sample_feat = {fn: float(np.random.rand()) for fn in FEATURE_NAMES}
        pred = clf.predict(sample_feat)
        assert pred["status"] == "PREDICTED"
        assert pred["predicted_label"] in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_save_and_load_roundtrip(self):
        """Trained classifier can be serialized to disk and reloaded with identical predictions."""
        clf = UncertaintyClassifier(model_type="hist_gb", random_state=42)
        X = np.random.randn(20, len(FEATURE_NAMES)).astype(np.float32)
        y = np.random.choice([0, 1, 2], size=20)
        clf.train(X, y, feature_names=FEATURE_NAMES)

        sample_feat = {fn: 0.5 for fn in FEATURE_NAMES}
        orig_pred = clf.predict(sample_feat)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_clf.joblib")
            clf.save(save_path)
            loaded_clf = UncertaintyClassifier.load(save_path)

            loaded_pred = loaded_clf.predict(sample_feat)
            assert loaded_pred["predicted_label"] == orig_pred["predicted_label"]
            assert loaded_pred["confidence"] == pytest.approx(orig_pred["confidence"], abs=1e-5)
