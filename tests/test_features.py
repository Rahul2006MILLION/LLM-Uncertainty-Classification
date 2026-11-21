"""Unit tests for feature extraction and label leakage prevention."""

import pytest
import numpy as np
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES


class TestFeatureExtractor:
    """Verifies feature extraction correctness and guarantees zero label leakage."""

    def test_all_expected_features_present(self):
        """Feature dictionary must contain exactly the declared feature names."""
        extractor = FeatureExtractor()
        question = "What is the capital of France?"
        responses = ["Paris is the capital of France.", "The capital is Paris.", "Paris."]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.98, 0.02, 0.0],
        ], dtype=np.float32)

        feats = extractor.extract_features(question, responses, embeddings)
        for fname in FEATURE_NAMES:
            assert fname in feats, f"Missing feature: {fname}"
            assert isinstance(feats[fname], (float, int))
            assert not np.isnan(feats[fname])

    def test_strict_absence_of_label_leakage(self):
        """Feature extractor interface takes NO target labels or expected answers."""
        extractor = FeatureExtractor()
        # Verify method signature: extract_features does not accept any 'label', 'target', or 'gold' parameter
        import inspect
        sig = inspect.signature(extractor.extract_features)
        forbidden_params = {"label", "y", "target", "gold_answer", "expected_answer"}
        assert not any(p in forbidden_params for p in sig.parameters.keys())

    def test_hedging_and_negation_detection(self):
        """Verifies that epistemic hedging and negation tokens are accurately counted."""
        extractor = FeatureExtractor()
        q = "Will it rain?"
        responses = [
            "Perhaps it might rain, but it is unclear and difficult to say.",
            "No, it will not rain and cannot rain.",
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        feats = extractor.extract_features(q, responses, embeddings)

        assert feats["hedging_score"] > 0.0, "Hedging score should detect 'perhaps', 'might', 'unclear'"
        assert feats["negation_count"] > 0.0, "Negation count should detect 'no', 'not', 'cannot'"
