"""Unit tests for UncertaintyClassifier logic, state reporting, and persistence."""

import tempfile
import unittest
from pathlib import Path

from src.classification.dataset import DatasetExample
from src.classification.classifier import UncertaintyClassifier


class TestUncertaintyClassifier(unittest.TestCase):
    def test_untrained_classifier_reports_not_trained(self):
        clf = UncertaintyClassifier()
        self.assertFalse(clf.is_trained)

        result = clf.predict({"majority_cluster_agreement": 0.9})
        self.assertEqual(result["prediction"], "CLASSIFIER NOT TRAINED")
        self.assertIn("uncalibrated", result["confidence"])
        self.assertFalse(result["is_trained"])

    def test_train_and_predict_random_forest_baseline(self):
        clf = UncertaintyClassifier(model_type="random_forest", random_state=42)

        # Create balanced synthetic examples for training
        examples = []
        for i in range(15):
            examples.append(DatasetExample(
                question=f"Q_known_{i}",
                label="KNOWN",
                features={"majority_cluster_agreement": 1.0, "raw_semantic_entropy": 0.0},
            ))
            examples.append(DatasetExample(
                question=f"Q_amb_{i}",
                label="AMBIGUOUS",
                features={"majority_cluster_agreement": 0.5, "raw_semantic_entropy": 0.69},
            ))
            examples.append(DatasetExample(
                question=f"Q_unk_{i}",
                label="UNKNOWN",
                features={"majority_cluster_agreement": 0.2, "raw_semantic_entropy": 1.6},
            ))

        report = clf.fit(train_examples=examples)
        self.assertTrue(clf.is_trained)

        # Test predict
        pred_known = clf.predict({"majority_cluster_agreement": 1.0, "raw_semantic_entropy": 0.0})
        self.assertTrue(pred_known["is_trained"])
        self.assertIn(pred_known["prediction"], ["KNOWN", "AMBIGUOUS", "UNKNOWN"])
        self.assertIn("KNOWN", pred_known["probabilities"])

    def test_save_and_load(self):
        clf = UncertaintyClassifier(model_type="random_forest", random_state=42)
        examples = [
            DatasetExample(question="Q1", label="KNOWN", features={"majority_cluster_agreement": 1.0}),
            DatasetExample(question="Q2", label="AMBIGUOUS", features={"majority_cluster_agreement": 0.5}),
            DatasetExample(question="Q3", label="UNKNOWN", features={"majority_cluster_agreement": 0.2}),
        ]
        clf.fit(examples)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.joblib"
            clf.save(save_path)
            self.assertTrue(save_path.exists())

            loaded_clf = UncertaintyClassifier.load(save_path)
            self.assertTrue(loaded_clf.is_trained)
            res = loaded_clf.predict({"majority_cluster_agreement": 1.0})
            self.assertTrue(res["is_trained"])


if __name__ == "__main__":
    unittest.main()
