"""Unit tests for dataset loading, label definitions, and data leakage checks."""

import unittest

from src.classification.dataset import (
    DatasetExample,
    DatasetValidationError,
    validate_examples,
    check_split_leakage,
    split_dataset,
    compute_class_distribution,
    VALID_LABELS,
)


class TestDatasetValidation(unittest.TestCase):
    def test_valid_dataset_examples(self):
        examples = [
            DatasetExample(
                question="What is 2+2?",
                label="KNOWN",
                features={"majority_cluster_agreement": 1.0, "raw_semantic_entropy": 0.0},
            ),
            DatasetExample(
                question="What is the capital of Georgia?",
                label="AMBIGUOUS",
                features={"majority_cluster_agreement": 0.5, "raw_semantic_entropy": 0.69},
            ),
            DatasetExample(
                question="What will be the GDP of Mars in 2050?",
                label="UNKNOWN",
                features={"majority_cluster_agreement": 0.2, "raw_semantic_entropy": 1.8},
            ),
        ]
        # Should validate without error
        validate_examples(examples)

        dist = compute_class_distribution(examples)
        self.assertEqual(dist["KNOWN"]["count"], 1)
        self.assertEqual(dist["AMBIGUOUS"]["count"], 1)
        self.assertEqual(dist["UNKNOWN"]["count"], 1)

    def test_invalid_label_raises_error(self):
        examples = [
            DatasetExample(
                question="Question 1",
                label="INVALID_LABEL",
                features={"majority_cluster_agreement": 0.8},
            )
        ]
        with self.assertRaises(DatasetValidationError):
            validate_examples(examples)

    def test_feature_label_leakage_detected(self):
        # If a feature column contains the target label or ground truth
        examples = [
            DatasetExample(
                question="Question 1",
                label="KNOWN",
                features={"ground_truth_label": 1.0, "majority_cluster_agreement": 0.8},
            )
        ]
        with self.assertRaises(DatasetValidationError):
            validate_examples(examples)

    def test_split_leakage_detection(self):
        ex_train = [DatasetExample(question="What is Python?", label="KNOWN", features={})]
        ex_val = [DatasetExample(question="What is Python?", label="KNOWN", features={})]
        ex_test = [DatasetExample(question="What is Java?", label="KNOWN", features={})]

        with self.assertRaises(DatasetValidationError):
            check_split_leakage(ex_train, ex_val, ex_test)

    def test_split_dataset_no_leakage(self):
        examples = [
            DatasetExample(question=f"Question {i}", label="KNOWN" if i % 2 == 0 else "AMBIGUOUS", features={})
            for i in range(20)
        ]
        train, val, test = split_dataset(examples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

        # Ensure no question in test is in train or val
        train_q = {e.question for e in train}
        val_q = {e.question for e in val}
        test_q = {e.question for e in test}

        self.assertTrue(train_q.isdisjoint(val_q))
        self.assertTrue(train_q.isdisjoint(test_q))
        self.assertTrue(val_q.isdisjoint(test_q))


if __name__ == "__main__":
    unittest.main()
