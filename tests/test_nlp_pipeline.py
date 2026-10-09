"""Unit tests for NLP semantic pipeline, embeddings, and feature extraction."""

import unittest
import numpy as np
from unittest.mock import patch

from src.features.semantic_analysis import (
    analyze_semantic_responses,
    compute_question_ambiguity_features,
)
from src.features.embeddings import cosine_similarity_matrix


class TestNLPPipeline(unittest.TestCase):
    def test_empty_responses(self):
        result = analyze_semantic_responses([])
        self.assertEqual(result.num_responses, 0)
        self.assertEqual(result.num_clusters, 0)
        self.assertEqual(result.majority_cluster_agreement, 0.0)
        self.assertEqual(result.raw_semantic_entropy, 0.0)

    def test_single_response(self):
        # 1 response
        mock_embedding = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = analyze_semantic_responses(
            responses=["4"],
            question="What is 2+2?",
            embeddings=mock_embedding,
        )
        self.assertEqual(result.num_responses, 1)
        self.assertEqual(result.num_clusters, 1)
        self.assertEqual(result.majority_cluster_agreement, 1.0)
        self.assertEqual(result.raw_semantic_entropy, 0.0)
        self.assertEqual(result.normalized_semantic_entropy, 0.0)

    def test_ten_responses_two_clusters(self):
        # 5 of Atlanta (vector [1, 0]), 5 of Tbilisi (vector [0, 1])
        embeddings = np.array(
            [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5,
            dtype=np.float32,
        )
        responses = ["Atlanta"] * 5 + ["Tbilisi"] * 5
        result = analyze_semantic_responses(
            responses=responses,
            question="What is the capital of Georgia?",
            embeddings=embeddings,
            similarity_threshold=0.80,
        )

        self.assertEqual(result.num_responses, 10)
        self.assertEqual(result.num_clusters, 2)
        self.assertAlmostEqual(result.majority_cluster_agreement, 0.50)
        self.assertAlmostEqual(result.second_cluster_agreement, 0.50)
        self.assertAlmostEqual(result.agreement_margin, 0.0)
        self.assertTrue(0.0 < result.normalized_semantic_entropy < 1.0)
        self.assertIn("question_word_count", result.features)
        self.assertIn("normalized_semantic_entropy", result.features)

    def test_cosine_similarity_properties(self):
        emb = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        sim = cosine_similarity_matrix(emb)
        self.assertAlmostEqual(sim[0, 0], 1.0)
        self.assertAlmostEqual(sim[0, 1], 1.0)
        self.assertAlmostEqual(sim[0, 2], 0.0)

    def test_question_ambiguity_independence(self):
        # Verify question ambiguity feature contains no label leakage
        feat = compute_question_ambiguity_features("Is it A or B?")
        self.assertEqual(feat["question_has_disjunction"], 1.0)
        self.assertNotIn("label", feat)
        self.assertNotIn("target", feat)


if __name__ == "__main__":
    unittest.main()
