"""Unit tests for semantic clustering and empirical semantic entropy."""

import math
import unittest
import numpy as np

from src.features.semantic_analysis import (
    SemanticCluster,
    calculate_semantic_entropy,
    perform_semantic_clustering,
    compute_lexical_statistics,
)


class TestClusteringAndEntropy(unittest.TestCase):
    def test_entropy_complete_consensus(self):
        # 1 cluster containing all 10 responses
        clusters = [
            SemanticCluster(
                cluster_id=0,
                member_indices=list(range(10)),
                size=10,
                proportion=1.0,
                representative_index=0,
                representative_text="Sample text",
            )
        ]
        raw_entropy, norm_entropy = calculate_semantic_entropy(clusters, total_responses=10)
        self.assertEqual(raw_entropy, 0.0)
        self.assertEqual(norm_entropy, 0.0)

    def test_entropy_maximum_dispersion(self):
        # 10 singleton clusters
        clusters = [
            SemanticCluster(
                cluster_id=i,
                member_indices=[i],
                size=1,
                proportion=0.1,
                representative_index=i,
                representative_text=f"Sample text {i}",
            )
            for i in range(10)
        ]
        raw_entropy, norm_entropy = calculate_semantic_entropy(clusters, total_responses=10)
        expected_raw = math.log(10)
        self.assertAlmostEqual(raw_entropy, expected_raw, places=4)
        self.assertAlmostEqual(norm_entropy, 1.0, places=4)

    def test_entropy_two_balanced_clusters(self):
        # 2 equal clusters of 5 responses each
        clusters = [
            SemanticCluster(
                cluster_id=0,
                member_indices=[0, 1, 2, 3, 4],
                size=5,
                proportion=0.5,
                representative_index=0,
                representative_text="Cluster A",
            ),
            SemanticCluster(
                cluster_id=1,
                member_indices=[5, 6, 7, 8, 9],
                size=5,
                proportion=0.5,
                representative_index=5,
                representative_text="Cluster B",
            ),
        ]
        raw_entropy, norm_entropy = calculate_semantic_entropy(clusters, total_responses=10)
        expected_raw = math.log(2)
        self.assertAlmostEqual(raw_entropy, expected_raw, places=4)
        self.assertTrue(0.0 < norm_entropy < 1.0)

    def test_semantic_clustering_block_matrix(self):
        # 4 responses: first two identical (sim=1.0), second two identical (sim=1.0), across groups sim=0.1
        sim_matrix = np.array([
            [1.0, 0.95, 0.1, 0.1],
            [0.95, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.95],
            [0.1, 0.1, 0.95, 1.0],
        ], dtype=np.float32)

        texts = ["Atlanta is capital", "The capital is Atlanta", "Tbilisi is capital", "Capital is Tbilisi"]
        clusters = perform_semantic_clustering(sim_matrix, texts, similarity_threshold=0.80)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].size, 2)
        self.assertEqual(clusters[1].size, 2)
        c1_members = set(clusters[0].member_indices)
        c2_members = set(clusters[1].member_indices)
        self.assertTrue(
            (c1_members == {0, 1} and c2_members == {2, 3}) or
            (c1_members == {2, 3} and c2_members == {0, 1})
        )

    def test_lexical_statistics(self):
        texts = ["Hello world", "Hello there beautiful world"]
        stats = compute_lexical_statistics(texts)
        self.assertGreater(stats["mean_length_words"], 0)
        self.assertTrue(0.0 <= stats["lexical_diversity"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
