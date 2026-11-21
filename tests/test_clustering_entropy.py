"""Unit tests for semantic clustering and entropy calculations."""

import math
import pytest
import numpy as np
from src.features.entropy import compute_semantic_entropy_metrics
from src.features.clustering import SemanticClusterer


class TestEntropyCalculations:
    """Rigorous mathematical tests for entropy calculations and edge cases."""

    def test_single_cluster_consensus(self):
        """Complete consensus: all responses in 1 cluster -> H = 0, dominant = 1.0."""
        res = compute_semantic_entropy_metrics(cluster_counts=[10], total_samples=10)
        assert res["semantic_entropy"] == pytest.approx(0.0, abs=1e-6)
        assert res["normalized_entropy"] == pytest.approx(0.0, abs=1e-6)
        assert res["dominant_agreement"] == pytest.approx(1.0, abs=1e-6)
        assert res["second_agreement"] == pytest.approx(0.0, abs=1e-6)
        assert res["agreement_margin"] == pytest.approx(1.0, abs=1e-6)
        assert res["cluster_count"] == 1.0

    def test_maximum_dispersion_uniform(self):
        """Maximum dispersion: 10 clusters of 1 item -> H = ln(10), H_norm = 1.0."""
        res = compute_semantic_entropy_metrics(cluster_counts=[1] * 10, total_samples=10)
        expected_h = math.log(10)
        assert res["semantic_entropy"] == pytest.approx(expected_h, rel=1e-4)
        assert res["normalized_entropy"] == pytest.approx(1.0, abs=1e-6)
        assert res["dominant_agreement"] == pytest.approx(0.1, abs=1e-6)
        assert res["second_agreement"] == pytest.approx(0.1, abs=1e-6)
        assert res["agreement_margin"] == pytest.approx(0.0, abs=1e-6)
        assert res["cluster_count"] == 10.0

    def test_balanced_two_clusters(self):
        """Two equal clusters of 5 -> H = ln(2) approx 0.69315, FSD = 0."""
        res = compute_semantic_entropy_metrics(cluster_counts=[5, 5], total_samples=10)
        expected_h = math.log(2)
        assert res["semantic_entropy"] == pytest.approx(expected_h, rel=1e-4)
        assert res["normalized_entropy"] == pytest.approx(expected_h / math.log(10), rel=1e-4)
        assert res["dominant_agreement"] == pytest.approx(0.5, abs=1e-6)
        assert res["second_agreement"] == pytest.approx(0.5, abs=1e-6)
        assert res["agreement_margin"] == pytest.approx(0.0, abs=1e-6)

    def test_empty_input(self):
        """Zero samples or empty cluster counts handle gracefully without division by zero."""
        res = compute_semantic_entropy_metrics(cluster_counts=[], total_samples=0)
        assert res["semantic_entropy"] == 0.0
        assert res["normalized_entropy"] == 0.0
        assert res["dominant_agreement"] == 0.0


class TestSemanticClusterer:
    """Unit tests for SemanticClusterer."""

    def test_identical_embeddings_form_single_cluster(self):
        """N identical embeddings must form exactly 1 cluster with dominant agreement 1.0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.vstack([vec] * 5)
        responses = ["Tbilisi"] * 5

        clusterer = SemanticClusterer(similarity_threshold=0.80)
        result = clusterer.cluster_responses(responses, embeddings)

        assert len(result["cluster_counts"]) == 1
        assert result["cluster_counts"][0] == 5
        assert result["dominant_representative_idx"] == 0
        assert result["second_representative_idx"] is None

    def test_orthogonal_embeddings_form_distinct_clusters(self):
        """Orthogonal vectors (similarity = 0.0) must form separate clusters."""
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        embeddings = np.vstack([e1, e2, e3])
        responses = ["Alpha", "Beta", "Gamma"]

        clusterer = SemanticClusterer(similarity_threshold=0.80)
        result = clusterer.cluster_responses(responses, embeddings)

        assert len(result["cluster_counts"]) == 3
        assert result["cluster_counts"] == [1, 1, 1]
