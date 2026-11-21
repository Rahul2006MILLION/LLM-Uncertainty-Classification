"""Semantic clustering of LLM responses using cosine similarity."""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.preprocessing.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class SemanticClusterer:
    """Clusters responses based on semantic similarity of their embeddings."""

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        linkage: str = "average",
    ):
        self.similarity_threshold = similarity_threshold
        self.linkage = linkage
        # In distance space: distance = 1.0 - similarity
        self.distance_threshold = max(0.0, 1.0 - similarity_threshold)

    def cluster_responses(
        self,
        responses: List[str],
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """Cluster N responses given their (N, D) normalized embeddings.
        
        Returns a dict containing:
            - 'labels': List[int] (cluster id for each response, 0 = dominant)
            - 'cluster_counts': List[int] (sizes of clusters sorted descending)
            - 'cluster_indices': Dict[int, List[int]] (map cluster id -> response indices)
            - 'dominant_cluster_id': int (always 0)
            - 'dominant_representative_idx': int (index of medoid in dominant cluster)
            - 'second_representative_idx': Optional[int] (index of medoid in 2nd cluster)
            - 'similarity_matrix': np.ndarray
        """
        n_samples = len(responses)
        if n_samples == 0:
            return {
                "labels": [],
                "cluster_counts": [],
                "cluster_indices": {},
                "dominant_cluster_id": 0,
                "dominant_representative_idx": -1,
                "second_representative_idx": None,
                "similarity_matrix": np.empty((0, 0)),
            }

        if n_samples == 1:
            return {
                "labels": [0],
                "cluster_counts": [1],
                "cluster_indices": {0: [0]},
                "dominant_cluster_id": 0,
                "dominant_representative_idx": 0,
                "second_representative_idx": None,
                "similarity_matrix": np.ones((1, 1), dtype=np.float32),
            }

        sim_matrix = EmbeddingEngine.compute_cosine_similarity_matrix(embeddings)
        # Convert cosine similarity to cosine distance
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        # Ensure zero diagonal
        np.fill_diagonal(dist_matrix, 0.0)

        # Agglomerative clustering with precomputed distance
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
            n_clusters=None,
        )
        raw_labels = clustering.fit_predict(dist_matrix)

        # Count frequencies of raw clusters and rank them descending
        unique_labels, counts = np.unique(raw_labels, return_counts=True)
        # Sort by count descending
        sorted_order = np.argsort(-counts)
        sorted_unique = unique_labels[sorted_order]
        sorted_counts = counts[sorted_order].tolist()

        # Remap raw cluster labels so cluster 0 is the most frequent
        label_remap = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted_unique)}
        canonical_labels = [label_remap[lbl] for lbl in raw_labels]

        # Build cluster -> indices mapping
        cluster_indices: Dict[int, List[int]] = {i: [] for i in range(len(sorted_unique))}
        for idx, lbl in enumerate(canonical_labels):
            cluster_indices[lbl].append(idx)

        # Identify representative (medoid) for each cluster
        def find_medoid(indices: List[int]) -> int:
            if not indices:
                return -1
            if len(indices) == 1:
                return indices[0]
            # Medoid has highest average similarity to other members in its cluster
            sub_sim = sim_matrix[np.ix_(indices, indices)]
            mean_sims = np.mean(sub_sim, axis=1)
            best_local_idx = int(np.argmax(mean_sims))
            return indices[best_local_idx]

        dominant_rep_idx = find_medoid(cluster_indices[0])
        second_rep_idx = find_medoid(cluster_indices[1]) if len(sorted_unique) > 1 else None

        return {
            "labels": canonical_labels,
            "cluster_counts": sorted_counts,
            "cluster_indices": cluster_indices,
            "dominant_cluster_id": 0,
            "dominant_representative_idx": dominant_rep_idx,
            "second_representative_idx": second_rep_idx,
            "similarity_matrix": sim_matrix,
        }
