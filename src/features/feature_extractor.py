"""Comprehensive feature extraction for uncertainty classification without label leakage."""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from src.features.entropy import compute_semantic_entropy_metrics
from src.features.clustering import SemanticClusterer
from src.preprocessing.text_cleaner import tokenize_preserving_negation, NEGATION_WORDS

# Hedging and epistemic uncertainty markers
HEDGING_PATTERNS = [
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bmight\b",
    r"\bpossibly\b",
    r"\bunclear\b",
    r"\bit depends\b",
    r"\bnot certain\b",
    r"\bcannot determine\b",
    r"\bcould refer to\b",
    r"\bdifficult to say\b",
    r"\bno definitive\b",
    r"\bunknown\b",
    r"\bunanswerable\b",
    r"\binsufficient information\b",
    r"\bambiguous\b",
    r"\bdepends on the context\b",
]

FEATURE_NAMES = [
    "semantic_entropy",
    "normalized_entropy",
    "dominant_agreement",
    "second_agreement",
    "agreement_margin",
    "cluster_count",
    "cluster_ratio",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
    "response_len_mean",
    "response_len_std",
    "distinct_1gram_ratio",
    "distinct_2gram_ratio",
    "question_overlap_mean",
    "hedging_score",
    "negation_count",
]


class FeatureExtractor:
    """Extracts consistency, information-theoretic, and linguistic features."""

    def __init__(self, clusterer: Optional[SemanticClusterer] = None):
        self.clusterer = clusterer or SemanticClusterer()

    def extract_features(
        self,
        question: str,
        responses: List[str],
        embeddings: np.ndarray,
        cluster_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Extract complete feature dictionary for a question and its responses.
        
        Strictly operates on question text and generated responses; NO TARGET LABELS
        are accessible, preventing any label leakage.
        """
        n_samples = len(responses)
        if n_samples == 0:
            return {name: 0.0 for name in FEATURE_NAMES}

        # 1. Clustering
        if cluster_result is None:
            cluster_result = self.clusterer.cluster_responses(responses, embeddings)

        cluster_counts = cluster_result["cluster_counts"]
        sim_matrix = cluster_result["similarity_matrix"]

        # 2. Entropy & Agreement
        entropy_metrics = compute_semantic_entropy_metrics(cluster_counts, n_samples)

        # 3. Pairwise Similarity (off-diagonal)
        if n_samples > 1 and sim_matrix.shape[0] == n_samples:
            upper_tri = sim_matrix[np.triu_indices(n_samples, k=1)]
            mean_pair_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 1.0
            min_pair_sim = float(np.min(upper_tri)) if len(upper_tri) > 0 else 1.0
        else:
            mean_pair_sim = 1.0
            min_pair_sim = 1.0

        # 4. Lexical & Length Statistics
        lens = [len(r) for r in responses]
        len_mean = float(np.mean(lens))
        len_std = float(np.std(lens))

        # Distinct n-grams and vocabulary
        all_unigrams: List[str] = []
        all_bigrams: List[str] = []
        q_tokens = set(tokenize_preserving_negation(question))
        overlaps: List[float] = []
        hedges: List[float] = []
        negations: List[float] = []

        for r in responses:
            tokens = tokenize_preserving_negation(r)
            all_unigrams.extend(tokens)
            if len(tokens) >= 2:
                all_bigrams.extend([f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)])

            # Overlap with question (Jaccard)
            r_tokens = set(tokens)
            union = q_tokens | r_tokens
            overlap = len(q_tokens & r_tokens) / len(union) if union else 0.0
            overlaps.append(overlap)

            # Hedging markers
            r_lower = r.lower()
            h_count = sum(1 for pat in HEDGING_PATTERNS if re.search(pat, r_lower))
            hedges.append(float(h_count))

            # Negation tokens
            n_count = sum(1 for t in tokens if t in NEGATION_WORDS)
            negations.append(float(n_count))

        distinct_1g = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
        distinct_2g = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        overlap_mean = float(np.mean(overlaps)) if overlaps else 0.0
        hedging_mean = float(np.mean(hedges)) if hedges else 0.0
        negation_mean = float(np.mean(negations)) if negations else 0.0

        features: Dict[str, float] = {
            "semantic_entropy": entropy_metrics["semantic_entropy"],
            "normalized_entropy": entropy_metrics["normalized_entropy"],
            "dominant_agreement": entropy_metrics["dominant_agreement"],
            "second_agreement": entropy_metrics["second_agreement"],
            "agreement_margin": entropy_metrics["agreement_margin"],
            "cluster_count": entropy_metrics["cluster_count"],
            "cluster_ratio": entropy_metrics["cluster_ratio"],
            "mean_pairwise_similarity": mean_pair_sim,
            "min_pairwise_similarity": min_pair_sim,
            "response_len_mean": len_mean,
            "response_len_std": len_std,
            "distinct_1gram_ratio": float(distinct_1g),
            "distinct_2gram_ratio": float(distinct_2g),
            "question_overlap_mean": overlap_mean,
            "hedging_score": hedging_mean,
            "negation_count": negation_mean,
        }

        return features

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return the fixed list of feature names in column order."""
        return list(FEATURE_NAMES)
