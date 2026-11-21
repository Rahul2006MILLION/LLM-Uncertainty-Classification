"""Features module for clustering, entropy, feature extraction, and candidate answer extraction."""

from src.features.answer_extractor import (
    extract_concise_answer,
    select_semantic_final_answer,
)
from src.features.clustering import SemanticClusterer
from src.features.entropy import compute_semantic_entropy_metrics
from src.features.feature_extractor import FEATURE_NAMES, FeatureExtractor

__all__ = [
    "SemanticClusterer",
    "compute_semantic_entropy_metrics",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "extract_concise_answer",
    "select_semantic_final_answer",
]
