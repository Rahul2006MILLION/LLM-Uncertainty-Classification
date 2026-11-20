"""Canonical feature definitions, group mappings, and experimental configurations."""

from typing import Dict, List, Set

# Canonical 16 features from feature_extractor.py
FEATURE_NAMES: List[str] = [
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

# Thematic Feature Groups based on scientific meaning
FEATURE_GROUPS: Dict[str, List[str]] = {
    "semantic_entropy": [
        "semantic_entropy",
        "normalized_entropy",
    ],
    "agreement_consensus": [
        "dominant_agreement",
        "second_agreement",
        "agreement_margin",
    ],
    "clustering": [
        "cluster_count",
        "cluster_ratio",
    ],
    "pairwise_similarity": [
        "mean_pairwise_similarity",
        "min_pairwise_similarity",
    ],
    "length_diversity": [
        "response_len_mean",
        "response_len_std",
        "distinct_1gram_ratio",
        "distinct_2gram_ratio",
    ],
    "question_overlap": [
        "question_overlap_mean",
    ],
    "hedging_negation": [
        "hedging_score",
        "negation_count",
    ],
}

# Strict Single-Response Feature Names (from responses[0] only)
TRUE_SINGLE_RESPONSE_FEATURES: List[str] = [
    "response_len",
    "distinct_1gram_ratio",
    "distinct_2gram_ratio",
    "question_overlap",
    "hedging_score",
    "negation_count",
]

# Ensemble Linguistic Feature Names (from 10 responses, non-consistency)
ENSEMBLE_LINGUISTIC_FEATURES: List[str] = [
    "response_len_mean",
    "response_len_std",
    "distinct_1gram_ratio",
    "distinct_2gram_ratio",
    "question_overlap_mean",
    "hedging_score",
    "negation_count",
]

# Consistency-only Feature Names (from 10 responses)
CONSISTENCY_ONLY_FEATURES: List[str] = [
    "semantic_entropy",
    "normalized_entropy",
    "dominant_agreement",
    "second_agreement",
    "agreement_margin",
    "cluster_count",
    "cluster_ratio",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
]

# Experimental Configurations for research comparisons
EXPERIMENT_CONFIGURATIONS: Dict[str, List[str]] = {
    # Experiment A: True Single-response baseline (r0 only, zero multi-response metrics)
    "true_single_response_baseline": TRUE_SINGLE_RESPONSE_FEATURES,
    "single_response_baseline": TRUE_SINGLE_RESPONSE_FEATURES, # alias

    # Experiment B: Ensemble linguistic baseline (10 responses, zero consistency metrics)
    "ensemble_linguistic_baseline": ENSEMBLE_LINGUISTIC_FEATURES,

    # Experiment C: Multi-response consistency features only (9 features)
    "multi_response_consistency_only": CONSISTENCY_ONLY_FEATURES,
    "multi_response_consistency": CONSISTENCY_ONLY_FEATURES, # alias

    # Experiment D: Full 16-feature model
    "full_16_feature_model": list(FEATURE_NAMES),
    "full_16_features": list(FEATURE_NAMES), # alias
    # Feature ablations (leave-one-group-out)
    "ablation_no_entropy": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["semantic_entropy"]
    ],
    "ablation_no_agreement": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["agreement_consensus"]
    ],
    "ablation_no_clustering": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["clustering"]
    ],
    "ablation_no_similarity": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["pairwise_similarity"]
    ],
    "ablation_no_length_diversity": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["length_diversity"]
    ],
    "ablation_no_question_overlap": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["question_overlap"]
    ],
    "ablation_no_hedging_negation": [
        f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["hedging_negation"]
    ],
}


def get_features_for_config(config_name: str) -> List[str]:
    """Retrieve ordered feature names for a given experiment configuration name."""
    if config_name not in EXPERIMENT_CONFIGURATIONS:
        raise ValueError(
            f"Unknown experiment configuration '{config_name}'. "
            f"Valid options: {sorted(list(EXPERIMENT_CONFIGURATIONS.keys()))}"
        )
    return list(EXPERIMENT_CONFIGURATIONS[config_name])
