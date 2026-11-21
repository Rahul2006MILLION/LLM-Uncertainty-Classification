"""Information-theoretic calculations: Semantic Entropy, Normalized Entropy, and Agreement."""

import math
from typing import List, Dict, Any


def compute_semantic_entropy_metrics(
    cluster_counts: List[int],
    total_samples: int,
) -> Dict[str, float]:
    """Compute exact entropy and agreement metrics from cluster sizes.
    
    Parameters:
        cluster_counts: Sizes of clusters sorted descending (e.g. [7, 2, 1])
        total_samples: Total number of responses (sum of cluster_counts)
        
    Returns:
        Dict with keys:
            - 'semantic_entropy': H(X) in nats, >= 0.0
            - 'normalized_entropy': H(X) / ln(N) in [0.0, 1.0]
            - 'dominant_agreement': p1 = N1 / N
            - 'second_agreement': p2 = N2 / N (or 0.0)
            - 'agreement_margin': p1 - p2 (First-Second Difference)
            - 'cluster_count': float(len(cluster_counts))
            - 'cluster_ratio': float(len(cluster_counts)) / total_samples
    """
    if total_samples <= 0 or not cluster_counts:
        return {
            "semantic_entropy": 0.0,
            "normalized_entropy": 0.0,
            "dominant_agreement": 0.0,
            "second_agreement": 0.0,
            "agreement_margin": 0.0,
            "cluster_count": 0.0,
            "cluster_ratio": 0.0,
        }

    probs = [c / total_samples for c in cluster_counts if c > 0]
    k = len(probs)

    # Shannon / Semantic Entropy: H = - sum(p * ln(p))
    raw_entropy = -sum(p * math.log(p) for p in probs)
    entropy = max(0.0, float(raw_entropy))

    # Normalized Entropy: H / ln(N) for N > 1
    if total_samples > 1:
        max_possible_entropy = math.log(total_samples)
        norm_entropy = min(1.0, max(0.0, entropy / max_possible_entropy))
    else:
        norm_entropy = 0.0

    p1 = probs[0] if k >= 1 else 0.0
    p2 = probs[1] if k >= 2 else 0.0
    fsd = p1 - p2

    return {
        "semantic_entropy": float(entropy),
        "normalized_entropy": float(norm_entropy),
        "dominant_agreement": float(p1),
        "second_agreement": float(p2),
        "agreement_margin": float(fsd),
        "cluster_count": float(k),
        "cluster_ratio": float(k / total_samples),
    }
