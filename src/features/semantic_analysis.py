"""Semantic analysis, clustering, entropy, and feature extraction.

Implements NLP-based uncertainty characterization over generated responses:
1. Semantic similarity matrix computation (cosine similarity over dense embeddings).
2. Configurable semantic clustering (agglomerative / threshold-based).
3. Empirical semantic entropy over answer clusters (raw and normalized).
4. Comprehensive, leak-free feature extraction for uncertainty classification.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from configs.config import DEFAULT_NLP_CONFIG
from src.preprocessing.text_cleaner import clean_text, tokenize_words
from src.features.embeddings import compute_embeddings, cosine_similarity_matrix


@dataclass
class SemanticCluster:
    cluster_id: int
    member_indices: List[int]
    size: int
    proportion: float
    representative_index: int
    representative_text: str


@dataclass
class SemanticAnalysisResult:
    num_responses: int
    num_clusters: int
    clusters: List[SemanticCluster]
    majority_cluster_agreement: float
    second_cluster_agreement: float
    agreement_margin: float  # Feature Difference Score (FSD): p1 - p2
    raw_semantic_entropy: float
    normalized_semantic_entropy: float
    mean_pairwise_similarity: float
    min_pairwise_similarity: float
    lexical_stats: Dict[str, float]
    features: Dict[str, float]


def perform_semantic_clustering(
    similarity_matrix: np.ndarray,
    texts: List[str],
    similarity_threshold: float = DEFAULT_NLP_CONFIG.similarity_threshold,
) -> List[SemanticCluster]:
    """Group responses into semantic clusters based on cosine similarity.

    Algorithm:
    Converts similarity matrix to a distance matrix D = 1 - S.
    Uses Agglomerative Clustering with average linkage and a distance threshold
    d_thresh = 1 - similarity_threshold. If scikit-learn is unavailable, falls back
    to threshold graph connected-components clustering.

    Representative response per cluster is selected as the medoid (member with highest
    average similarity to all other members in the same cluster).

    Args:
        similarity_matrix: Pairwise cosine similarity matrix (N, N).
        texts: List of response text strings.
        similarity_threshold: Minimum similarity to merge responses (default 0.80).

    Returns:
        List of SemanticCluster objects sorted by cluster size descending.
    """
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [
            SemanticCluster(
                cluster_id=0,
                member_indices=[0],
                size=1,
                proportion=1.0,
                representative_index=0,
                representative_text=texts[0],
            )
        ]

    distance_threshold = max(0.01, 1.0 - similarity_threshold)
    dist_matrix = np.clip(1.0 - similarity_matrix, 0.0, 2.0)
    # Ensure zero diagonal
    np.fill_diagonal(dist_matrix, 0.0)

    cluster_labels = None
    try:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
        cluster_labels = clustering.fit_predict(dist_matrix)
    except Exception:
        # Robust fallback: graph connected-components where S_ij >= threshold
        adj = similarity_matrix >= similarity_threshold
        visited = set()
        labels = np.zeros(n, dtype=int)
        current_cluster = 0
        for i in range(n):
            if i not in visited:
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    labels[curr] = current_cluster
                    for neighbor in range(n):
                        if neighbor not in visited and adj[curr, neighbor]:
                            visited.add(neighbor)
                            queue.append(neighbor)
                current_cluster += 1
        cluster_labels = labels

    # Group members by cluster ID
    clusters_dict: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(cluster_labels):
        clusters_dict.setdefault(lbl, []).append(idx)

    # Convert to SemanticCluster list sorted by cluster size descending
    clusters: List[SemanticCluster] = []
    sorted_groups = sorted(clusters_dict.values(), key=len, reverse=True)

    for new_id, members in enumerate(sorted_groups):
        # Choose medoid as representative: member with highest mean similarity to cluster members
        if len(members) == 1:
            rep_idx = members[0]
        else:
            sub_sim = similarity_matrix[np.ix_(members, members)]
            mean_sims = sub_sim.mean(axis=1)
            best_local_idx = int(np.argmax(mean_sims))
            rep_idx = members[best_local_idx]

        clusters.append(
            SemanticCluster(
                cluster_id=new_id,
                member_indices=members,
                size=len(members),
                proportion=len(members) / float(n),
                representative_index=rep_idx,
                representative_text=texts[rep_idx],
            )
        )

    return clusters


def calculate_semantic_entropy(
    clusters: List[SemanticCluster],
    total_responses: int,
) -> Tuple[float, float]:
    """Calculate empirical semantic entropy and normalized semantic entropy.

    Formula:
    For K semantic clusters with empirical probabilities p_k = |C_k| / N:
        Raw Entropy: H = - sum_{k=1}^K p_k * ln(p_k)
        Normalized Entropy: H_norm = H / ln(N)  (for N > 1, else 0.0)

    Theoretical properties:
    - H_norm is bounded in [0.0, 1.0].
    - H_norm = 0.0 when all responses belong to 1 semantic cluster (complete consensus).
    - H_norm = 1.0 when each response forms its own distinct cluster (maximum dispersion).
    - Note: This measures response dispersion / consistency, NOT factual correctness.

    Returns:
        (raw_entropy, normalized_entropy)
    """
    if total_responses <= 1 or not clusters:
        return 0.0, 0.0

    raw_entropy = 0.0
    for cluster in clusters:
        p = cluster.size / float(total_responses)
        if p > 0:
            raw_entropy -= p * math.log(p)

    max_entropy = math.log(total_responses)
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))

    return raw_entropy, normalized_entropy


def compute_lexical_statistics(texts: List[str]) -> Dict[str, float]:
    """Compute basic lexical statistics across generated responses."""
    if not texts:
        return {
            "mean_length_chars": 0.0,
            "mean_length_words": 0.0,
            "length_variance_words": 0.0,
            "lexical_diversity": 0.0,
        }

    char_lengths = [len(t) for t in texts]
    word_tokens_list = [tokenize_words(t) for t in texts]
    word_lengths = [len(tokens) for tokens in word_tokens_list]

    all_tokens = [token for tokens in word_tokens_list for token in tokens]
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    lexical_diversity = (unique_tokens / float(total_tokens)) if total_tokens > 0 else 0.0

    return {
        "mean_length_chars": float(np.mean(char_lengths)),
        "mean_length_words": float(np.mean(word_lengths)),
        "length_variance_words": float(np.var(word_lengths)),
        "lexical_diversity": float(lexical_diversity),
    }


def compute_question_ambiguity_features(question: str) -> Dict[str, float]:
    """Estimate question-level ambiguity independently from question text.

    IMPORTANT: This feature is computed strictly from the input question itself,
    completely independent from dataset labels, preventing any data leakage.
    """
    q_clean = clean_text(question).lower()
    q_words = tokenize_words(q_clean)
    num_words = len(q_words)

    # Disjunction / polysemy signals
    has_or = 1.0 if (" or " in f" {q_clean} ") else 0.0
    has_which = 1.0 if ("which" in q_words) else 0.0
    has_what = 1.0 if ("what" in q_words) else 0.0
    is_short_prompt = 1.0 if num_words <= 5 else 0.0

    return {
        "question_word_count": float(num_words),
        "question_has_disjunction": has_or,
        "question_has_which": has_which,
        "question_has_what": has_what,
        "question_is_short": is_short_prompt,
    }


def analyze_semantic_responses(
    responses: List[str],
    question: str = "",
    similarity_threshold: float = DEFAULT_NLP_CONFIG.similarity_threshold,
    embeddings: Optional[np.ndarray] = None,
) -> SemanticAnalysisResult:
    """Run full NLP semantic analysis over a list of generated responses.

    Args:
        responses: List of text responses to analyze.
        question: Original question string.
        similarity_threshold: Cosine similarity cutoff for clustering.
        embeddings: Precomputed embeddings (optional).

    Returns:
        SemanticAnalysisResult containing clusters, entropy, and feature dictionary.
    """
    n = len(responses)
    if n == 0:
        return SemanticAnalysisResult(
            num_responses=0,
            num_clusters=0,
            clusters=[],
            majority_cluster_agreement=0.0,
            second_cluster_agreement=0.0,
            agreement_margin=0.0,
            raw_semantic_entropy=0.0,
            normalized_semantic_entropy=0.0,
            mean_pairwise_similarity=0.0,
            min_pairwise_similarity=0.0,
            lexical_stats=compute_lexical_statistics([]),
            features={},
        )

    # 1. Compute embeddings and similarity matrix
    if embeddings is None:
        embeddings = compute_embeddings(responses)
    sim_matrix = cosine_similarity_matrix(embeddings)

    # 2. Semantic clustering
    clusters = perform_semantic_clustering(
        similarity_matrix=sim_matrix,
        texts=responses,
        similarity_threshold=similarity_threshold,
    )

    # 3. Agreement metrics
    majority_agreement = clusters[0].proportion if clusters else 0.0
    second_agreement = clusters[1].proportion if len(clusters) > 1 else 0.0
    agreement_margin = majority_agreement - second_agreement

    # 4. Semantic entropy
    raw_entropy, norm_entropy = calculate_semantic_entropy(clusters, n)

    # 5. Pairwise similarity metrics (off-diagonal elements)
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim_matrix[mask]
        mean_sim = float(np.mean(off_diag))
        min_sim = float(np.min(off_diag))
    else:
        mean_sim = 1.0
        min_sim = 1.0

    # 6. Lexical statistics
    lexical_stats = compute_lexical_statistics(responses)

    # 7. Independent question ambiguity features
    q_features = compute_question_ambiguity_features(question) if question else {}

    # Assemble complete feature vector for ML classifier
    features: Dict[str, float] = {
        "num_responses": float(n),
        "num_clusters": float(len(clusters)),
        "majority_cluster_agreement": float(majority_agreement),
        "second_cluster_agreement": float(second_agreement),
        "agreement_margin": float(agreement_margin),
        "raw_semantic_entropy": float(raw_entropy),
        "normalized_semantic_entropy": float(norm_entropy),
        "mean_pairwise_similarity": float(mean_sim),
        "min_pairwise_similarity": float(min_sim),
        "mean_length_chars": lexical_stats["mean_length_chars"],
        "mean_length_words": lexical_stats["mean_length_words"],
        "length_variance_words": lexical_stats["length_variance_words"],
        "lexical_diversity": lexical_stats["lexical_diversity"],
    }
    features.update(q_features)

    return SemanticAnalysisResult(
        num_responses=n,
        num_clusters=len(clusters),
        clusters=clusters,
        majority_cluster_agreement=majority_agreement,
        second_cluster_agreement=second_agreement,
        agreement_margin=agreement_margin,
        raw_semantic_entropy=raw_entropy,
        normalized_semantic_entropy=norm_entropy,
        mean_pairwise_similarity=mean_sim,
        min_pairwise_similarity=min_sim,
        lexical_stats=lexical_stats,
        features=features,
    )
