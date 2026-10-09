"""Embedding generator using sentence embeddings (all-MiniLM-L6-v2).

Encodes textual responses into dense vector representations (384-dimensional) using
the local `all-MiniLM-L6-v2` model. Supports both native sentence-transformers
and local Ollama `all-minilm` embedding service with automatic fallback.
"""

from typing import List, Optional
import numpy as np
import requests

from configs.config import DEFAULT_NLP_CONFIG, DEFAULT_MODEL_CONFIG

# Module-level cache for the model instance to avoid re-loading on every invocation
_EMBEDDING_MODEL_CACHE = None


def get_sentence_transformer_model(model_name: Optional[str] = None):
    """Retrieve or initialize native SentenceTransformer model instance if available."""
    global _EMBEDDING_MODEL_CACHE

    if model_name is None:
        model_name = DEFAULT_NLP_CONFIG.embedding_model_name

    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
        return _EMBEDDING_MODEL_CACHE
    except Exception:
        return None


get_embedding_model = get_sentence_transformer_model


def compute_ollama_embeddings(
    texts: List[str],
    model_name: str = "all-minilm",
    base_url: str = DEFAULT_MODEL_CONFIG.ollama_base_url,
) -> np.ndarray:
    """Encode texts using the local all-minilm embedding model running in Ollama.

    Returns float32 ndarray of shape (N, 384).
    """
    clean_inputs = [t.strip() if t and t.strip() else " " for t in texts]

    resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": model_name, "input": clean_inputs},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama embedding request failed (HTTP {resp.status_code}): {resp.text}"
        )

    data = resp.json()
    embeddings_list = data.get("embeddings", [])
    return np.asarray(embeddings_list, dtype=np.float32)


def compute_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of text strings into dense embedding vectors.

    Args:
        texts: List of response strings to embed.
        model_name: Sentence-transformer model identifier.
        normalize: If True, L2-normalizes vectors so dot product equals cosine similarity.

    Returns:
        np.ndarray of shape (N, 384) with float32 embeddings.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    clean_inputs = [t.strip() if t and t.strip() else " " for t in texts]

    # Try native sentence-transformers first
    st_model = get_sentence_transformer_model(model_name)
    if st_model is not None:
        embeddings = st_model.encode(
            clean_inputs,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # Fallback to local Ollama all-minilm embedding service
    embeddings = compute_ollama_embeddings(clean_inputs)

    if normalize and embeddings.shape[0] > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        embeddings = embeddings / norms

    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix from embeddings.

    If vectors are already L2-normalized, cosine_sim(u, v) = u . v.
    Otherwise, computes standard cosine similarity.
    """
    if embeddings.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms

    similarity_matrix = np.matmul(normed, normed.T)
    return np.clip(similarity_matrix, -1.0, 1.0)
