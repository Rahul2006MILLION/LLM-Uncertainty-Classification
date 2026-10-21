"""Semantic embedding management, L2 normalization, and cosine similarity."""

import logging
from typing import List, Optional
import numpy as np
from src.generation.ollama_client import OllamaClient
from src.preprocessing.text_cleaner import clean_text_for_embedding

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Computes and normalizes semantic embeddings for texts."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        embedding_model: str = "all-minilm",
        expected_dim: Optional[int] = 384,
    ):
        self.client = client or OllamaClient()
        self.embedding_model = embedding_model
        self.expected_dim = expected_dim
        self._dim: Optional[int] = expected_dim

    def embed_single(self, text: str) -> np.ndarray:
        """Compute an L2-normalized embedding vector for a single text.
        
        Handles empty or whitespace-only strings gracefully by returning a zero vector.
        """
        cleaned = clean_text_for_embedding(text)
        if not cleaned:
            dim = self._dim or self.expected_dim or 384
            return np.zeros(dim, dtype=np.float32)

        try:
            vec_list = self.client.embed(cleaned, model=self.embedding_model)
            vec = np.array(vec_list, dtype=np.float32)
            self._dim = vec.shape[0]

            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.zeros_like(vec)
            return vec
        except Exception as e:
            logger.warning("Embedding call failed for text '%s...': %s", cleaned[:30], e)
            dim = self._dim or self.expected_dim or 384
            return np.zeros(dim, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, D) normalized numpy array."""
        if not texts:
            dim = self._dim or self.expected_dim or 384
            return np.empty((0, dim), dtype=np.float32)

        vectors = [self.embed_single(t) for t in texts]
        return np.vstack(vectors)

    @staticmethod
    def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute the NxN pairwise cosine similarity matrix.
        
        Assumes embeddings are already L2-normalized.
        Returns clipped values in [-1.0, 1.0].
        """
        if embeddings.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)

        sim_matrix = np.matmul(embeddings, embeddings.T)
        return np.clip(sim_matrix, -1.0, 1.0)
