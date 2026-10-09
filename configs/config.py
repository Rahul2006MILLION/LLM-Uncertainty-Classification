"""Central configuration for Adaptive LLM Uncertainty Classification Project.
"""

from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "llama3.1:8b-research"
    temperature: float = 0.8
    top_p: float = 0.9
    num_generations: int = 10
    timeout_seconds: int = 120
    # Expected context length (verified dynamically from /api/show)
    expected_context_window: int = 1096


@dataclass
class NLPConfig:
    # Small, efficient local embedding model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Cosine similarity threshold for grouping answers into the same semantic cluster.
    # Responses with cosine similarity >= threshold are connected/grouped together.
    # Documented & configurable: typical range [0.75, 0.85].
    similarity_threshold: float = 0.80


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT
    results_dir: Path = PROJECT_ROOT / "results"
    raw_generations_dir: Path = PROJECT_ROOT / "results" / "raw_generations"
    features_dir: Path = PROJECT_ROOT / "results" / "features"
    models_dir: Path = PROJECT_ROOT / "results" / "models"


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_NLP_CONFIG = NLPConfig()
DEFAULT_PATH_CONFIG = PathConfig()
