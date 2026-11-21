"""Reproducible ML Experiment Framework for LLM Uncertainty Classification.

Research Objective:
    "LLM Uncertainty Classification Using Multi-Response Consistency
     and Feature-Based Machine Learning"

Evaluates whether multi-response consistency and information-theoretic features
improve classification of KNOWN, AMBIGUOUS, and UNKNOWN queries over simpler approaches.
"""

from src.experimentation.feature_groups import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    EXPERIMENT_CONFIGURATIONS,
    get_features_for_config,
)
from src.experimentation.dataset_loader import (
    ExperimentDatasetLoader,
    DatasetValidationReport,
)
from src.experimentation.data_splitter import (
    LeakageSafeSplitter,
    SplitResult,
)
from src.experimentation.models import (
    ModelFactory,
    SUPPORTED_MODELS,
)
from src.experimentation.metrics import (
    calculate_experiment_metrics,
    MetricsResult,
)
from src.experimentation.tracker import (
    ExperimentTracker,
    ExperimentRecord,
)
from src.experimentation.reporter import (
    ExperimentReporter,
)
from src.experimentation.runner import (
    ExperimentRunner,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "EXPERIMENT_CONFIGURATIONS",
    "get_features_for_config",
    "ExperimentDatasetLoader",
    "DatasetValidationReport",
    "LeakageSafeSplitter",
    "SplitResult",
    "ModelFactory",
    "SUPPORTED_MODELS",
    "calculate_experiment_metrics",
    "MetricsResult",
    "ExperimentTracker",
    "ExperimentRecord",
    "ExperimentReporter",
    "ExperimentRunner",
]
