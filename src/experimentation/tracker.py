"""Reproducible experiment tracking and JSON metadata artifact persistence."""

import os
import sys
import json
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

from src.experimentation.metrics import MetricsResult


@dataclass
class ExperimentRecord:
    """Full reproducible record of a single ML model evaluation run."""
    experiment_id: str
    run_timestamp: str
    dataset_path: str
    dataset_sha256: str
    dataset_size: int
    class_distribution: Dict[str, int]
    model_name: str
    model_parameters: Dict[str, Any]
    feature_config_name: str
    feature_names: List[str]
    split_info: Dict[str, Any]
    metrics: Dict[str, Any]
    execution_time_seconds: float
    output_dir: str
    software_versions: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_timestamp": self.run_timestamp,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "dataset_size": self.dataset_size,
            "class_distribution": self.class_distribution,
            "model_name": self.model_name,
            "model_parameters": self.model_parameters,
            "feature_config_name": self.feature_config_name,
            "feature_names": self.feature_names,
            "split_info": self.split_info,
            "metrics": self.metrics,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "output_dir": self.output_dir,
            "software_versions": self.software_versions,
            "notes": self.notes,
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """Serialize experiment record to JSON."""
        target_path = filepath or os.path.join(self.output_dir, f"{self.experiment_id}_record.json")
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path


class ExperimentTracker:
    """Manages experiment runs, isolated directories, and reproducible artifact storage."""

    def __init__(self, base_output_dir: str = "experiments/ml_runs"):
        self.base_output_dir = base_output_dir

    @staticmethod
    def get_software_versions() -> Dict[str, str]:
        """Collect runtime environment versions."""
        import sklearn
        versions = {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        }
        try:
            import xgboost
            versions["xgboost"] = xgboost.__version__
        except ImportError:
            versions["xgboost"] = "not_installed"
        return versions

    def create_run_directory(self, suite_id: str) -> str:
        """Create a timestamped, unique directory for an experiment suite."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_output_dir, f"{suite_id}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
