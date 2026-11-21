"""Experiment runner orchestrating baselines, feature groups, and ablation studies."""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

from src.experimentation.dataset_loader import ExperimentDatasetLoader
from src.experimentation.data_splitter import LeakageSafeSplitter, SplitResult
from src.experimentation.models import ModelFactory, SUPPORTED_MODELS
from src.experimentation.feature_groups import (
    FEATURE_NAMES,
    EXPERIMENT_CONFIGURATIONS,
    get_features_for_config,
)
from src.experimentation.metrics import calculate_experiment_metrics
from src.experimentation.tracker import ExperimentTracker, ExperimentRecord
from src.experimentation.reporter import ExperimentReporter

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates reproducible model evaluations across feature groups and ablations."""

    def __init__(
        self,
        dataset_path: str,
        output_dir: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.dataset_path = dataset_path
        self.random_seed = random_seed
        self.tracker = ExperimentTracker()
        self.run_dir = output_dir or self.tracker.create_run_directory("uncertainty_ml_experiment")
        self.loader = ExperimentDatasetLoader()
        self.splitter = LeakageSafeSplitter(random_seed=random_seed)

    def run_single(
        self,
        model_name: str,
        feature_config_name: str = "full_16_features",
        hyperparameters: Optional[Dict[str, Any]] = None,
        split_result: Optional[SplitResult] = None,
    ) -> ExperimentRecord:
        """Run a single model on a specified feature configuration."""
        start_t = time.time()
        records, val_report = self.loader.load_and_validate(self.dataset_path)

        # Extract complete feature matrices
        feature_list = get_features_for_config(feature_config_name)
        X, y, ids, groups = self.loader.extract_matrices(records, feature_names=feature_list)

        # Split if not already provided
        if split_result is None:
            split_result = self.splitter.split(records, y, ids, groups=groups)

        # Save split artifact once
        split_path = os.path.join(self.run_dir, "split_metadata.json")
        if not os.path.exists(split_path):
            split_result.save(split_path)

        # Train partition
        X_train = X[split_result.train_indices]
        y_train = y[split_result.train_indices]

        # Test partition (strictly isolated)
        X_test = X[split_result.test_indices]
        y_test = y[split_result.test_indices]

        # Instantiate and train model
        model = ModelFactory.create_model(
            model_name=model_name,
            random_seed=self.random_seed,
            hyperparameters=hyperparameters,
        )
        model.fit(X_train, y_train)

        # Inference
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        # Metrics
        metrics_res = calculate_experiment_metrics(y_test, y_pred, y_prob=y_prob)

        elapsed = time.time() - start_t
        exp_id = f"{model_name}_{feature_config_name}"

        rec = ExperimentRecord(
            experiment_id=exp_id,
            run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_path=self.dataset_path,
            dataset_sha256=val_report.file_sha256,
            dataset_size=len(records),
            class_distribution=val_report.class_distribution,
            model_name=model_name,
            model_parameters=hyperparameters or {},
            feature_config_name=feature_config_name,
            feature_names=feature_list,
            split_info=split_result.to_dict(),
            metrics=metrics_res.to_dict(),
            execution_time_seconds=elapsed,
            output_dir=self.run_dir,
            software_versions=self.tracker.get_software_versions(),
            notes=split_result.limitations,
        )
        rec.save()
        return rec

    def run_suite(
        self,
        models: Optional[List[str]] = None,
        feature_configs: Optional[List[str]] = None,
    ) -> List[ExperimentRecord]:
        """Execute a matrix of models across feature configurations on the identical split."""
        target_models = models or ["majority_baseline", "logistic_regression", "random_forest", "hist_gradient_boosting"]
        target_configs = feature_configs or [
            "single_response_baseline",
            "multi_response_consistency",
            "full_16_features",
        ]

        # Load and split ONCE for the entire suite to guarantee identical, fair comparisons
        records, _ = self.loader.load_and_validate(self.dataset_path)
        X_all, y_all, ids, groups = self.loader.extract_matrices(records)
        shared_split = self.splitter.split(records, y_all, ids, groups=groups)

        suite_records: List[ExperimentRecord] = []
        for mname in target_models:
            for cfg_name in target_configs:
                logger.info("Executing model '%s' with feature configuration '%s'...", mname, cfg_name)
                rec = self.run_single(
                    model_name=mname,
                    feature_config_name=cfg_name,
                    split_result=shared_split,
                )
                suite_records.append(rec)

        # Generate suite report
        report_path = os.path.join(self.run_dir, "experiment_suite_report.md")
        ExperimentReporter.generate_markdown_report(
            suite_id="ml_uncertainty_comparison_suite",
            records=suite_records,
            output_path=report_path,
        )
        logger.info("Experiment suite report generated at %s", report_path)
        return suite_records

    def run_ablation_study(
        self,
        model_name: str = "random_forest",
    ) -> List[ExperimentRecord]:
        """Run systematic leave-one-group-out feature ablation study."""
        ablation_configs = [
            "full_16_features",
            "ablation_no_entropy",
            "ablation_no_agreement",
            "ablation_no_clustering",
            "ablation_no_similarity",
            "ablation_no_length_diversity",
            "ablation_no_question_overlap",
            "ablation_no_hedging_negation",
        ]
        return self.run_suite(models=[model_name], feature_configs=ablation_configs)
