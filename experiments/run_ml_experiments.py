#!/usr/bin/env python3
"""CLI runner for Reproducible ML Uncertainty Experiments.

Allows running single model comparisons, multi-model benchmark suites,
and feature ablation studies with zero leakage and automatic reporting.
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experimentation.runner import ExperimentRunner
from src.experimentation.models import SUPPORTED_MODELS
from src.experimentation.feature_groups import EXPERIMENT_CONFIGURATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ml_experiments")


def main():
    parser = argparse.ArgumentParser(description="Reproducible ML Experiment Runner for Uncertainty Classification.")
    parser.add_argument("--features", "-f", type=str, default="data/features.jsonl",
                        help="Input features JSONL file")
    parser.add_argument("--mode", "-m", type=str, default="suite",
                        choices=["single", "suite", "ablation"],
                        help="Experiment mode: single model, multi-model suite, or ablation study")
    parser.add_argument("--model", type=str, default="hist_gradient_boosting",
                        choices=SUPPORTED_MODELS,
                        help="Model to evaluate (used for 'single' and 'ablation' modes)")
    parser.add_argument("--feature-config", type=str, default="full_16_features",
                        choices=list(EXPERIMENT_CONFIGURATIONS.keys()),
                        help="Feature configuration (used for 'single' mode)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Custom output directory for tracking and reports")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting and models")

    args = parser.parse_args()

    if not os.path.exists(args.features):
        logger.error("Features file '%s' does not exist.", args.features)
        sys.exit(1)

    runner = ExperimentRunner(
        dataset_path=args.features,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    if args.mode == "single":
        logger.info("Running single model evaluation: %s on %s...", args.model, args.feature_config)
        rec = runner.run_single(
            model_name=args.model,
            feature_config_name=args.feature_config,
        )
        print(f"\n[EXPERIMENT COMPLETED]")
        print(f"ID:        {rec.experiment_id}")
        print(f"Macro-F1:  {rec.metrics['macro_f1']:.4f}")
        print(f"Accuracy:  {rec.metrics['accuracy']:.4f}")
        print(f"Artifacts: {rec.output_dir}")

    elif args.mode == "suite":
        logger.info("Running comparative experiment suite across all baseline models...")
        records = runner.run_suite()
        print(f"\n[SUITE COMPLETED: {len(records)} runs]")
        print(f"Report: {os.path.join(runner.run_dir, 'experiment_suite_report.md')}")

    elif args.mode == "ablation":
        logger.info("Running feature ablation study for model '%s'...", args.model)
        records = runner.run_ablation_study(model_name=args.model)
        print(f"\n[ABLATION STUDY COMPLETED: {len(records)} configurations]")
        print(f"Report: {os.path.join(runner.run_dir, 'experiment_suite_report.md')}")


if __name__ == "__main__":
    main()
