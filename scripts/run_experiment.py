#!/usr/bin/env python3
"""CLI Driver for the Experimental Evaluation Pipeline.

Milestone 7.5.5: Experimental Evaluation Pipeline Infrastructure
Research Project: LLM Uncertainty Classification Using Multi-Response Consistency
                  and Feature-Based Machine Learning

Executes reproducible evaluation of baseline and proposed models on the finalized gold benchmark.
Enforces the Gold Benchmark Execution Gate: fails safely (exit code 2) when real gold benchmark is missing.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluation_pipeline import (
    ExperimentConfig,
    ExperimentRunner,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible model evaluation on the finalized gold uncertainty benchmark."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML or JSON experiment configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, dataset provenance, and split integrity without training models.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Alias for --dry-run (validates datasets and configuration only).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for random seed.",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = ExperimentConfig.from_file(args.config)
    except Exception as ex:
        print(f"Error loading configuration '{args.config}': {ex}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.dry_run or args.validate_only:
        config.dry_run = True
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.random_seed = args.seed

    runner = ExperimentRunner(config)
    exit_code, results, errors = runner.run()

    if exit_code != 0:
        print("\n" + "=" * 50)
        print("EXPERIMENTAL EVALUATION PIPELINE")
        print("=" * 50)
        print("Status:\nBLOCKED\n")
        print("Reason:\nFinalized gold benchmark is unavailable or input dataset failed validation.")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} additional errors.")
        print("\nSafe failure safeguard active: No models were trained. No experimental results were generated.")
        print("=" * 50 + "\n")
        sys.exit(exit_code)

    if config.dry_run:
        print("\n" + "=" * 50)
        print("EXPERIMENTAL EVALUATION PIPELINE: DRY-RUN SUCCESS")
        print("=" * 50)
        print(f"Status:          {results.get('status')}")
        print(f"Model:           {results.get('model_name')}")
        print(f"Dataset Version: {results.get('dataset_version')}")
        print(f"Sample Count:    {results.get('sample_count')}")
        print(f"Feature Count:   {results.get('feature_count')} ({config.feature_set})")
        print("All provenance, schema, and leakage gates passed successfully.")
        print("=" * 50 + "\n")
        sys.exit(0)

    print("\n" + "=" * 50)
    print(f"EXPERIMENTAL EVALUATION COMPLETED: {results.get('experiment_id')}")
    print("=" * 50)
    print(f"Model:           {results.get('model_name')}")
    print(f"Results File:    {os.path.join(config.output_dir, f'{config.experiment_id}_results.json')}")
    print(f"Audit Report:    {os.path.join(config.output_dir, f'{config.experiment_id}_audit.md')}")
    print(f"Test Accuracy:   {results.get('test_metrics', {}).get('accuracy')}")
    print(f"Test Macro F1:   {results.get('test_metrics', {}).get('macro_f1')}")
    print("=" * 50 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
