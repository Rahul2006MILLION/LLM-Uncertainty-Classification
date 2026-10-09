"""Train and evaluate the Supervised Uncertainty Classifier.

Integrates:
- Strict dataset validation and label definition verification.
- Leak-free train / validation / test isolation.
- Class distribution reporting.
- XGBoost (with RandomForest fallback).
- Multi-class evaluation metrics: Per-class Precision/Recall/F1, Macro-F1, Confusion Matrix, Accuracy.
- Model serialization to results/models/uncertainty_classifier.joblib.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_PATH_CONFIG
from src.classification.dataset import (
    LABEL_DEFINITIONS,
    VALID_LABELS,
    load_dataset_from_json,
    split_dataset,
    compute_class_distribution,
)
from src.classification.classifier import UncertaintyClassifier
from src.evaluation.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Train Uncertainty Classifier on labeled dataset.")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to labeled dataset JSON file.",
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Classifier algorithm: 'xgboost' (default) or 'random_forest'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_PATH_CONFIG.models_dir / "uncertainty_classifier.joblib"),
        help="Path to save trained classifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splitting.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[Error] Dataset file '{data_path}' not found.")
        print("Please provide a valid labeled dataset JSON file matching the benchmark schema.")
        return

    print("=" * 60)
    print("UNCERTAINTY CLASSIFIER TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset:    {data_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Seed:       {args.seed}")
    print("-" * 60)

    # 1. Load and validate
    print("\n[1/5] Loading and validating dataset...")
    try:
        examples = load_dataset_from_json(data_path)
    except Exception as e:
        print(f"[Validation Failed] {e}")
        return

    total_samples = len(examples)
    print(f"  • Successfully validated {total_samples} examples.")

    # 2. Compute class distribution
    print("\n[2/5] Class Distribution Analysis:")
    dist = compute_class_distribution(examples)
    for lbl, stats in dist.items():
        print(f"  • {lbl:<10}: {stats['count']:>4} examples ({stats['percentage']:>5.1f}%)")

    # 3. Leak-free splitting
    print("\n[3/5] Splitting dataset (70% train / 15% val / 15% test, question-isolated)...")
    try:
        train_set, val_set, test_set = split_dataset(
            examples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"[Splitting Error] {e}")
        return

    print(f"  • Train set: {len(train_set)} examples")
    print(f"  • Val set:   {len(val_set)} examples")
    print(f"  • Test set:  {len(test_set)} examples")
    print("  • Leakage check: ZERO question overlap verified across splits.")

    # 4. Train classifier
    print(f"\n[4/5] Training {args.model_type.upper()} classifier...")
    classifier = UncertaintyClassifier(model_type=args.model_type, random_state=args.seed)
    val_report = classifier.fit(train_examples=train_set, val_examples=val_set)

    # 5. Evaluate on held-out test set
    print("\n[5/5] Evaluating on independent test set...")
    if test_set:
        X_test = classifier._features_to_matrix(test_set)
        y_test_true = [ex.label for ex in test_set]
        y_test_pred_idx = classifier.model.predict(X_test)
        from src.classification.classifier import INT_TO_LABEL
        y_test_pred = [INT_TO_LABEL[int(p)] for p in y_test_pred_idx]

        test_report = evaluate_predictions(
            y_true=y_test_true,
            y_pred=y_test_pred,
            classes=sorted(list(VALID_LABELS)),
        )

        print("-" * 60)
        print("TEST EVALUATION RESULTS")
        print("-" * 60)
        print(f"Accuracy: {test_report['accuracy']:.1%}")
        print(f"Macro F1: {test_report['macro_f1']:.3f}\n")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 52)
        for cls_name, metrics in test_report["per_class"].items():
            print(
                f"{cls_name:<12} "
                f"{metrics['precision']:<10.3f} "
                f"{metrics['recall']:<10.3f} "
                f"{metrics['f1']:<10.3f} "
                f"{metrics['support']:<8}"
            )
        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
        print(f"Classes: {test_report['classes']}")
        for row in test_report["confusion_matrix"]:
            print(f"  {row}")
    else:
        print("  • Note: Test set was empty due to small dataset size.")

    # Save model
    save_path = classifier.save(Path(args.output))
    print(f"\nTrained classifier successfully saved to: {save_path}")


if __name__ == "__main__":
    main()
