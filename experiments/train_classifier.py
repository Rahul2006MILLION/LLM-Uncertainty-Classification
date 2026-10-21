#!/usr/bin/env python3
"""Reproducible classifier training, group-based evaluation, baselines, and ablations."""

import sys
import os
import json
import argparse
import datetime
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import yaml
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.classifier import UncertaintyClassifier, LABEL_TO_ID, ID_TO_LABEL
from src.evaluation.metrics import evaluate_classifier_predictions
from src.evaluation.data_validator import DatasetValidator
from src.features.feature_extractor import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_classifier")


def load_feature_records(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_xy(
    records: List[Dict[str, Any]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert JSON feature records to numpy feature matrix X and label vector y."""
    X_list = []
    y_list = []
    valid_ids = []

    for r in records:
        lbl = r.get("label")
        if lbl not in LABEL_TO_ID:
            continue
        feats = r.get("features", {})
        vec = [float(feats.get(fn, 0.0)) for fn in feature_names]
        X_list.append(vec)
        y_list.append(LABEL_TO_ID[lbl])
        valid_ids.append(r.get("id"))

    if not X_list:
        return np.empty((0, len(feature_names))), np.empty((0,), dtype=int), []

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int), valid_ids


def run_entropy_threshold_baseline(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic baseline:
    - If semantic_entropy < 0.20 and dominant_agreement >= 0.85 -> KNOWN
    - If second_agreement >= 0.20 or agreement_margin < 0.40 -> AMBIGUOUS
    - Otherwise -> UNKNOWN
    """
    y_true = []
    y_pred = []
    for r in records:
        lbl = r.get("label")
        if lbl not in LABEL_TO_ID:
            continue
        y_true.append(LABEL_TO_ID[lbl])
        feats = r.get("features", {})
        h = feats.get("semantic_entropy", 0.0)
        p1 = feats.get("dominant_agreement", 0.0)
        p2 = feats.get("second_agreement", 0.0)
        fsd = feats.get("agreement_margin", 0.0)

        if h < 0.20 and p1 >= 0.85:
            pred = "KNOWN"
        elif p2 >= 0.20 or (fsd < 0.40 and h > 0.40):
            pred = "AMBIGUOUS"
        else:
            pred = "UNKNOWN"
        y_pred.append(LABEL_TO_ID[pred])

    return evaluate_classifier_predictions(y_true, y_pred)


def run_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Majority class baseline predictor."""
    if len(y_train) == 0 or len(y_test) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    vals, counts = np.unique(y_train, return_counts=True)
    maj_class = vals[np.argmax(counts)]
    y_pred = [maj_class] * len(y_test)
    return evaluate_classifier_predictions(y_test.tolist(), y_pred)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate uncertainty classifier.")
    parser.add_argument("--features", "-f", type=str, default="data/features.jsonl",
                        help="Input features JSONL file")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "hist_gb", "random_forest"])
    parser.add_argument("--output-model", "-o", type=str, default="models/uncertainty_classifier.json")
    parser.add_argument("--report", type=str, default="experiments/results/training_report.json")
    parser.add_argument("--save-artifact", action="store_true", help="Save model artifact if validation passes")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        logger.error("Features file %s does not exist. Run analyze_responses.py first.", args.features)
        sys.exit(1)

    records = load_feature_records(args.features)
    logger.info("Loaded %d feature records from %s.", len(records), args.features)

    # Validate dataset coverage
    val_check = DatasetValidator.validate_records(records)
    logger.info("Dataset validation: %s", val_check)

    if len(records) < 15:
        logger.warning(
            "Dataset sample size (%d) is too small for defensible generalization. "
            "Model will be trained for pipeline demonstration, but not certified for production.",
            len(records),
        )

    # 1. Group-based train/test split to prevent leakage
    train_records, val_records, test_records = DatasetValidator.group_train_val_test_split(
        records, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_seed=42
    )
    # If test is empty due to small grouping, fallback to train/test split
    if not test_records:
        train_records, test_records, _ = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.70, val_ratio=0.30, test_ratio=0.0, random_seed=42
        )

    all_features = FEATURE_NAMES
    # Multi-response features vs Single-response features (ablation)
    single_features = [
        "response_len_mean", "distinct_1gram_ratio", "question_overlap_mean",
        "hedging_score", "negation_count"
    ]
    multi_features = [
        "semantic_entropy", "normalized_entropy", "dominant_agreement",
        "second_agreement", "agreement_margin", "cluster_count",
        "cluster_ratio", "mean_pairwise_similarity", "min_pairwise_similarity"
    ]

    X_train, y_train, _ = records_to_xy(train_records, all_features)
    X_test, y_test, _ = records_to_xy(test_records, all_features)

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    # 2. Train Primary Model
    classifier = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    classifier.train(
        X_train, y_train, feature_names=all_features,
        metadata={"timestamp": datetime.datetime.now().isoformat(), "feature_count": len(all_features)}
    )

    # Evaluate Primary Model on Test Partition
    test_preds = [classifier.predict({fn: X_test[i, j] for j, fn in enumerate(all_features)})["predicted_class_id"]
                  for i in range(len(y_test))]
    primary_eval = evaluate_classifier_predictions(y_test.tolist(), test_preds)

    # 3. Baselines
    maj_eval = run_majority_baseline(y_train, y_test)
    heuristic_eval = run_entropy_threshold_baseline(test_records)

    # 4. Feature Ablations
    # Ablation A: Single-response features only
    X_train_single, _, _ = records_to_xy(train_records, single_features)
    X_test_single, _, _ = records_to_xy(test_records, single_features)
    single_clf = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    single_clf.train(X_train_single, y_train, feature_names=single_features)
    single_preds = [single_clf.predict({fn: X_test_single[i, j] for j, fn in enumerate(single_features)})["predicted_class_id"]
                    for i in range(len(y_test))]
    single_eval = evaluate_classifier_predictions(y_test.tolist(), single_preds)

    # Ablation B: Multi-response consistency features only
    X_train_multi, _, _ = records_to_xy(train_records, multi_features)
    X_test_multi, _, _ = records_to_xy(test_records, multi_features)
    multi_clf = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    multi_clf.train(X_train_multi, y_train, feature_names=multi_features)
    multi_preds = [multi_clf.predict({fn: X_test_multi[i, j] for j, fn in enumerate(multi_features)})["predicted_class_id"]
                   for i in range(len(y_test))]
    multi_eval = evaluate_classifier_predictions(y_test.tolist(), multi_preds)

    # Tabulate comparison
    table_data = [
        ["Majority Class Baseline", maj_eval["accuracy"], maj_eval["macro_f1"]],
        ["Entropy Heuristic Baseline", heuristic_eval["accuracy"], heuristic_eval["macro_f1"]],
        ["Single-Response Features Only", single_eval["accuracy"], single_eval["macro_f1"]],
        ["Multi-Response Features Only", multi_eval["accuracy"], multi_eval["macro_f1"]],
        [f"Full Model ({args.model_type})", primary_eval["accuracy"], primary_eval["macro_f1"]],
    ]
    print("\n" + tabulate(table_data, headers=["Model / Configuration", "Test Accuracy", "Test Macro-F1"], floatfmt=".3f", tablefmt="github") + "\n")

    # Confusion matrix
    print("Confusion Matrix (Full Model):")
    conf_mat = primary_eval["confusion_matrix"]
    print(tabulate(conf_mat, headers=["Pred: KNOWN", "Pred: AMBIGUOUS", "Pred: UNKNOWN"], showindex=["True: KNOWN", "True: AMBIGUOUS", "True: UNKNOWN"], tablefmt="github"))
    print("")

    # Export report
    report_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": args.model_type,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "primary_model_eval": primary_eval,
        "majority_baseline_eval": maj_eval,
        "heuristic_baseline_eval": heuristic_eval,
        "ablation_single_features": single_eval,
        "ablation_multi_features": multi_eval,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Training and evaluation report saved to %s", args.report)

    # Save model artifact if requested
    if args.save_artifact:
        classifier.save(args.output_model)
        logger.info("Classifier artifact saved to %s", args.output_model)


if __name__ == "__main__":
    main()
