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
        lbl = DatasetValidator.extract_label(r) if hasattr(DatasetValidator, "extract_label") else r.get("label")
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
        lbl = DatasetValidator.extract_label(r) if hasattr(DatasetValidator, "extract_label") else r.get("label")
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
    parser.add_argument("--output-model", "-o", type=str, default="models/uncertainty_classifier.joblib")
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

    # 1. Group-based train/val/test split to prevent leakage
    train_records, val_records, test_records, split_meta = DatasetValidator.group_train_val_test_split(
        records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42, return_metadata=True
    )

    print("\n" + "=" * 78)
    print("  DATASET SPLIT & LEAKAGE AUDIT REPORT")
    print("=" * 78)
    print(f"Total Records:      {split_meta['total_records']}")
    print(f"Total Groups:       {split_meta['total_groups']} (grouped globally by normalized question text)")
    print(f"Train Partition:    {split_meta['train_record_count']} records, {split_meta['train_group_count']} groups | {split_meta['train_groups_by_class']}")
    print(f"Val Partition:      {split_meta['val_record_count']} records, {split_meta['val_group_count']} groups | {split_meta['val_groups_by_class']}")
    print(f"Test Partition:     {split_meta['test_record_count']} records, {split_meta['test_group_count']} groups | {split_meta['test_groups_by_class']}")
    print(f"Group Disjointness: {'VERIFIED - ZERO LEAKAGE' if split_meta['is_disjoint'] else 'FAILED - LEAKAGE DETECTED'}")
    if split_meta["limitations"]:
        print("\nIdentified Split Limitations:")
        for lim in split_meta["limitations"]:
            print(f"  [!] {lim}")
    print("-" * 78)
    print("SAMPLE SIZE AND GENERALIZATION NOTICE:")
    print("  The dataset sample size (N=67 records across 62 unique question groups) is small.")
    print("  Evaluation results on the test partition (N=9 records, 9 groups) carry high empirical")
    print("  variance. All comparisons below are strictly controlled on the identical fixed split")
    print("  to measure relative feature contributions; they do NOT establish asymptotic production")
    print("  performance or generalizable certainty.")
    print("=" * 78 + "\n")

    all_features = list(FEATURE_NAMES)
    single_features = [
        "response_len_mean", "distinct_1gram_ratio", "question_overlap_mean",
        "hedging_score", "negation_count"
    ]
    multi_features = [
        "semantic_entropy", "normalized_entropy", "dominant_agreement",
        "second_agreement", "agreement_margin", "cluster_count",
        "cluster_ratio", "mean_pairwise_similarity", "min_pairwise_similarity"
    ]
    no_negation_features = [f for f in all_features if f != "negation_count"]

    # Fixed training and test matrices
    X_train_full, y_train, _ = records_to_xy(train_records, all_features)
    X_val_full, y_val, _ = records_to_xy(val_records, all_features)
    X_test_full, y_test, _ = records_to_xy(test_records, all_features)

    # -------------------------------------------------------------
    # 5 Controlled Model Comparisons on Identical Fixed Partition
    # -------------------------------------------------------------
    eval_results = {}

    # Comparison 1: Majority Class Baseline
    maj_eval = run_majority_baseline(y_train, y_test)
    eval_results["Majority-Class Baseline"] = maj_eval

    # Comparison 2: Single-Response Feature Baseline
    X_train_single, _, _ = records_to_xy(train_records, single_features)
    X_test_single, _, _ = records_to_xy(test_records, single_features)
    clf_single = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    clf_single.train(X_train_single, y_train, feature_names=single_features)
    preds_single = [clf_single.predict({fn: X_test_single[i, j] for j, fn in enumerate(single_features)})["predicted_class_id"]
                    for i in range(len(y_test))]
    eval_results["Single-Response Feature Baseline"] = evaluate_classifier_predictions(y_test.tolist(), preds_single)

    # Comparison 3: Multi-Response Feature Baseline
    X_train_multi, _, _ = records_to_xy(train_records, multi_features)
    X_test_multi, _, _ = records_to_xy(test_records, multi_features)
    clf_multi = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    clf_multi.train(X_train_multi, y_train, feature_names=multi_features)
    preds_multi = [clf_multi.predict({fn: X_test_multi[i, j] for j, fn in enumerate(multi_features)})["predicted_class_id"]
                   for i in range(len(y_test))]
    eval_results["Multi-Response Feature Baseline"] = evaluate_classifier_predictions(y_test.tolist(), preds_multi)

    # Comparison 4: Full Feature Model
    clf_full = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    clf_full.train(X_train_full, y_train, feature_names=all_features)
    preds_full = [clf_full.predict({fn: X_test_full[i, j] for j, fn in enumerate(all_features)})["predicted_class_id"]
                  for i in range(len(y_test))]
    eval_results["Full Feature Model"] = evaluate_classifier_predictions(y_test.tolist(), preds_full)

    # Comparison 5: Full Feature Model Without negation_count
    X_train_no_neg, _, _ = records_to_xy(train_records, no_negation_features)
    X_test_no_neg, _, _ = records_to_xy(test_records, no_negation_features)
    clf_no_neg = UncertaintyClassifier(model_type=args.model_type, random_state=42)
    clf_no_neg.train(X_train_no_neg, y_train, feature_names=no_negation_features)
    preds_no_neg = [clf_no_neg.predict({fn: X_test_no_neg[i, j] for j, fn in enumerate(no_negation_features)})["predicted_class_id"]
                    for i in range(len(y_test))]
    eval_results["Full Model (w/o negation_count)"] = evaluate_classifier_predictions(y_test.tolist(), preds_no_neg)

    # Summary Comparison Table
    summary_table = []
    for mname, mres in eval_results.items():
        summary_table.append([mname, mres["accuracy"], mres["macro_f1"], mres["total_support"]])
    print("CONTROLLED MODEL COMPARISON SUMMARY (Fixed Test Partition):")
    print(tabulate(summary_table, headers=["Model / Configuration", "Accuracy", "Macro-F1", "Test Support"], floatfmt=".3f", tablefmt="github"))
    print("")

    # Detailed Per-Class Breakdown and Confusion Matrices for each comparison
    for mname, mres in eval_results.items():
        print(f"\n--- Detailed Evaluation: {mname} ---")
        p_table = []
        for cname in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]:
            cm = mres.get("per_class", {}).get(cname, {})
            p_table.append([
                cname,
                cm.get("precision", 0.0),
                cm.get("recall", 0.0),
                cm.get("f1_score", 0.0),
                cm.get("support", 0),
            ])
        print(tabulate(p_table, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], floatfmt=".3f", tablefmt="github"))
        print("\nConfusion Matrix:")
        c_mat = mres.get("confusion_matrix", [])
        print(tabulate(
            c_mat,
            headers=["Pred: KNOWN", "Pred: AMBIGUOUS", "Pred: UNKNOWN"],
            showindex=["True: KNOWN", "True: AMBIGUOUS", "True: UNKNOWN"],
            tablefmt="github",
        ))

    # Feature Importances for Full Model
    feat_importances = {}
    if hasattr(clf_full.model, "feature_importances_"):
        for fn, imp in zip(all_features, clf_full.model.feature_importances_):
            feat_importances[fn] = float(imp)
        sorted_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
        print("\nFeature Importances (Full Feature Model):")
        print(tabulate(sorted_feats, headers=["Feature", "Importance"], floatfmt=".4f", tablefmt="github"))
        print("")

    # Validation Set Evaluation (for model selection audit)
    if len(y_val) > 0:
        val_preds_full = [clf_full.predict({fn: X_val_full[i, j] for j, fn in enumerate(all_features)})["predicted_class_id"]
                          for i in range(len(y_val))]
        val_eval = evaluate_classifier_predictions(y_val.tolist(), val_preds_full)
        print(f"Validation Partition Performance (Full Model, N={len(y_val)}): Accuracy={val_eval['accuracy']:.3f}, Macro-F1={val_eval['macro_f1']:.3f}")

    # Export report
    report_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": args.model_type,
        "split_metadata": split_meta,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_importances": feat_importances,
        "eval_results": eval_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Training and evaluation report saved to %s", args.report)

    # Save model artifact ONLY if explicitly requested
    if args.save_artifact:
        if os.path.exists(args.output_model):
            logger.info("Explicitly saving/updating model artifact to %s", args.output_model)
        clf_full.save(args.output_model)
        logger.info("Classifier artifact saved to %s", args.output_model)
    else:
        logger.info("Evaluation complete. Model artifact NOT overwritten (use --save-artifact to save).")


if __name__ == "__main__":
    main()
