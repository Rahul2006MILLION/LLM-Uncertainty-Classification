#!/usr/bin/env python3
"""Automated, reproducible experiment runner and complete pipeline evaluation.

Evaluates the full uncertainty pipeline across the diverse 60-question benchmark:
- Target 1: Uncertainty Classification (Accuracy, Macro-F1, Per-Class, Confusion Matrix)
- Target 2: Routing Efficiency (Fast Path vs Multi-Response, compute savings)
- Target 3: Factual Correctness, Ambiguity Polysemy, & Hallucination / Refusal Analysis
"""

import sys
import os
import json
import time
import argparse
import datetime
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from tabulate import tabulate
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.router import AdaptiveRouter
from src.classification.classifier import UncertaintyClassifier, LABEL_TO_ID, ID_TO_LABEL
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.features.clustering import SemanticClusterer
from src.evaluation.metrics import evaluate_classifier_predictions
from src.preprocessing.text_cleaner import strip_boilerplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmarks")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def assess_correctness_or_behavior(
    expected_class: str,
    final_answer: str,
    ground_truth_answers: Optional[List[str]],
    interpretations: Optional[List[str]],
    subcategory: str,
) -> Dict[str, Any]:
    """Assess whether answer is correct (KNOWN), acknowledges ambiguity (AMBIGUOUS), or acknowledges unknowability (UNKNOWN)."""
    ans_lower = final_answer.lower()

    if expected_class == "KNOWN":
        if not ground_truth_answers:
            return {"status": "UNKNOWN_GROUND_TRUTH", "detail": "No gold answers"}
        # Check if any ground truth answer appears in final answer
        match = any(gt.lower() in ans_lower for gt in ground_truth_answers)
        return {
            "status": "CORRECT" if match else "INCORRECT_OR_PARTIAL",
            "detail": f"Matched ground truth {ground_truth_answers}" if match else f"Did not explicitly match {ground_truth_answers}"
        }

    elif expected_class == "AMBIGUOUS":
        # Check if response mentions multiple senses or conditional language
        has_conditional = any(w in ans_lower for w in ["depending on", "either", "or", "both", "can refer to", "refers to", "could mean", "disambiguation"])
        interp_matches = sum(1 for interp in (interpretations or []) if any(w.lower() in ans_lower for w in interp.split()[:2])) if interpretations else 0
        acknowledged = has_conditional or (interp_matches >= 2)
        return {
            "status": "ACKNOWLEDGED_AMBIGUITY" if acknowledged else "COLLAPSED_TO_SINGLE_SENSE",
            "detail": "Multiple interpretations discussed" if acknowledged else "Single interpretation given without ambiguity caveats"
        }

    elif expected_class == "UNKNOWN":
        # Check if model admits it cannot know vs confidently hallucinating
        refusal_markers = [
            "cannot know", "can't know", "cannot predict", "impossible to determine",
            "private thought", "not public", "future event", "unknown", "false premise",
            "does not organize", "there is no", "does not exist", "neither", "no such"
        ]
        admits_uncertainty = any(m in ans_lower for m in refusal_markers)
        return {
            "status": "ACKNOWLEDGED_UNKNOWABLE" if admits_uncertainty else "CONFIDENT_HALLUCINATION",
            "detail": "Admitted uncertainty / unknowability" if admits_uncertainty else "Hallucinated or fabricated answer"
        }

    return {"status": "UNASSESSED", "detail": "N/A"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete pipeline on the uncertainty benchmark.")
    parser.add_argument("--benchmark", "-b", type=str, default="data/benchmarks/uncertainty_benchmark_v2.jsonl")
    parser.add_argument("--responses", "-r", type=str, default="data/raw/generated_responses.jsonl")
    parser.add_argument("--features", "-f", type=str, default="data/features.jsonl")
    parser.add_argument("--model", "-m", type=str, default="models/uncertainty_classifier.joblib")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--output", "-o", type=str, default="experiments/results/benchmark_report.json")
    args = parser.parse_args()

    benchmark_items = load_jsonl(args.benchmark)
    response_items = load_jsonl(args.responses)
    feature_items = load_jsonl(args.features)

    logger.info("Loaded %d benchmark items, %d response items, %d feature items.",
                len(benchmark_items), len(response_items), len(feature_items))

    # Index responses and features by question
    resp_by_q = {r["question"]: r for r in response_items}
    feat_by_q = {f["question"]: f for f in feature_items}

    # Load classifier
    classifier = None
    if os.path.exists(args.model):
        classifier = UncertaintyClassifier.load(args.model)
        logger.info("Loaded classifier from %s (trained: %s)", args.model, classifier.is_trained)
    else:
        logger.warning("Classifier %s not found. Inference will report untrained status.", args.model)
        classifier = UncertaintyClassifier()

    router = AdaptiveRouter(fast_path_enabled=True)

    records_eval: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    # Routing stats
    fast_path_count = 0
    multi_response_count = 0
    total_generations = 0

    for item in benchmark_items:
        qid = item["id"]
        q = item["question"]
        expected_label = item["label"]
        subcategory = item.get("subcategory", "general")
        gt_answers = item.get("ground_truth_answers")
        interpretations = item.get("interpretations")

        resp_record = resp_by_q.get(q)
        feat_record = feat_by_q.get(q)

        if not resp_record or not feat_record:
            logger.warning("Missing responses or features for question: %s", q)
            continue

        responses = resp_record.get("responses", [])
        initial_text = responses[0]["text"] if responses else ""

        # Step 1: Router decision
        route_decision = router.route(q, initial_text)
        is_fast_path = (route_decision.path == "FAST_PATH")

        if is_fast_path:
            fast_path_count += 1
            gen_count = 1
            # Fast path single-response feature extraction
            clean_initial = strip_boilerplate(initial_text)
            words = clean_initial.split()
            feats = {
                "response_len_mean": float(len(words)),
                "response_len_std": 0.0,
                "distinct_1gram_ratio": float(len(set(words)) / max(len(words), 1)),
                "distinct_2gram_ratio": 0.0,
                "question_overlap_mean": 0.0,
                "hedging_score": 0.0,
                "negation_count": float(sum(1 for w in words if w.lower() in ["not", "never", "cannot", "no", "neither"])),
                "semantic_entropy": 0.0,
                "normalized_entropy": 0.0,
                "dominant_agreement": 1.0,
                "second_agreement": 0.0,
                "agreement_margin": 1.0,
                "cluster_count": 1.0,
                "cluster_ratio": 1.0,
                "mean_pairwise_similarity": 1.0,
                "min_pairwise_similarity": 1.0,
            }
        else:
            multi_response_count += 1
            gen_count = 10
            feats = feat_record.get("features", {})

        total_generations += gen_count

        # Step 2: Classifier prediction
        if classifier and classifier.is_trained:
            pred_res = classifier.predict(feats)
            predicted_label = pred_res["predicted_label"]
            confidence = pred_res["confidence"]
        else:
            predicted_label = "UNKNOWN"
            confidence = 0.0

        y_true.append(LABEL_TO_ID[expected_label])
        y_pred.append(LABEL_TO_ID.get(predicted_label, 2))

        # Step 3: Assess correctness or behavioral response
        behavior = assess_correctness_or_behavior(
            expected_class=expected_label,
            final_answer=initial_text,
            ground_truth_answers=gt_answers,
            interpretations=interpretations,
            subcategory=subcategory,
        )

        records_eval.append({
            "id": qid,
            "question": q,
            "subcategory": subcategory,
            "expected_class": expected_label,
            "predicted_class": predicted_label,
            "is_correct_prediction": (expected_label == predicted_label),
            "router_decision": route_decision.path,
            "generation_count": gen_count,
            "confidence": confidence,
            "semantic_entropy": feats.get("semantic_entropy", 0.0),
            "dominant_agreement": feats.get("dominant_agreement", 1.0),
            "second_agreement": feats.get("second_agreement", 0.0),
            "cluster_count": int(feats.get("cluster_count", 1)),
            "final_answer": initial_text[:120] + ("..." if len(initial_text) > 120 else ""),
            "behavior_assessment": behavior,
        })

    # Evaluate Metrics
    clf_metrics = evaluate_classifier_predictions(y_true, y_pred)
    acc = clf_metrics["accuracy"]
    macro_f1 = clf_metrics["macro_f1"]
    conf_mat = clf_metrics["confusion_matrix"]

    # Compute Routing Efficiency
    total_q = len(records_eval)
    naive_generations = total_q * 10
    compute_savings_pct = (naive_generations - total_generations) / naive_generations * 100.0

    # Subcategory Performance Breakdown
    subcat_stats: Dict[str, Dict[str, Any]] = {}
    for r in records_eval:
        sc = r["subcategory"]
        if sc not in subcat_stats:
            subcat_stats[sc] = {"total": 0, "correct_pred": 0, "expected_classes": {}}
        subcat_stats[sc]["total"] += 1
        if r["is_correct_prediction"]:
            subcat_stats[sc]["correct_pred"] += 1
        exp_c = r["expected_class"]
        subcat_stats[sc]["expected_classes"][exp_c] = subcat_stats[sc]["expected_classes"].get(exp_c, 0) + 1

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("      RESEARCH MILESTONE EVALUATION REPORT — 60-QUESTION BENCHMARK")
    print("=" * 80 + "\n")

    print("### TARGET 1: UNCERTAINTY CLASSIFICATION METRICS")
    print(f"- Overall Accuracy: {acc:.3f}")
    print(f"- Macro-F1 Score:   {macro_f1:.3f}\n")

    per_class_table = []
    for cls_name, m in clf_metrics["per_class"].items():
        per_class_table.append([
            cls_name,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1_score']:.3f}",
            m["support"],
        ])
    print(tabulate(per_class_table, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="github") + "\n")

    print("### CONFUSION MATRIX")
    print(tabulate(
        conf_mat,
        headers=["Pred: KNOWN", "Pred: AMBIGUOUS", "Pred: UNKNOWN"],
        showindex=["True: KNOWN", "True: AMBIGUOUS", "True: UNKNOWN"],
        tablefmt="github"
    ) + "\n")

    print("### TARGET 2: ROUTING EFFICIENCY")
    print(f"- Fast-Path Routed:       {fast_path_count} / {total_q} ({fast_path_count/total_q*100:.1f}%)")
    print(f"- Multi-Response Routed:  {multi_response_count} / {total_q} ({multi_response_count/total_q*100:.1f}%)")
    print(f"- Total Generations:      {total_generations} (vs {naive_generations} naive)")
    print(f"- Compute Savings:        {compute_savings_pct:.1f}%\n")

    print("### TARGET 3: SUBCATEGORY BREAKDOWN & BEHAVIORAL FIDELITY")
    subcat_rows = []
    for sc, s in sorted(subcat_stats.items()):
        acc_sc = s["correct_pred"] / s["total"] * 100.0
        exp_summary = ", ".join(f"{k}:{v}" for k, v in s["expected_classes"].items())
        subcat_rows.append([sc, s["total"], f"{acc_sc:.1f}%", exp_summary])
    print(tabulate(subcat_rows, headers=["Subcategory", "Total", "Accuracy", "Class Composition"], tablefmt="github") + "\n")

    # Representative Successes & Failures
    print("### REPRESENTATIVE CASES")
    rep_cases = []
    # Pick 2 correct and 2 incorrect cases
    correct_samples = [r for r in records_eval if r["is_correct_prediction"]][:3]
    error_samples = [r for r in records_eval if not r["is_correct_prediction"]][:3]
    for r in correct_samples + error_samples:
        rep_cases.append([
            r["question"][:36] + ("..." if len(r["question"]) > 36 else ""),
            r["subcategory"],
            r["expected_class"],
            r["predicted_class"],
            r["router_decision"],
            f"{r['confidence']:.2f}",
            "SUCCESS" if r["is_correct_prediction"] else "ERROR",
        ])
    print(tabulate(rep_cases, headers=["Question", "Subcat", "Expected", "Predicted", "Route", "Conf", "Status"], tablefmt="github") + "\n")

    # Export Report
    final_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "benchmark_file": args.benchmark,
        "total_examples": total_q,
        "classification_metrics": clf_metrics,
        "routing_metrics": {
            "fast_path_count": fast_path_count,
            "multi_response_count": multi_response_count,
            "total_generations": total_generations,
            "naive_generations": naive_generations,
            "compute_savings_pct": compute_savings_pct,
        },
        "subcategory_breakdown": subcat_stats,
        "detailed_results": records_eval,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Complete benchmark report exported to %s", args.output)


if __name__ == "__main__":
    main()

