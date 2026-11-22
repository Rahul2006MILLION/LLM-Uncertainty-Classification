"""Error Analysis and Shortcut Investigation Module for Single-Response Benchmark.

Analyzes:
1. Class-specific errors (Precision, Recall, F1 for KNOWN, AMBIGUOUS, UNKNOWN).
2. Confusion patterns and transition matrices.
3. Category-level performance across 15 semantic domains.
4. Linguistic shortcut correlations (length, hedging, negation, question length).
5. Feature importance (coefficients / tree importances).
6. Difficult example identification and CSV generation.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

def analyze_confusion_and_classes(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str]
) -> Dict[str, Any]:
    """Compute confusion matrix and per-class metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cr = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    per_class = {}
    for c in classes:
        per_class[c] = {
            "precision": cr[c]["precision"],
            "recall": cr[c]["recall"],
            "f1-score": cr[c]["f1-score"],
            "support": cr[c]["support"],
        }
        
    return {
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

def analyze_category_performance(
    records: List[Dict[str, Any]],
    y_pred: List[str]
) -> pd.DataFrame:
    """Analyze accuracy per question category."""
    rows = []
    for r, pred in zip(records, y_pred):
        rows.append({
            "category": r.get("category", "unknown"),
            "target_label": r["target_label"],
            "predicted_label": pred,
            "correct": 1 if pred == r["target_label"] else 0,
        })
    df = pd.DataFrame(rows)
    cat_summary = df.groupby("category").agg(
        total_questions=("correct", "count"),
        correct_count=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    cat_summary = cat_summary.sort_values(by="accuracy", ascending=False)
    return cat_summary

def analyze_linguistic_shortcuts(
    records: List[Dict[str, Any]],
    feature_dicts: List[Dict[str, float]],
    y_pred: List[str]
) -> Dict[str, Any]:
    """Check whether predictions or correct classifications correlate with linguistic shortcuts."""
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    class_map = {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 2}
    
    y_num = np.array([class_map[r["target_label"]] for r in records])
    pred_num = np.array([class_map[p] for p in y_pred])
    correct_num = (y_num == pred_num).astype(int)
    
    lengths = np.array([len(r["responses"][0]["text"]) for r in records])
    q_lengths = np.array([len(r["question"]) for r in records])
    hedgings = np.array([fd["hedging_score"] for fd in feature_dicts])
    negations = np.array([fd["negation_count"] for fd in feature_dicts])
    
    shortcuts = {
        "response_len_vs_prediction_spearman": float(stats.spearmanr(lengths, pred_num).statistic),
        "response_len_vs_correct_spearman": float(stats.spearmanr(lengths, correct_num).statistic),
        "question_len_vs_prediction_spearman": float(stats.spearmanr(q_lengths, pred_num).statistic),
        "hedging_vs_prediction_spearman": float(stats.spearmanr(hedgings, pred_num).statistic),
        "negation_vs_prediction_spearman": float(stats.spearmanr(negations, pred_num).statistic),
    }
    return shortcuts

def generate_difficult_examples_csv(
    records: List[Dict[str, Any]],
    baseline_preds: List[str],
    improved_preds: List[str],
    out_csv_path: str
) -> pd.DataFrame:
    """Generate difficult examples comparison table."""
    rows = []
    for r, b_pred, imp_pred in zip(records, baseline_preds, improved_preds):
        gt = r["target_label"]
        r_text = r["responses"][0]["text"]
        
        status = "NEITHER_CORRECT"
        if b_pred == gt and imp_pred == gt:
            status = "BOTH_CORRECT"
        elif imp_pred == gt and b_pred != gt:
            status = "IMPROVED_ONLY"
        elif b_pred == gt and imp_pred != gt:
            status = "BASELINE_ONLY"
            
        rows.append({
            "question_id": r["id"],
            "category": r.get("category", ""),
            "question": r["question"],
            "ground_truth": gt,
            "response_text": r_text[:200] + ("..." if len(r_text) > 200 else ""),
            "baseline_pred": b_pred,
            "improved_pred": imp_pred,
            "status": status,
            "ambiguity_mechanism": r.get("ambiguity_mechanism", ""),
            "epistemic_barrier": r.get("epistemic_barrier", ""),
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    return df
