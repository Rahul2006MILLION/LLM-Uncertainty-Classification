"""Evaluation metrics for uncertainty classification."""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def evaluate_classifier_predictions(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate multi-class classification evaluation metrics.
    
    Returns:
        - accuracy
        - macro_f1
        - per_class_metrics
        - confusion_matrix
        - total_support
    """
    if target_names is None:
        target_names = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))).tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for name in target_names:
        if name in report:
            per_class[name] = {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf_mat,
        "total_support": len(y_true),
    }
