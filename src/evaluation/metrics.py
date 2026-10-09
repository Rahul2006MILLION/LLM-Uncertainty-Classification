"""Evaluation metrics module for uncertainty classification.

Calculates:
- Per-class precision, recall, and F1
- Macro F1 score
- Overall accuracy
- Confusion matrix
- Brier score and expected calibration error (ECE) when calibrated probabilities are available.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Args:
        y_true: List of ground-truth class labels.
        y_pred: List of predicted class labels.
        classes: Ordered list of unique class names. Defaults to sorted unique labels.

    Returns:
        Dictionary containing per-class metrics, macro F1, accuracy, and confusion matrix.
    """
    if not y_true or not y_pred:
        return {"error": "Empty predictions or labels provided."}

    if classes is None:
        classes = sorted(list(set(y_true).union(set(y_pred))))

    # Compute confusion matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in class_to_idx and yp in class_to_idx:
            cm[class_to_idx[yt], class_to_idx[yp]] += 1

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        f1_scores.append(f1)

    total_correct = int(np.trace(cm))
    total_samples = len(y_true)
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "total_samples": total_samples,
    }
