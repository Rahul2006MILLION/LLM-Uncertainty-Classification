"""Comprehensive multi-class classification metrics."""

from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
    roc_auc_score,
)

CANONICAL_CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]


@dataclass
class MetricsResult:
    """Standardized classification metrics container."""
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    cohen_kappa: float
    matthews_corrcoef: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    roc_auc_ovr: Optional[float]
    brier_score: Optional[float]
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "cohen_kappa": self.cohen_kappa,
            "matthews_corrcoef": self.matthews_corrcoef,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "roc_auc_ovr": self.roc_auc_ovr,
            "brier_score": self.brier_score,
            "total_samples": self.total_samples,
        }


def calculate_experiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> MetricsResult:
    """Calculate complete multi-class metrics for uncertainty classification."""
    names = class_names or CANONICAL_CLASSES
    n_samples = len(y_true)

    if n_samples == 0:
        return MetricsResult(
            accuracy=0.0,
            balanced_accuracy=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            cohen_kappa=0.0,
            matthews_corrcoef=0.0,
            per_class={c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in names},
            confusion_matrix=[[0]*len(names) for _ in range(len(names))],
            roc_auc_ovr=None,
            brier_score=None,
            total_samples=0,
        )

    labels = list(range(len(names)))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Per-class breakdown
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class_dict = {}
    for i, cname in enumerate(names):
        supp = int(np.sum(y_true == i))
        per_class_dict[cname] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1": float(per_class_f1[i]),
            "support": supp,
        }

    # Probability-based metrics (ROC-AUC and Brier Score)
    roc_auc = None
    brier = None
    if y_prob is not None and y_prob.shape == (n_samples, len(names)):
        try:
            # One-vs-Rest Macro ROC-AUC
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

        try:
            # Multiclass Brier score: mean squared error across one-hot probability vectors
            y_one_hot = np.zeros_like(y_prob)
            for idx, true_c in enumerate(y_true):
                if 0 <= true_c < len(names):
                    y_one_hot[idx, true_c] = 1.0
            brier = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))
        except Exception:
            brier = None

    return MetricsResult(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        cohen_kappa=kappa,
        matthews_corrcoef=mcc,
        per_class=per_class_dict,
        confusion_matrix=cm,
        roc_auc_ovr=roc_auc,
        brier_score=brier,
        total_samples=n_samples,
    )
