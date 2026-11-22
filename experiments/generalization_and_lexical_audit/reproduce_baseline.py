#!/usr/bin/env python3
"""Stage 1: Reproduce the Existing TF-IDF + Linear SVM Baseline.

Dataset: data/batch_2_combined_unified.jsonl
Methodology: 5-Fold StratifiedGroupKFold on question_group_id
Pipeline: TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2, sublinear_tf=True) -> LinearSVC(dual='auto', random_state=42, max_iter=2000, C=1.0)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

def reproduce_baseline():
    unified_path = os.path.join(PROJECT_ROOT, "data", "batch_2_combined_unified.jsonl")
    with open(unified_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {unified_path}")

    y = np.array([r["target_label"] for r in records])
    groups = [r["question_group_id"] for r in records]
    texts = [r["responses"][0]["text"] for r in records]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs = []
    fold_f1s = []
    fold_bal_accs = []
    fold_kappas = []
    fold_mccs = []
    oof_preds = [None] * len(records)
    class_precisions = {c: [] for c in CLASSES}
    class_recalls = {c: [] for c in CLASSES}
    class_f1s = {c: [] for c in CLASSES}

    fold_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(texts, y, groups)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            sublinear_tf=True,
        )
        X_tr = vec.fit_transform(train_texts)
        X_te = vec.transform(test_texts)

        clf = LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)

        for te_i, p in zip(test_idx, preds):
            oof_preds[te_i] = p

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_test, preds))
        kappa = float(cohen_kappa_score(y_test, preds))
        mcc = float(matthews_corrcoef(y_test, preds))

        p_c = precision_score(y_test, preds, labels=CLASSES, average=None, zero_division=0)
        r_c = recall_score(y_test, preds, labels=CLASSES, average=None, zero_division=0)
        f_c = f1_score(y_test, preds, labels=CLASSES, average=None, zero_division=0)

        for ci, c in enumerate(CLASSES):
            class_precisions[c].append(float(p_c[ci]))
            class_recalls[c].append(float(r_c[ci]))
            class_f1s[c].append(float(f_c[ci]))

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_bal_accs.append(bal_acc)
        fold_kappas.append(kappa)
        fold_mccs.append(mcc)

        fold_rows.append({
            "fold": fold_i,
            "accuracy": acc,
            "macro_f1": f1,
            "balanced_accuracy": bal_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
            "known_f1": float(f_c[0]),
            "ambiguous_f1": float(f_c[1]),
            "unknown_f1": float(f_c[2]),
        })

    oof_acc = float(accuracy_score(y, oof_preds))
    oof_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    cm = confusion_matrix(y, oof_preds, labels=CLASSES)

    print("\n--- BASELINE REPRODUCTION RESULTS ---")
    print(f"Mean Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs, ddof=1):.4f} (OOF: {oof_acc:.4f}, {sum(np.array(oof_preds)==y)}/60)")
    print(f"Mean Macro-F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s, ddof=1):.4f} (OOF: {oof_f1:.4f})")
    print(f"Fold Accuracies: {[round(x, 4) for x in fold_accs]}")
    print(f"Fold Macro-F1s: {[round(x, 4) for x in fold_f1s]}")
    print("\nConfusion Matrix (Labels: KNOWN, AMBIGUOUS, UNKNOWN):")
    print(cm)
    print("\nClassification Report (OOF):")
    print(classification_report(y, oof_preds, labels=CLASSES, digits=4))

    out_csv = os.path.join(os.path.dirname(__file__), "results", "baseline_reproduction.csv")
    df_res = pd.DataFrame(fold_rows)
    df_res.loc["mean"] = {
        "fold": "mean",
        "accuracy": float(np.mean(fold_accs)),
        "macro_f1": float(np.mean(fold_f1s)),
        "balanced_accuracy": float(np.mean(fold_bal_accs)),
        "cohen_kappa": float(np.mean(fold_kappas)),
        "mcc": float(np.mean(fold_mccs)),
        "known_f1": float(np.mean(class_f1s["KNOWN"])),
        "ambiguous_f1": float(np.mean(class_f1s["AMBIGUOUS"])),
        "unknown_f1": float(np.mean(class_f1s["UNKNOWN"])),
    }
    df_res.loc["std"] = {
        "fold": "std",
        "accuracy": float(np.std(fold_accs, ddof=1)),
        "macro_f1": float(np.std(fold_f1s, ddof=1)),
        "balanced_accuracy": float(np.std(fold_bal_accs, ddof=1)),
        "cohen_kappa": float(np.std(fold_kappas, ddof=1)),
        "mcc": float(np.std(fold_mccs, ddof=1)),
        "known_f1": float(np.std(class_f1s["KNOWN"], ddof=1)),
        "ambiguous_f1": float(np.std(class_f1s["AMBIGUOUS"], ddof=1)),
        "unknown_f1": float(np.std(class_f1s["UNKNOWN"], ddof=1)),
    }
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved reproduction results to {out_csv}")
    return oof_acc, float(np.mean(fold_f1s))

if __name__ == "__main__":
    reproduce_baseline()
