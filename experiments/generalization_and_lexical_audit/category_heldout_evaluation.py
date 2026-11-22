#!/usr/bin/env python3
"""Stage 2: Category-Held-Out Generalization (Leave-One-Category-Out).

Evaluates 5 models across 15 held-out question categories (4 test questions per category, 56 train):
A. TF-IDF + Linear SVM
B. Original Random Forest TRUE_SINGLE_RESPONSE baseline (canonical 6 features)
C. TF-IDF + Logistic Regression
D. Semantic Embeddings + Linear SVM
E. Hybrid Embeddings + Canonical Features

Ensures ZERO question leakage and ZERO category overlap between training and test sets.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features.feature_extractor import FeatureExtractor
from src.preprocessing.embeddings import EmbeddingEngine

CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = logging.getLogger("category_heldout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_category_heldout():
    unified_path = os.path.join(PROJECT_ROOT, "data", "batch_2_combined_unified.jsonl")
    with open(unified_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    n = len(records)
    logger.info("Loaded %d unified records.", n)

    categories = sorted(list(set(r["category"] for r in records)))
    logger.info("Found %d distinct categories: %s", len(categories), categories)

    texts = [r["responses"][0]["text"] for r in records]
    questions = [r["question"] for r in records]
    y = np.array([r["target_label"] for r in records])

    # Extract canonical 6 features
    canonical_feats = [
        "response_len", "distinct_1gram_ratio", "distinct_2gram_ratio",
        "question_overlap", "hedging_score", "negation_count"
    ]
    X_canonical = np.array([[r["r0_features"][k] for k in canonical_feats] for r in records], dtype=np.float32)

    # Precompute sentence embeddings (all-minilm)
    logger.info("Generating/retrieving 384-d sentence embeddings...")
    embed_engine = EmbeddingEngine(embedding_model="all-minilm")
    X_embed = embed_engine.embed_batch(texts)
    X_hybrid = np.hstack([X_embed, X_canonical])

    # Models to benchmark
    model_defs = {
        "A_TFIDF_LinearSVC": {
            "type": "text",
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
        },
        "B_Baseline_RandomForest": {
            "type": "dense",
            "X": X_canonical,
            "clf_factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        },
        "C_TFIDF_LogisticRegression": {
            "type": "text",
            "clf_factory": lambda: LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        },
        "D_Embedding_LinearSVC": {
            "type": "dense",
            "X": X_embed,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
        },
        "E_HybridA_LinearSVC": {
            "type": "dense",
            "X": X_hybrid,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
        },
    }

    # Tracking outputs
    oof_predictions = {m: [None] * n for m in model_defs}
    per_cat_accuracies = {m: {} for m in model_defs}
    per_cat_correct = {m: {} for m in model_defs}

    logger.info("Executing Leave-One-Category-Out across 15 folds...")
    for cat_idx, held_out_cat in enumerate(categories):
        test_mask = np.array([r["category"] == held_out_cat for r in records])
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        y_tr, y_te = y[train_indices], y[test_indices]
        cat_class = y_te[0] # All 4 in category share class

        for m_name, m_cfg in model_defs.items():
            clf = m_cfg["clf_factory"]()

            if m_cfg["type"] == "text":
                vec = TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=10000,
                    min_df=2,
                    sublinear_tf=True,
                )
                tr_texts = [texts[i] for i in train_indices]
                te_texts = [texts[i] for i in test_indices]

                X_tr = vec.fit_transform(tr_texts)
                X_te = vec.transform(te_texts)
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
            else:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(m_cfg["X"][train_indices])
                X_te = scaler.transform(m_cfg["X"][test_indices])
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)

            for te_i, p in zip(test_indices, preds):
                oof_predictions[m_name][te_i] = p

            n_corr = int(np.sum(preds == y_te))
            acc = float(n_corr / len(y_te))
            per_cat_accuracies[m_name][held_out_cat] = acc
            per_cat_correct[m_name][held_out_cat] = n_corr

    # Overall Metrics Calculation
    overall_summary = []
    for m_name in model_defs:
        preds = oof_predictions[m_name]
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y, preds))
        kappa = float(cohen_kappa_score(y, preds))
        mcc = float(matthews_corrcoef(y, preds))

        p_cls = precision_score(y, preds, labels=CLASSES, average=None, zero_division=0)
        r_cls = recall_score(y, preds, labels=CLASSES, average=None, zero_division=0)
        f_cls = f1_score(y, preds, labels=CLASSES, average=None, zero_division=0)

        cat_acc_list = list(per_cat_accuracies[m_name].values())
        mean_cat_acc = float(np.mean(cat_acc_list))
        std_cat_acc = float(np.std(cat_acc_list, ddof=1))

        cm = confusion_matrix(y, preds, labels=CLASSES).tolist()

        overall_summary.append({
            "model_name": m_name,
            "overall_accuracy": acc,
            "overall_macro_f1": f1,
            "balanced_accuracy": bal_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
            "known_precision": float(p_cls[0]),
            "known_recall": float(r_cls[0]),
            "known_f1": float(f_cls[0]),
            "ambiguous_precision": float(p_cls[1]),
            "ambiguous_recall": float(r_cls[1]),
            "ambiguous_f1": float(f_cls[1]),
            "unknown_precision": float(p_cls[2]),
            "unknown_recall": float(r_cls[2]),
            "unknown_f1": float(f_cls[2]),
            "mean_category_acc": mean_cat_acc,
            "std_category_acc": std_cat_acc,
            "confusion_matrix": str(cm),
        })

    df_overall = pd.DataFrame(overall_summary).sort_values(by="overall_macro_f1", ascending=False)
    summary_csv = os.path.join(RESULTS_DIR, "category_heldout_results.csv")
    df_overall.to_csv(summary_csv, index=False)
    logger.info("Saved category held-out summary to %s", summary_csv)

    # Per-Category Performance Breakdown Table
    cat_rows = []
    for cat in categories:
        class_label = [r["target_label"] for r in records if r["category"] == cat][0]
        row = {"category": cat, "target_class": class_label, "n_questions": 4}
        for m_name in model_defs:
            row[f"{m_name}_acc"] = per_cat_accuracies[m_name][cat]
            row[f"{m_name}_correct"] = per_cat_correct[m_name][cat]
        cat_rows.append(row)

    df_cat = pd.DataFrame(cat_rows)
    cat_csv = os.path.join(RESULTS_DIR, "per_category_results.csv")
    df_cat.to_csv(cat_csv, index=False)
    logger.info("Saved per-category results to %s", cat_csv)

    print("\n" + "="*80)
    print("CATEGORY-HELD-OUT GENERALIZATION SUMMARY (15 CATEGORIES, N=60)")
    print("="*80)
    print(df_overall[["model_name", "overall_accuracy", "overall_macro_f1", "balanced_accuracy", "mcc", "mean_category_acc", "std_category_acc"]].to_string())

    print("\n" + "="*80)
    print("PER-CATEGORY ACCURACY ACROSS MODELS")
    print("="*80)
    acc_cols = ["category", "target_class"] + [f"{m}_acc" for m in model_defs]
    print(df_cat[acc_cols].to_string())

    return df_overall, df_cat, oof_predictions

if __name__ == "__main__":
    run_category_heldout()
