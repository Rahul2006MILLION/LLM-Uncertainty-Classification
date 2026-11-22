#!/usr/bin/env python3
"""Stage 3 & 4: Lexical Shortcut Audit across Two Evaluation Protocols.

Experiments:
A. Original Text (reference)
B. Stopwords Removed (protecting epistemic words)
C. Named Entity Masking
D. Domain-Term Masking
E. Epistemic-Only Representation
F. Marker-Count Feature Representation (evaluated with LR, Linear SVM, RF)
G. Domain-Neutral TF-IDF (Entity Masked + Domain-Term Masked + Stopwords Removed)

Evaluated under TWO Protocols:
1. Protocol 1: Standard Grouped CV (5-Fold StratifiedGroupKFold on question_group_id)
2. Protocol 2: Category-Held-Out Generalization (15-Fold Leave-One-Category-Out)
"""

import os
import sys
import re
import json
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = logging.getLogger("lexical_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Lexicons and Patterns
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "by", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "which", "while", "who",
    "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves"
}

# Protected uncertainty / epistemic words that MUST NOT be removed
PROTECTED_EPISTEMIC_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "cant", "don't",
    "doesnt", "wont", "isnt", "arent", "wasnt", "werent", "hardly", "scarcely",
    "barely", "unknown", "uncertain", "unclear", "unproven", "unresolved",
    "impossible", "maybe", "perhaps", "possibly", "probably", "likely",
    "unlikely", "might", "could", "may", "if", "whether", "depends", "depending",
    "assuming", "ambiguous", "definitely", "certainly", "precisely", "undoubtedly",
    "exact", "always", "consensus", "question", "open", "interpretation",
    "determine", "definitive", "insufficient", "debated", "conjecture",
    "hypothetical", "established", "proven", "known", "clearly"
}

FILTERED_STOPWORDS = STANDARD_STOPWORDS - PROTECTED_EPISTEMIC_WORDS

NAMED_ENTITY_REGEXES = [
    (r"\b(1[0-9]{3}|20[0-2][0-9])\b", "[DATE]"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b", "[DATE]"),
    (r"\b(United\s+States|United\s+Kingdom|Great\s+Britain|Soviet\s+Union|European\s+Union|Roman\s+Empire|Ottoman\s+Empire)\b", "[LOCATION]"),
    (r"\b(France|Germany|Spain|Italy|Russia|China|Japan|India|Serbia|America|British|French|German|Russian|Spanish|Serbian-American|Roman|Greek|Byzantine)\b", "[LOCATION]"),
    (r"\b(Treaty\s+of\s+[A-Z][a-z]+|Peace\s+of\s+[A-Z][a-z]+|Act\s+of\s+[A-Z][a-z]+|Battle\s+of\s+[A-Z][a-z]+)\b", "[HISTORICAL_EVENT]"),
    (r"\b(Albert\s+Einstein|Isaac\s+Newton|Nikola\s+Tesla|Marie\s+Curie|Alexander\s+Fleming|Robert\s+Walpole|George\s+Washington|Thomas\s+Jefferson|Abraham\s+Lincoln)\b", "[PERSON]"),
    (r"\b(Kelvin|Tesla|Newton|Joule|Watt|Pascal|Volt|Ampere|Ohm|Hertz)\b", "[NAMED_UNIT]"),
    (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "[ENTITY]"),
]

DOMAIN_TERM_REGEXES = [
    # Units and measurements
    (r"\b(kelvin|celsius|fahrenheit|tesla|meters?|seconds?|joules?|watts?|grams?|kilograms?|volts?|amperes?|km|mph|m/s|m/s\^2|kg|mol|k)\b", "[UNIT]"),
    # Biological and chemical terms
    (r"\b(transpeptidase|peptidoglycan|penicillin|enzyme|enzymes|bacterial|bacteria|cell\s+wall|synthesis|photosynthesis|chloroplast|mitochondria|dna|rna|protein|proteins|amino\s+acid|cross-linking|beta-lactam)\b", "[BIO_TERM]"),
    # Math & physics
    (r"\b(velocity|acceleration|speed|distance|gravity|mass|friction|dice|die|probability|fraction|ratio|roll|rolls|sum)\b", "[MATH_PHYS_TERM]"),
    # History & politics
    (r"\b(treaty|treaties|president|prime\s+minister|monarch|king|queen|dynasty|empire|battle|war|parliament|congress|chancellor|minister|foreign\s+policy|sovereignty)\b", "[POL_HIST_TERM]"),
    # Geography
    (r"\b(city|capital|country|ocean|sea|river|mountain|peak|island|continent|strait|peninsula)\b", "[GEO_TERM]"),
    # Trade & economy
    (r"\b(agricultural|agriculture|export|import|tariff|tariffs|subsidy|subsidies|trade|economy|economic|goods|wheat|corn|grain|commodities)\b", "[ECON_TERM]"),
]

HEDGING_PATTERNS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b", r"\blikely\b",
    r"\bunlikely\b", r"\bmight\b", r"\bcould\b", r"\bmay\b", r"\bsuggests?\b",
    r"\bseems?\b", r"\bappears?\b", r"\broughly\b", r"\bapproximately\b"
]

UNCERTAINTY_PATTERNS = [
    r"\buncertain\b", r"\bunclear\b", r"\bunproven\b", r"\bunresolved\b",
    r"\bunknown\b", r"\bdebated\b", r"\bhypothetical\b", r"\bambiguous\b",
    r"\bopen\s+question\b", r"\bno\s+consensus\b", r"\binsufficient\s+information\b",
    r"\bno\s+definitive\b", r"\bremains\s+unknown\b"
]

CONFIDENCE_PATTERNS = [
    r"\bdefinitely\b", r"\bcertainly\b", r"\bprecisely\b", r"\bundoubtedly\b",
    r"\bexact\b", r"\balways\b", r"\bproven\b", r"\bknown\b", r"\bclearly\b",
    r"\bestablished\b", r"\binvariably\b", r"\bconclusively\b"
]

REFUSAL_PATTERNS = [
    r"\bi\s+do\s+not\s+know\b", r"\bcannot\s+determine\b", r"\bno\s+one\s+knows\b",
    r"\bunknown\s+to\s+science\b", r"\bcannot\s+be\s+definitively\b"
]

MODAL_PATTERNS = [
    r"\bcan\b", r"\bcould\b", r"\bmay\b", r"\bmight\b", r"\bshall\b",
    r"\bshould\b", r"\bwill\b", r"\bwould\b", r"\bmust\b"
]

CONDITIONAL_PATTERNS = [
    r"\bif\b", r"\bdepends\s+on\b", r"\bdepending\s+on\b", r"\bassuming\b",
    r"\bwhether\b", r"\bcontingent\s+upon\b"
]

ALTERNATIVE_PATTERNS = [
    r"\balternatively\b", r"\beither\b", r"\bor\b", r"\bon\s+the\s+other\s+hand\b",
    r"\banother\s+interpretation\b", r"\bmultiple\s+meanings\b", r"\btwo\s+interpretations\b"
]

def transform_stopwords_removed(text: str) -> str:
    words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)
    filtered = [w for w in words if w.lower() not in FILTERED_STOPWORDS]
    return " ".join(filtered)

def transform_named_entities_masked(text: str) -> str:
    masked = text
    for pat, rep in NAMED_ENTITY_REGEXES:
        masked = re.sub(pat, rep, masked, flags=re.IGNORECASE if "[ENTITY]" not in rep else 0)
    return masked

def transform_domain_terms_masked(text: str) -> str:
    masked = text
    for pat, rep in DOMAIN_TERM_REGEXES:
        masked = re.sub(pat, rep, masked, flags=re.IGNORECASE)
    return masked

def transform_epistemic_only(text: str) -> str:
    """Extract only epistemic indicators, modals, certainty markers, and their immediate context."""
    lower = text.lower()
    extracted_tokens = []
    
    # Tokenize words
    words = re.findall(r"\b[a-z0-9_\'-]+\b", lower)
    for i, w in enumerate(words):
        if w in PROTECTED_EPISTEMIC_WORDS or any(re.match(pat, w) for pat in HEDGING_PATTERNS + UNCERTAINTY_PATTERNS + CONFIDENCE_PATTERNS):
            # Include window of 1 preceding word and 1 succeeding word
            start = max(0, i - 1)
            end = min(len(words), i + 2)
            extracted_tokens.append(" ".join(words[start:end]))
            
    if not extracted_tokens:
        return "neutral_epistemic_statement"
    return " ; ".join(extracted_tokens)

def extract_marker_count_vector(response_text: str, question_text: str) -> np.ndarray:
    """Create 12-dimensional count/density vector of epistemic and structural features."""
    lower = response_text.lower()
    words = re.findall(r"\b[a-z0-9_\'-]+\b", lower)
    n_words = max(1, len(words))
    q_words = set(re.findall(r"\b[a-z0-9_\'-]+\b", question_text.lower()))
    
    h_count = sum(len(re.findall(pat, lower)) for pat in HEDGING_PATTERNS)
    u_count = sum(len(re.findall(pat, lower)) for pat in UNCERTAINTY_PATTERNS)
    c_count = sum(len(re.findall(pat, lower)) for pat in CONFIDENCE_PATTERNS)
    n_count = sum(1 for w in words if w in {"no", "not", "never", "none", "neither", "nor", "cannot", "cant"})
    r_count = sum(len(re.findall(pat, lower)) for pat in REFUSAL_PATTERNS)
    kl_count = 1.0 if any(re.search(pat, lower) for pat in REFUSAL_PATTERNS + UNCERTAINTY_PATTERNS) else 0.0
    modal_count = sum(len(re.findall(pat, lower)) for pat in MODAL_PATTERNS)
    resp_len = float(len(response_text))
    overlap = float(len(set(words) & q_words) / len(set(words) | q_words)) if (set(words) | q_words) else 0.0
    
    sentences = [s for s in re.split(r"[.!?]+", response_text) if s.strip()]
    sent_count = float(max(1, len(sentences)))
    cond_count = sum(len(re.findall(pat, lower)) for pat in CONDITIONAL_PATTERNS)
    alt_count = sum(len(re.findall(pat, lower)) for pat in ALTERNATIVE_PATTERNS)
    
    return np.array([
        h_count, u_count, c_count, n_count, r_count, kl_count,
        modal_count, resp_len, overlap, sent_count, cond_count, alt_count
    ], dtype=np.float32)

def transform_domain_neutral_tfidf(text: str) -> str:
    """Mask entities, mask domain terms, and remove non-epistemic stopwords."""
    masked = transform_named_entities_masked(text)
    masked = transform_domain_terms_masked(masked)
    cleaned = transform_stopwords_removed(masked)
    return cleaned

def evaluate_representation_under_cv(
    representation_data: Any, # List[str] if text, np.ndarray if dense
    is_dense: bool,
    clf_factory: Any,
    y: np.ndarray,
    groups: List[str]
) -> Dict[str, Any]:
    """Protocol 1: 5-Fold StratifiedGroupKFold on question_group_id."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(y)
    oof_preds = [None] * n
    fold_accs = []
    fold_f1s = []

    for train_idx, test_idx in cv.split(representation_data, y, groups):
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = clf_factory()

        if not is_dense:
            vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2, sublinear_tf=True)
            tr_texts = [representation_data[i] for i in train_idx]
            te_texts = [representation_data[i] for i in test_idx]
            X_tr = vec.fit_transform(tr_texts)
            X_te = vec.transform(te_texts)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
        else:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(representation_data[train_idx])
            X_te = scaler.transform(representation_data[test_idx])
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

        for idx, p in zip(test_idx, preds):
            oof_preds[idx] = p
        fold_accs.append(float(accuracy_score(y_te, preds)))
        fold_f1s.append(float(f1_score(y_te, preds, average="macro", zero_division=0)))

    acc = float(accuracy_score(y, oof_preds))
    f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, oof_preds))
    mcc = float(matthews_corrcoef(y, oof_preds))

    return {
        "oof_accuracy": acc,
        "oof_macro_f1": f1,
        "mean_fold_acc": float(np.mean(fold_accs)),
        "std_fold_acc": float(np.std(fold_accs, ddof=1)),
        "mean_fold_f1": float(np.mean(fold_f1s)),
        "std_fold_f1": float(np.std(fold_f1s, ddof=1)),
        "balanced_acc": bal_acc,
        "mcc": mcc,
        "oof_preds": oof_preds,
    }

def evaluate_representation_under_heldout(
    representation_data: Any,
    is_dense: bool,
    clf_factory: Any,
    y: np.ndarray,
    categories: List[str],
    cat_list: List[str]
) -> Dict[str, Any]:
    """Protocol 2: 15-Fold Leave-One-Category-Out."""
    n = len(y)
    oof_preds = [None] * n
    cat_accs = {}

    for cat in cat_list:
        test_mask = np.array([c == cat for c in categories])
        train_mask = ~test_mask
        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]

        y_tr, y_te = y[tr_idx], y[te_idx]
        clf = clf_factory()

        if not is_dense:
            vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2, sublinear_tf=True)
            tr_texts = [representation_data[i] for i in tr_idx]
            te_texts = [representation_data[i] for i in te_idx]
            X_tr = vec.fit_transform(tr_texts)
            X_te = vec.transform(te_texts)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
        else:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(representation_data[tr_idx])
            X_te = scaler.transform(representation_data[te_idx])
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

        for idx, p in zip(te_idx, preds):
            oof_preds[idx] = p
        cat_accs[cat] = float(np.mean(preds == y_te))

    acc = float(accuracy_score(y, oof_preds))
    f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y, oof_preds))
    mcc = float(matthews_corrcoef(y, oof_preds))

    return {
        "oof_accuracy": acc,
        "oof_macro_f1": f1,
        "balanced_acc": bal_acc,
        "mcc": mcc,
        "cat_accs": cat_accs,
        "mean_cat_acc": float(np.mean(list(cat_accs.values()))),
        "std_cat_acc": float(np.std(list(cat_accs.values()), ddof=1)),
        "oof_preds": oof_preds,
    }

def run_lexical_audit():
    unified_path = os.path.join(PROJECT_ROOT, "data", "batch_2_combined_unified.jsonl")
    with open(unified_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    n = len(records)
    y = np.array([r["target_label"] for r in records])
    groups = [r["question_group_id"] for r in records]
    categories = [r["category"] for r in records]
    cat_list = sorted(list(set(categories)))

    raw_responses = [r["responses"][0]["text"] for r in records]
    raw_questions = [r["question"] for r in records]

    logger.info("Generating transformed representations for %d records...", n)
    texts_stopwords = [transform_stopwords_removed(t) for t in raw_responses]
    texts_entity_masked = [transform_named_entities_masked(t) for t in raw_responses]
    texts_domain_masked = [transform_domain_terms_masked(t) for t in raw_responses]
    texts_epistemic_only = [transform_epistemic_only(t) for t in raw_responses]
    X_marker_counts = np.array([extract_marker_count_vector(t, q) for t, q in zip(raw_responses, raw_questions)])
    texts_domain_neutral = [transform_domain_neutral_tfidf(t) for t in raw_responses]

    representations = {
        "ExpA_Original_TFIDF": {
            "data": raw_responses,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Original raw response text with standard TF-IDF (Reference condition)",
        },
        "ExpB_Stopwords_Removed_TFIDF": {
            "data": texts_stopwords,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Stopwords removed while protecting epistemic/uncertainty tokens",
        },
        "ExpC_Entity_Masked_TFIDF": {
            "data": texts_entity_masked,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Named entities (persons, locations, organizations, dates) masked",
        },
        "ExpD_Domain_Term_Masked_TFIDF": {
            "data": texts_domain_masked,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Domain-specific jargon, units, chemical, and historical terms masked",
        },
        "ExpE_Epistemic_Only_TFIDF": {
            "data": texts_epistemic_only,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Text filtered strictly to epistemic markers, hedging words, and modals",
        },
        "ExpF_Marker_Counts_LinearSVC": {
            "data": X_marker_counts,
            "is_dense": True,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "12-dimensional dense count vector of epistemic/structural markers with LinearSVC",
        },
        "ExpF_Marker_Counts_LogisticReg": {
            "data": X_marker_counts,
            "is_dense": True,
            "clf_factory": lambda: LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            "description": "12-dimensional dense count vector with Logistic Regression",
        },
        "ExpF_Marker_Counts_RandomForest": {
            "data": X_marker_counts,
            "is_dense": True,
            "clf_factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
            "description": "12-dimensional dense count vector with Random Forest",
        },
        "ExpG_Domain_Neutral_TFIDF": {
            "data": texts_domain_neutral,
            "is_dense": False,
            "clf_factory": lambda: LinearSVC(dual="auto", random_state=42, max_iter=2000, C=1.0),
            "description": "Entities masked + domain terms masked + non-epistemic stopwords removed",
        },
    }

    results_table = []
    oof_predictions_all = {}

    logger.info("Evaluating all 9 representations under Protocol 1 (Standard CV) and Protocol 2 (Category-Held-Out)...")
    for exp_id, exp_cfg in representations.items():
        # Protocol 1: Standard Grouped CV
        res_cv = evaluate_representation_under_cv(
            representation_data=exp_cfg["data"],
            is_dense=exp_cfg["is_dense"],
            clf_factory=exp_cfg["clf_factory"],
            y=y,
            groups=groups
        )
        oof_predictions_all[f"{exp_id}_StandardCV"] = res_cv["oof_preds"]

        # Protocol 2: Category-Held-Out Evaluation
        res_heldout = evaluate_representation_under_heldout(
            representation_data=exp_cfg["data"],
            is_dense=exp_cfg["is_dense"],
            clf_factory=exp_cfg["clf_factory"],
            y=y,
            categories=categories,
            cat_list=cat_list
        )
        oof_predictions_all[f"{exp_id}_CategoryHeldOut"] = res_heldout["oof_preds"]

        acc_drop = res_cv["oof_accuracy"] - res_heldout["oof_accuracy"]
        f1_drop = res_cv["oof_macro_f1"] - res_heldout["oof_macro_f1"]

        results_table.append({
            "experiment": exp_id,
            "description": exp_cfg["description"],
            "cv_accuracy": res_cv["oof_accuracy"],
            "cv_macro_f1": res_cv["oof_macro_f1"],
            "cv_mcc": res_cv["mcc"],
            "heldout_accuracy": res_heldout["oof_accuracy"],
            "heldout_macro_f1": res_heldout["oof_macro_f1"],
            "heldout_mcc": res_heldout["mcc"],
            "accuracy_drop": acc_drop,
            "macro_f1_drop": f1_drop,
            "heldout_mean_cat_acc": res_heldout["mean_cat_acc"],
            "heldout_std_cat_acc": res_heldout["std_cat_acc"],
        })

    df_results = pd.DataFrame(results_table)
    out_csv = os.path.join(RESULTS_DIR, "lexical_audit_results.csv")
    df_results.to_csv(out_csv, index=False)
    logger.info("Saved lexical audit results to %s", out_csv)

    # Save out-of-fold predictions
    oof_rows = []
    for i, r in enumerate(records):
        row = {
            "question_id": r["id"],
            "target_label": r["target_label"],
            "category": r["category"],
            "question_group_id": r["question_group_id"],
        }
        for k in oof_predictions_all:
            row[k] = oof_predictions_all[k][i]
        oof_rows.append(row)
    df_oof = pd.DataFrame(oof_rows)
    oof_csv = os.path.join(RESULTS_DIR, "oof_predictions.csv")
    df_oof.to_csv(oof_csv, index=False)
    logger.info("Saved all OOF predictions to %s", oof_csv)

    print("\n" + "="*90)
    print("LEXICAL SHORTCUT AUDIT RESULTS (STANDARD CV vs CATEGORY-HELD-OUT)")
    print("="*90)
    print(df_results[["experiment", "cv_accuracy", "cv_macro_f1", "heldout_accuracy", "heldout_macro_f1", "accuracy_drop", "macro_f1_drop"]].to_string())

    return df_results, oof_predictions_all

if __name__ == "__main__":
    run_lexical_audit()
