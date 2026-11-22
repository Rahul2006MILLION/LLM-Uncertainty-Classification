#!/usr/bin/env python3
"""Stage 5: Statistical Analysis of Generalization and Lexical Shortcut Audit.

Calculates:
1. Question-level McNemar test (with Edwards' continuity correction) and exact binomial test.
2. Question-level non-parametric bootstrap (B=2000) for Macro-F1 differences with 95% percentile CIs.
3. Quantifies degradation between Standard CV and Category-Held-Out evaluation.
4. Confirmatory vs exploratory hypothesis testing.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")

logger = logging.getLogger("statistical_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_mcnemar_and_binomial(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray
) -> Dict[str, Any]:
    """Compute McNemar test and exact two-sided binomial test."""
    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)

    n11 = int(np.sum(corr_a & corr_b))
    n00 = int(np.sum((~corr_a) & (~corr_b)))
    n10 = int(np.sum(corr_a & (~corr_b))) # A wins
    n01 = int(np.sum((~corr_a) & corr_b)) # B wins
    discordant = n10 + n01

    if discordant > 0:
        chi2 = float(((abs(n10 - n01) - 1.0) ** 2) / discordant)
        mcnemar_p = float(1.0 - stats.chi2.cdf(chi2, df=1))
        binom_p = float(stats.binomtest(n10, discordant, p=0.5, alternative="two-sided").pvalue)
    else:
        chi2, mcnemar_p, binom_p = 0.0, 1.0, 1.0

    return {
        "n11_both_correct": n11,
        "n00_both_wrong": n00,
        "n10_a_wins": n10,
        "n01_b_wins": n01,
        "discordant": discordant,
        "mcnemar_chi2": chi2,
        "mcnemar_pvalue": mcnemar_p,
        "exact_binomial_pvalue": binom_p,
    }

def bootstrap_macro_f1_difference(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """Non-parametric question-level bootstrap for Macro-F1 difference."""
    np.random.seed(random_state)
    n = len(y_true)
    f1_a_list = []
    f1_b_list = []
    diff_list = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_samp = y_true[idx]
        if len(set(y_samp)) < 3:
            continue
        f1_a = f1_score(y_samp, preds_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(y_samp, preds_b[idx], average="macro", zero_division=0)
        f1_a_list.append(f1_a)
        f1_b_list.append(f1_b)
        diff_list.append(f1_a - f1_b)

    diff_arr = np.array(diff_list)
    mean_diff = float(np.mean(diff_arr))
    ci_low = float(np.percentile(diff_arr, 2.5))
    ci_high = float(np.percentile(diff_arr, 97.5))

    if mean_diff > 0:
        emp_p = float(np.mean(diff_arr <= 0) * 2.0)
    else:
        emp_p = float(np.mean(diff_arr >= 0) * 2.0)
    emp_p = min(1.0, emp_p)

    return {
        "mean_diff": mean_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "empirical_pvalue": emp_p,
        "f1_a_mean": float(np.mean(f1_a_list)),
        "f1_b_mean": float(np.mean(f1_b_list)),
    }

def run_statistical_analysis():
    oof_csv = os.path.join(RESULTS_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_csv):
        raise FileNotFoundError(f"OOF predictions file not found at {oof_csv}")

    df_oof = pd.read_csv(oof_csv)
    y_true = df_oof["target_label"].to_numpy()

    # Load category held-out predictions from Stage 2
    # Load and map models
    comparisons = [
        # 1. Primary Degradation: TFIDF LinearSVC Standard CV vs Category-Held-Out
        ("Generalization_Collapse_TFIDF_LinearSVC",
         "ExpA_Original_TFIDF_StandardCV", "ExpA_Original_TFIDF_CategoryHeldOut",
         "Standard CV vs Category-Held-Out for TF-IDF LinearSVC"),

        # 2. Lexical Shortcut Impact in Standard CV: Original TFIDF vs Epistemic-Only
        ("Lexical_Shortcut_CV_Original_vs_EpistemicOnly",
         "ExpA_Original_TFIDF_StandardCV", "ExpE_Epistemic_Only_TFIDF_StandardCV",
         "Standard CV: Original TF-IDF vs Epistemic-Only TF-IDF"),

        # 3. Lexical Shortcut Impact in Standard CV: Original TFIDF vs Domain-Neutral
        ("Lexical_Shortcut_CV_Original_vs_DomainNeutral",
         "ExpA_Original_TFIDF_StandardCV", "ExpG_Domain_Neutral_TFIDF_StandardCV",
         "Standard CV: Original TF-IDF vs Domain-Neutral TF-IDF"),

        # 4. Lexical Shortcut Impact in Category-Held-Out: Original vs Epistemic-Only
        ("Lexical_Shortcut_Heldout_Original_vs_EpistemicOnly",
         "ExpA_Original_TFIDF_CategoryHeldOut", "ExpE_Epistemic_Only_TFIDF_CategoryHeldOut",
         "Category-Held-Out: Original TF-IDF vs Epistemic-Only TF-IDF"),

        # 5. Entity Masking Impact under Category-Held-Out
        ("Entity_Masking_Heldout_Original_vs_EntityMasked",
         "ExpA_Original_TFIDF_CategoryHeldOut", "ExpC_Entity_Masked_TFIDF_CategoryHeldOut",
         "Category-Held-Out: Original TF-IDF vs Entity-Masked TF-IDF"),

        # 6. Domain Term Masking Impact under Category-Held-Out
        ("Domain_Term_Masking_Heldout_Original_vs_DomainMasked",
         "ExpA_Original_TFIDF_CategoryHeldOut", "ExpD_Domain_Term_Masked_TFIDF_CategoryHeldOut",
         "Category-Held-Out: Original TF-IDF vs Domain-Term-Masked TF-IDF"),

        # 7. Marker Counts vs Epistemic-Only under Category-Held-Out
        ("Marker_Counts_vs_Epistemic_Heldout",
         "ExpF_Marker_Counts_RandomForest_CategoryHeldOut", "ExpE_Epistemic_Only_TFIDF_CategoryHeldOut",
         "Category-Held-Out: Marker-Count RF vs Epistemic-Only TF-IDF"),
    ]

    stat_results = []
    print("\n" + "="*85)
    print("STATISTICAL COMPARISON OF GENERALIZATION AND SHORTCUT AUDIT")
    print("="*85)

    for comp_id, col_a, col_b, desc in comparisons:
        preds_a = df_oof[col_a].to_numpy()
        preds_b = df_oof[col_b].to_numpy()

        acc_a = float(accuracy_score(y_true, preds_a))
        acc_b = float(accuracy_score(y_true, preds_b))
        f1_a = float(f1_score(y_true, preds_a, average="macro", zero_division=0))
        f1_b = float(f1_score(y_true, preds_b, average="macro", zero_division=0))

        mc = compute_mcnemar_and_binomial(y_true, preds_a, preds_b)
        bs = bootstrap_macro_f1_difference(y_true, preds_a, preds_b, n_bootstrap=2000)

        stat_results.append({
            "comparison_id": comp_id,
            "description": desc,
            "condition_a": col_a,
            "condition_b": col_b,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "acc_diff": acc_a - acc_b,
            "f1_a": f1_a,
            "f1_b": f1_b,
            "f1_diff": f1_a - f1_b,
            "bootstrap_mean_f1_diff": bs["mean_diff"],
            "bootstrap_95_ci_low": bs["ci_95_low"],
            "bootstrap_95_ci_high": bs["ci_95_high"],
            "bootstrap_pvalue": bs["empirical_pvalue"],
            "mcnemar_chi2": mc["mcnemar_chi2"],
            "mcnemar_pvalue": mc["mcnemar_pvalue"],
            "exact_binomial_pvalue": mc["exact_binomial_pvalue"],
            "n_a_wins": mc["n10_a_wins"],
            "n_b_wins": mc["n01_b_wins"],
            "n_discordant": mc["discordant"],
        })

        print(f"\n{desc}:")
        print(f"  Acc A: {acc_a*100:.1f}%, Acc B: {acc_b*100:.1f}%, Diff: {(acc_a-acc_b)*100:+.1f}%")
        print(f"  Macro-F1 A: {f1_a:.4f}, Macro-F1 B: {f1_b:.4f}, Diff: {f1_a-f1_b:+.4f}")
        print(f"  Bootstrap 95% CI: [{bs['ci_95_low']:+.4f}, {bs['ci_95_high']:+.4f}], Emp p = {bs['empirical_pvalue']:.4f}")
        print(f"  McNemar chi2 = {mc['mcnemar_chi2']:.4f}, p = {mc['mcnemar_pvalue']:.4f}, Binomial p = {mc['exact_binomial_pvalue']:.4f}")
        print(f"  Discordant: {mc['discordant']} (A wins: {mc['n10_a_wins']}, B wins: {mc['n01_b_wins']})")

    df_stat = pd.DataFrame(stat_results)
    stat_csv = os.path.join(RESULTS_DIR, "statistical_comparisons.csv")
    df_stat.to_csv(stat_csv, index=False)
    logger.info("Saved statistical comparisons to %s", stat_csv)

    return df_stat

if __name__ == "__main__":
    run_statistical_analysis()
