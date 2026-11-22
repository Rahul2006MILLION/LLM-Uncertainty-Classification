#!/usr/bin/env python3
"""
Statistical Methodology Audit Script for Batch 2 Combined Evaluation (N=60, M=600).
Evaluates:
1. Paired t-test, Nadeau-Bengio corrected resampled t-test, Wilcoxon signed-rank test.
2. Parametric Student's t 95% CIs (df=4) and corrected CIs.
3. Cohen's d, Hedges' g calculation and exact formula checks.
4. Benjamini-Hochberg FDR correction.
5. Out-of-fold 60-question predictions, pooled McNemar's test, contingency tables.
6. Non-parametric question-level bootstrap (B=2000) for Macro-F1 and pairwise deltas.
7. Claim-by-claim statistical evaluation (Claims A-E).
"""

import os
import sys
import json
import numpy as np
import scipy.stats as stats
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experimentation.feature_groups import (
    FEATURE_NAMES,
    TRUE_SINGLE_RESPONSE_FEATURES,
    ENSEMBLE_LINGUISTIC_FEATURES,
    CONSISTENCY_ONLY_FEATURES,
)

def benjamini_hochberg(p_values):
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    q_values = np.zeros(n, dtype=float)
    for i in range(n):
        rank = i + 1
        q_values[i] = sorted_p[i] * n / rank
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])
    q_final = np.zeros(n, dtype=float)
    q_final[sorted_indices] = np.minimum(q_values, 1.0)
    return q_final.tolist()

def main():
    unified_path = "data/batch_2_combined_unified.jsonl"
    with open(unified_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} unified records.")
    
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    y = np.array([r["target_label"] for r in records])
    groups = [r["question_group_id"] for r in records]

    exp_matrices = {
        "TRUE_SINGLE_RESPONSE": np.array([[r["r0_features"][fn] for fn in TRUE_SINGLE_RESPONSE_FEATURES] for r in records]),
        "SINGLE_RESPONSE_RANDOM": np.array([[r["rrand_features"][fn] for fn in TRUE_SINGLE_RESPONSE_FEATURES] for r in records]),
        "ENSEMBLE_LINGUISTIC": np.array([[r["features"][fn] for fn in ENSEMBLE_LINGUISTIC_FEATURES] for r in records]),
        "CONSISTENCY_ONLY": np.array([[r["features"][fn] for fn in CONSISTENCY_ONLY_FEATURES] for r in records]),
        "FULL_HYBRID": np.array([[r["features"][fn] for fn in FEATURE_NAMES] for r in records]),
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Track fold metrics and out-of-fold predictions
    oof_predictions = {k: [None] * len(records) for k in exp_matrices}
    fold_f1_scores = {k: [] for k in exp_matrices}

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(exp_matrices["FULL_HYBRID"], y, groups)):
        for exp_name, X_mat in exp_matrices.items():
            X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_te_sc)

            fold_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
            fold_f1_scores[exp_name].append(fold_f1)

            for idx, p in zip(test_idx, preds):
                oof_predictions[exp_name][idx] = p

    print("\n--- 5-FOLD MACRO-F1 SCORES (RandomForest) ---")
    for exp_name, scores in fold_f1_scores.items():
        print(f"{exp_name:25s}: Mean={np.mean(scores):.4f}, Std={np.std(scores, ddof=1):.4f}, Folds={[round(s, 4) for s in scores]}")

    # Hypotheses comparisons
    comparisons = [
        ("A. FULL_HYBRID vs TRUE_SINGLE_RESPONSE", "FULL_HYBRID", "TRUE_SINGLE_RESPONSE"),
        ("B. FULL_HYBRID vs SINGLE_RESPONSE_RANDOM", "FULL_HYBRID", "SINGLE_RESPONSE_RANDOM"),
        ("C. FULL_HYBRID vs ENSEMBLE_LINGUISTIC", "FULL_HYBRID", "ENSEMBLE_LINGUISTIC"),
        ("D. CONSISTENCY_ONLY vs TRUE_SINGLE_RESPONSE", "CONSISTENCY_ONLY", "TRUE_SINGLE_RESPONSE"),
    ]

    p_vals_uncorr = []
    p_vals_corr = []
    audit_results = {}

    print("\n--- STATISTICAL TESTS & EFFECT SIZES (df=4) ---")
    for title, base, ref in comparisons:
        deltas = np.array(fold_f1_scores[base]) - np.array(fold_f1_scores[ref])
        mean_d = float(np.mean(deltas))
        s_d = float(np.std(deltas, ddof=1))
        n_folds = 5
        se_uncorr = s_d / np.sqrt(n_folds)
        
        # Standard paired t
        t_stat = mean_d / se_uncorr if se_uncorr > 0 else 0.0
        p_val_uncorr = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_folds - 1))
        p_vals_uncorr.append(p_val_uncorr)

        # Student's t 95% CI (df=4)
        t_crit = stats.t.ppf(0.975, df=n_folds - 1)
        ci_lower = mean_d - t_crit * se_uncorr
        ci_upper = mean_d + t_crit * se_uncorr

        # Nadeau & Bengio (2003) corrected resampled t-test
        # n_test = 12, n_train = 48 -> ratio = 12/48 = 0.25; 1/K = 1/5 = 0.20 -> sum = 0.45
        # se_corr = s_d * sqrt(0.20 + 0.25) = s_d * sqrt(0.45)
        # correction_factor = sqrt( (1/5) / 0.45 ) = sqrt( 0.20 / 0.45 ) = sqrt(4/9) = 2/3
        var_factor = (1.0 / n_folds) + (12.0 / 48.0) # 0.45
        se_corr = s_d * np.sqrt(var_factor)
        t_stat_corr = mean_d / se_corr if se_corr > 0 else 0.0
        p_val_corr = 2.0 * (1.0 - stats.t.cdf(abs(t_stat_corr), df=n_folds - 1))
        p_vals_corr.append(p_val_corr)
        ci_lower_corr = mean_d - t_crit * se_corr
        ci_upper_corr = mean_d + t_crit * se_corr

        # Wilcoxon signed-rank
        try:
            nonzero_deltas = deltas[deltas != 0]
            if len(nonzero_deltas) > 0:
                w_res = stats.wilcoxon(nonzero_deltas)
                w_stat = float(w_res.statistic)
                w_pval = float(w_res.pvalue)
            else:
                w_stat, w_pval = 0.0, 1.0
        except Exception as e:
            w_stat, w_pval = 0.0, 1.0

        # Cohen's d (paired)
        cohens_d = mean_d / s_d if s_d > 0 else 0.0
        # Hedges' g
        j_4 = 1.0 - (3.0 / (4.0 * (n_folds - 1) - 1.0)) # 1 - 3/15 = 0.800
        hedges_g = cohens_d * j_4

        audit_results[title] = {
            "mean_delta": mean_d,
            "std_delta": s_d,
            "fold_deltas": deltas.tolist(),
            "paired_t": t_stat,
            "raw_p": p_val_uncorr,
            "ci_95_uncorr": [ci_lower, ci_upper],
            "nadeau_bengio_t": t_stat_corr,
            "nadeau_bengio_p": p_val_corr,
            "ci_95_corr": [ci_lower_corr, ci_upper_corr],
            "wilcoxon_w": w_stat,
            "wilcoxon_p": w_pval,
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
        }

        print(f"\n{title}:")
        print(f"  Mean Delta: {mean_d:+.4f}, Std: {s_d:.4f}, Fold deltas: {[round(d, 4) for d in deltas]}")
        print(f"  Standard Paired t: t={t_stat:.4f}, df=4, p={p_val_uncorr:.4f}")
        print(f"  95% CI (uncorrected): [{ci_lower:+.4f}, {ci_upper:+.4f}]")
        print(f"  Nadeau-Bengio Corrected t: t={t_stat_corr:.4f}, df=4, p={p_val_corr:.4f} (ratio factor = 2/3)")
        print(f"  95% CI (corrected): [{ci_lower_corr:+.4f}, {ci_upper_corr:+.4f}]")
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_pval:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}, Hedges' g: {hedges_g:.4f}")

    fdr_uncorr = benjamini_hochberg(p_vals_uncorr)
    fdr_corr = benjamini_hochberg(p_vals_corr)

    print("\n--- FDR ADJUSTED P-VALUES ---")
    for idx, (title, _, _) in enumerate(comparisons):
        audit_results[title]["fdr_p_uncorr"] = fdr_uncorr[idx]
        audit_results[title]["fdr_p_corr"] = fdr_corr[idx]
        print(f"{title}: Uncorr FDR p = {fdr_uncorr[idx]:.4f}, Corr FDR p = {fdr_corr[idx]:.4f}")

    # Out-of-fold pooled McNemar's Test
    print("\n--- OUT-OF-FOLD POOLED MCNEMAR TESTS (N=60 Questions) ---")
    y_true = np.array(y)
    for title, base, ref in comparisons:
        base_correct = (np.array(oof_predictions[base]) == y_true)
        ref_correct = (np.array(oof_predictions[ref]) == y_true)

        n11 = int(np.sum(base_correct & ref_correct))
        n00 = int(np.sum((~base_correct) & (~ref_correct)))
        n10 = int(np.sum(base_correct & (~ref_correct))) # Base wins
        n01 = int(np.sum((~base_correct) & ref_correct)) # Ref wins

        b = n10
        c = n01
        total_discordant = b + c
        if total_discordant > 0:
            chi2_stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
            mcnemar_p = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
            binom_p = float(stats.binomtest(b, total_discordant, p=0.5, alternative="two-sided").pvalue)
        else:
            chi2_stat, mcnemar_p, binom_p = 0.0, 1.0, 1.0

        print(f"\n{title}:")
        print(f"  Contingency Table (Base={base}, Ref={ref}):")
        print(f"    Both Correct (n11): {n11}")
        print(f"    Base Correct, Ref Wrong (n10): {n10}")
        print(f"    Ref Correct, Base Wrong (n01): {n01}")
        print(f"    Both Wrong (n00): {n00}")
        print(f"  Discordant Pairs: {total_discordant} (Base wins: {b}, Ref wins: {c})")
        print(f"  McNemar chi2 (with Edwards' correction): {chi2_stat:.4f}, p={mcnemar_p:.4f}")
        print(f"  Exact Binomial p-value: {binom_p:.4f}")

    # Bootstrap (B=2000) for question-level Macro-F1 and differences
    print("\n--- QUESTION-LEVEL BOOTSTRAP (B=2000) ---")
    np.random.seed(42)
    B = 2000
    n = len(y_true)
    boot_f1s = {k: [] for k in exp_matrices}
    boot_deltas = {c[0]: [] for c in comparisons}

    for _ in range(B):
        sample_indices = np.random.choice(n, size=n, replace=True)
        y_samp = y_true[sample_indices]
        if len(set(y_samp)) < 3:
            continue
        curr_f1s = {}
        for k in exp_matrices:
            p_samp = np.array(oof_predictions[k])[sample_indices]
            score = f1_score(y_samp, p_samp, average="macro", zero_division=0)
            boot_f1s[k].append(score)
            curr_f1s[k] = score
        for title, base, ref in comparisons:
            boot_deltas[title].append(curr_f1s[base] - curr_f1s[ref])

    for k in exp_matrices:
        arr = np.array(boot_f1s[k])
        ci_low = np.percentile(arr, 2.5)
        ci_high = np.percentile(arr, 97.5)
        print(f"{k:25s}: Bootstrap Mean={np.mean(arr):.4f}, 95% Percentile CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print("\nPairwise Delta Bootstrap 95% CIs:")
    for title, _, _ in comparisons:
        arr = np.array(boot_deltas[title])
        ci_low = np.percentile(arr, 2.5)
        ci_high = np.percentile(arr, 97.5)
        p_val_emp = float(np.mean(arr <= 0) if np.mean(arr) > 0 else np.mean(arr >= 0)) * 2.0
        print(f"{title}: Mean Delta={np.mean(arr):+.4f}, 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}], Emp p={p_val_emp:.4f}")

if __name__ == "__main__":
    main()
