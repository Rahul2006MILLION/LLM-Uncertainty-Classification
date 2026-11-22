"""Statistical Analysis Module for Single-Response Accuracy Benchmark.

Implements:
1. Fold-level comparisons: Paired t-test, Nadeau-Bengio corrected resampled t-test, Wilcoxon signed-rank test, Cohen's d, Hedges' g.
2. Question-level out-of-fold comparisons: 2x2 contingency matrix, McNemar test (with Edwards' correction), exact binomial test.
3. Non-parametric question-level bootstrap (B=2000) for Macro-F1 deltas with 95% percentile CI.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

def compute_paired_effect_sizes(diffs: np.ndarray) -> Tuple[float, float]:
    """Compute paired Cohen's d and Hedges' g."""
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0
    mean_d = float(np.mean(diffs))
    s_d = float(np.std(diffs, ddof=1))
    if s_d == 0.0:
        return 0.0, 0.0
    d = float(mean_d / s_d)
    j_factor = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0)) if (4.0 * (n - 1) - 1.0) > 0 else 1.0
    g = float(d * j_factor)
    return d, g

def compare_fold_metrics(
    candidate_scores: List[float],
    baseline_scores: List[float],
    n_test: int = 12,
    n_train: int = 48
) -> Dict[str, Any]:
    """Perform fold-level paired t-test, Nadeau-Bengio corrected t-test, and Wilcoxon signed-rank test."""
    deltas = np.array(candidate_scores) - np.array(baseline_scores)
    n_folds = len(deltas)
    mean_delta = float(np.mean(deltas))
    s_delta = float(np.std(deltas, ddof=1))
    
    se_uncorr = s_delta / np.sqrt(n_folds) if n_folds > 0 else 0.0
    t_uncorr = mean_delta / se_uncorr if se_uncorr > 0 else 0.0
    p_uncorr = 2.0 * (1.0 - stats.t.cdf(abs(t_uncorr), df=n_folds - 1)) if n_folds > 1 else 1.0
    
    t_crit = float(stats.t.ppf(0.975, df=n_folds - 1)) if n_folds > 1 else 0.0
    ci_uncorr = [float(mean_delta - t_crit * se_uncorr), float(mean_delta + t_crit * se_uncorr)]
    
    # Nadeau & Bengio (2003) correction
    var_factor = (1.0 / n_folds) + (float(n_test) / float(n_train))
    se_corr = s_delta * np.sqrt(var_factor)
    t_corr = mean_delta / se_corr if se_corr > 0 else 0.0
    p_corr = 2.0 * (1.0 - stats.t.cdf(abs(t_corr), df=n_folds - 1)) if n_folds > 1 else 1.0
    ci_corr = [float(mean_delta - t_crit * se_corr), float(mean_delta + t_crit * se_corr)]
    
    # Wilcoxon signed-rank
    try:
        nonzero = deltas[deltas != 0]
        if len(nonzero) > 0:
            w_res = stats.wilcoxon(nonzero)
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
        else:
            w_stat, w_pval = 0.0, 1.0
    except Exception:
        w_stat, w_pval = 0.0, 1.0
        
    cohens_d, hedges_g = compute_paired_effect_sizes(deltas)
    
    return {
        "mean_delta": mean_delta,
        "std_delta": s_delta,
        "fold_deltas": deltas.tolist(),
        "paired_t_statistic": t_uncorr,
        "paired_t_pvalue": p_uncorr,
        "ci_95_uncorr": ci_uncorr,
        "nadeau_bengio_t": t_corr,
        "nadeau_bengio_p": p_corr,
        "ci_95_corr": ci_corr,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
    }

def compare_question_predictions(
    y_true: List[str],
    candidate_preds: List[str],
    baseline_preds: List[str]
) -> Dict[str, Any]:
    """Perform pooled out-of-fold McNemar test and exact binomial test."""
    yt = np.array(y_true)
    cand_corr = (np.array(candidate_preds) == yt)
    base_corr = (np.array(baseline_preds) == yt)
    
    n11 = int(np.sum(cand_corr & base_corr))
    n00 = int(np.sum((~cand_corr) & (~base_corr)))
    n10 = int(np.sum(cand_corr & (~base_corr))) # Candidate wins
    n01 = int(np.sum((~cand_corr) & base_corr)) # Baseline wins
    
    discordant = n10 + n01
    if discordant > 0:
        chi2_stat = float(((abs(n10 - n01) - 1.0) ** 2) / discordant)
        mcnemar_p = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))
        binom_p = float(stats.binomtest(n10, discordant, p=0.5, alternative="two-sided").pvalue)
    else:
        chi2_stat, mcnemar_p, binom_p = 0.0, 1.0, 1.0
        
    return {
        "both_correct_n11": n11,
        "both_wrong_n00": n00,
        "candidate_wins_n10": n10,
        "baseline_wins_n01": n01,
        "discordant_pairs": discordant,
        "mcnemar_chi2": chi2_stat,
        "mcnemar_pvalue": mcnemar_p,
        "exact_binomial_pvalue": binom_p,
    }

def run_question_bootstrap(
    y_true: List[str],
    candidate_preds: List[str],
    baseline_preds: List[str],
    n_bootstrap: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """Non-parametric question-level bootstrap for Macro-F1 delta with 95% percentile CI."""
    np.random.seed(random_state)
    yt = np.array(y_true)
    cand_p = np.array(candidate_preds)
    base_p = np.array(baseline_preds)
    n = len(yt)
    
    cand_f1s = []
    base_f1s = []
    delta_f1s = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_samp = yt[idx]
        if len(set(y_samp)) < 3:
            continue
        c_score = f1_score(y_samp, cand_p[idx], average="macro", zero_division=0)
        b_score = f1_score(y_samp, base_p[idx], average="macro", zero_division=0)
        cand_f1s.append(c_score)
        base_f1s.append(b_score)
        delta_f1s.append(c_score - b_score)
        
    delta_arr = np.array(delta_f1s)
    ci_low = float(np.percentile(delta_arr, 2.5))
    ci_high = float(np.percentile(delta_arr, 97.5))
    mean_delta = float(np.mean(delta_arr))
    
    # Empirical p-value
    if mean_delta > 0:
        emp_p = float(np.mean(delta_arr <= 0) * 2.0)
    else:
        emp_p = float(np.mean(delta_arr >= 0) * 2.0)
    emp_p = min(1.0, emp_p)
    
    return {
        "bootstrap_mean_delta": mean_delta,
        "ci_95_percentile": [ci_low, ci_high],
        "empirical_pvalue": emp_p,
        "candidate_bootstrap_mean": float(np.mean(cand_f1s)),
        "candidate_bootstrap_ci": [float(np.percentile(cand_f1s, 2.5)), float(np.percentile(cand_f1s, 97.5))],
        "baseline_bootstrap_mean": float(np.mean(base_f1s)),
        "baseline_bootstrap_ci": [float(np.percentile(base_f1s, 2.5)), float(np.percentile(base_f1s, 97.5))],
    }
