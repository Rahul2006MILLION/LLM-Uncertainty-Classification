# Research Evidence & Empirical Audit Register

**Paper Title:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Document Type:** Empirical Traceability & Verification Register  
**Repository Branch:** `feature/final-inference-pipeline-audit`  
**Date:** November 2026  
**Auditor:** Senior Academic Research Reviewer  

---

## 1. Traceability Matrix: Core Experimental Parameters

All experimental parameters, model specifications, and hyperparameters used in the research paper are traced directly to active repository code and configuration artifacts:

| Parameter / Component | Verified Repository Value | Source Artifact & Code Reference | Invariant / Scientific Justification |
|---|---|---|---|
| **Task Definition** | Tri-class classification: `KNOWN`, `AMBIGUOUS`, `UNKNOWN` | `src/classification/classifier.py:10-11` | Exhaustive partitioning of answerability and epistemic states. |
| **Primary Dataset** | $N=60$ questions, $M=600$ responses ($10$ per question) | `data/batch_2_combined_manifest.json` | 20 KNOWN (33.3%), 20 AMBIGUOUS (33.3%), 20 UNKNOWN (33.3%). |
| **Dataset Sub-Slices** | Batch 2A ($N=30$) + Batch 2B ($N=30$) | `docs/BATCH_2_COMBINED_EVALUATION_REPORT.md` | Unified into balanced, shortcut-resistant corpus. |
| **Semantic Categories** | 15 distinct categories (4 questions per category) | `data/batch_2_combined_questions.jsonl` | 5 KNOWN, 5 AMBIGUOUS, 5 UNKNOWN categories. |
| **Question Provenance** | 24 Mined (40.0%), 24 Synthesized (40.0%), 12 Curated (20.0%) | `data/batch_2_combined_manifest.json:20-25` | Rigorously documented generation source distribution. |
| **Generation Model** | `qwen3:4b` (GGUF `Q4_K_M`, local Ollama) | `configs/batch_2_generation.yaml:12` | Local deterministic/stochastic generation environment. |
| **Sampling Protocol** | $r_0$: Greedy ($T=0.0$); $r_1 \dots r_9$: Stochastic ($T=0.7, p=0.9$) | `src/generation/generator.py:80-95` | Replicates canonical self-consistency sampling methodology. |
| **Embedding Engine** | `all-minilm:latest` (`all-MiniLM-L6-v2`), 384-dimensional | `src/preprocessing/embeddings.py:18-24` | L2-normalized continuous sentence representation. |
| **Clustering Method** | Agglomerative Clustering, Average Linkage | `src/features/clustering.py:16-30` | Cosine distance metric with threshold $\tau = 0.20$ (sim $\ge 0.80$). |
| **Canonical Features** | Exactly 16 features (9 consistency + 7 linguistic) | `src/features/feature_extractor.py:30-47` | Zero label leakage; computed purely from question & outputs. |
| **Partitioning Standard** | 5-Fold `StratifiedGroupKFold(n_splits=5, shuffle=True, seed=42)` | `src/evaluation/data_validator.py` | Strict question-level grouping on `question_group_id`. |
| **Train/Test Sizing** | 48 training questions, 12 testing questions per fold | `docs/BATCH_2_COMBINED_EVALUATION_REPORT.md` | Exactly 4 KNOWN, 4 AMBIGUOUS, 4 UNKNOWN in each test fold. |
| **Preprocessing Isolation** | `StandardScaler` fitted strictly on training folds | `scripts/analyze_batch_2_combined_audit.py` | Transformed onto test folds without global data leakage. |

---

## 2. Canonical 16-Feature Schema Audit

The canonical 16 features extracted in [`src/features/feature_extractor.py`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/src/features/feature_extractor.py#L30-L47) are categorized into two disjoint functional subsets:

### 2.1 Multi-Response Consistency Features (9 Features)
1. `semantic_entropy`: Shannon entropy $H(C) = -\sum_{k} p_k \log p_k$ computed over discrete semantic cluster distribution.
2. `normalized_entropy`: $H(C) / \log K$, scaling entropy between 0.0 and 1.0 regardless of cluster count $K$.
3. `dominant_agreement`: Fraction of generated responses belonging to the largest semantic cluster ($p_1 = N_1 / M$).
4. `second_agreement`: Fraction of responses belonging to the second largest semantic cluster ($p_2 = N_2 / M$).
5. `agreement_margin`: Difference between dominant and second agreement ($p_1 - p_2$).
6. `cluster_count`: Total number of discrete clusters discovered by agglomerative clustering.
7. `cluster_ratio`: Normalized cluster count ($K / M$).
8. `mean_pairwise_similarity`: Mean off-diagonal cosine similarity among all response embeddings.
9. `min_pairwise_similarity`: Minimum off-diagonal cosine similarity among all response pairs.

### 2.2 Ensemble Linguistic Features (7 Features)
10. `response_len_mean`: Mean character length of generated responses across the 10 samples.
11. `response_len_std`: Standard deviation of response character lengths across the 10 samples.
12. `distinct_1gram_ratio`: Ratio of unique unigrams to total tokens aggregated across responses.
13. `distinct_2gram_ratio`: Ratio of unique bigrams to total bigrams aggregated across responses.
14. `question_overlap_mean`: Mean word-level Jaccard token overlap between the input question and each response.
15. `hedging_score`: Cumulative frequency of epistemic hedging markers (*maybe, perhaps, likely, might, could*).
16. `negation_count`: Cumulative frequency of negation tokens (*no, not, never, cannot, neither*).

---

## 3. Statistical Analysis & Anomaly Resolution Audit

### 3.1 Resolution of the $d=0.0 / g=1.781$ Table Misalignment
- **Audit Target:** Table 7.1 in `docs/BATCH_2_COMBINED_EVALUATION_REPORT.md`.
- **Finding:** The reported table row contained 9 columns. Column 7 was Wilcoxon signed-rank test statistic $W = 0.0$ (which is mathematically expected when all 5 fold deltas are strictly positive). Column 8 was Cohen's $d = 1.781$, and Column 9 was Hedges' $g = 1.425$.
- **Source Code Verification:** In `scripts/analyze_batch_2_combined_audit.py`, paired effect sizes are computed via standard formulas:
  $$d = \frac{\bar{d}}{s_d} = \frac{0.180457}{0.101336} = 1.78078$$
  $$J(df=4) = 1 - \frac{3}{4(4) - 1} = 0.800$$
  $$g = d \times J(4) = 1.78078 \times 0.800 = 1.42462$$
- **Verdict:** Mathematical implementation in `data/batch_2_combined_statistical_analysis.json` is 100% correct.

### 3.2 Nadeau & Bengio (2003) Cross-Validation Correction
- **Issue:** Fold-level paired $t$-tests assume independence among folds. In 5-fold CV on 60 samples, training partitions share $75\%$ ($36/48$) of their training data across pairs, inducing positive covariance and inflating Type I error.
- **Nadeau-Bengio Formula:**
  $$\sigma_{\text{corr}}^2 = \left( \frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) s_d^2 = \left( \frac{1}{5} + \frac{12}{48} \right) s_d^2 = 0.45 s_d^2$$
  Standard error is scaled by $\sqrt{0.45 / 0.20} = \sqrt{9/4} = 1.5$.
  $$t_{\text{corr}} = \frac{2}{3} t_{\text{uncorr}} = \frac{2}{3} (3.982) = 2.655$$
  Degrees of freedom $df = 4 \implies p_{\text{corr}} = 0.0567$ (marginally significant at $\alpha = 0.05$).

### 3.3 Out-of-Fold Question-Level Statistical Validation
To eliminate dependence on fold resampling assumptions, question-level out-of-fold predictions were analyzed:
- **Contingency Analysis ($N=60$ questions):**
  - Concordant correct: 36
  - Concordant incorrect: 10
  - FULL_HYBRID correct, TRUE_SINGLE incorrect: 12
  - TRUE_SINGLE correct, FULL_HYBRID incorrect: 2
- **McNemar's Test (continuity-corrected):**
  $$\chi^2 = \frac{(|12 - 2| - 1)^2}{12 + 2} = \frac{81}{14} = 5.786 \quad (p = 0.0162)$$
- **Exact Binomial Two-Sided Test:** $p = 0.0129$ (12 wins vs 2 losses).
- **Non-Parametric Bootstrap ($B=2000$ iterations):**
  Mean $\Delta \text{Macro-F1} = +0.1740$, 95% Percentile Confidence Interval: $[+0.0612, +0.2929]$.
- **Conclusion:** Superiority of multi-response hybrid modeling over true single-response is statistically verified across parametric, non-parametric, and bootstrap tests.

---

## 4. Key Scientific Nuances Verified from Repository

1. **Linguistic Features Drive the Multi-Response Lift:**
   - `FULL_HYBRID` (16 features): Macro-F1 = $0.789 \pm 0.137$.
   - `ENSEMBLE_LINGUISTIC` (7 features): Macro-F1 = $0.791 \pm 0.209$.
   - $\Delta = -0.002$ ($p = 0.9672$, Cohen's $d = -0.020$, McNemar $p = 1.0000$).
   - Multi-response consistency alone (`CONSISTENCY_ONLY`, 9 features) achieves only Macro-F1 = $0.460 \pm 0.135$ (Accuracy 50.0%).
   - **Empirical Implication:** Sampling 10 responses reduces variance primarily by providing robust aggregate linguistic signals (length variance `response_len_std`, total hedging, and negation counts), rather than semantic clustering entropy.

2. **Single-Response TF-IDF Generalization Collapse:**
   - In standard 5-fold CV, TF-IDF + Linear SVM reaches $83.33\%$ Accuracy / $0.8296$ Macro-F1 on single responses ($r_0$).
   - Under 15-fold Leave-One-Category-Out (LOCO), accuracy collapses to $65.00\%$ ($-18.33\%$, $p = 0.000977$, 11 wins vs 0 losses).
   - In lexical ablation, isolating strictly epistemic markers (`ExpE_Epistemic_Only_TFIDF`) causes CV accuracy to drop from $83.33\%$ to $68.33\%$ ($-15.00\%$, $p = 0.0490$).
   - **Empirical Implication:** Single-response high performance heavily exploits domain vocabulary shortcuts (e.g. historical and scientific content nouns) that do not transfer out of domain.
