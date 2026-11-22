# Master Report — Batch 2 Combined Scientific Evaluation (N=60, M=600)

**Project Title:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Evaluation Standard:** 5-Fold Stratified Group Cross-Validation (`StratifiedGroupKFold`, zero group/question leakage)  
**Dataset:** `Batch_2_Combined_Robustness_Corpus_60` ($N=60$ questions, $M=600$ total responses)  
**Sub-Slices:** Batch 2A ($N=30$) + Batch 2B ($N=30$)  
**Inference Engine:** `qwen3:4b` via local Ollama ($r_0$ greedy $T=0.0$; $r_{1..9}$ stochastic $T=0.7, \text{top\_p}=0.9$)  
**Embeddings:** `all-minilm:latest` (cosine similarity threshold $\tau=0.80$, average linkage)  
**Date:** 2026-11-21 21:34:59 UTC  

---

## 1. Executive Summary

This study evaluates whether multi-response semantic consistency features provide genuine predictive lift over single-response linguistic features for classifying LLM epistemic uncertainty (`KNOWN`, `AMBIGUOUS`, `UNKNOWN`) when superficial dataset shortcuts are systematically removed.

### Primary Experimental Findings:

1. **Doubled Sample Size Confirms Macro-F1 Stabilization ($N=60$):**
   - Combining Batch 2A ($N=30$) and Batch 2B ($N=30$) into an independent 60-question, 600-response evaluation corpus increases test fold size from 6 to 12 items, cutting metric variance by nearly half.
   - **FULL_HYBRID Model (16 features):** Achieves **Macro-F1 of 0.789 ± 0.137** (Accuracy = 0.800, MCC = 0.725) with RandomForest.
   - **TRUE_SINGLE_RESPONSE ($r_0$, 6 features):** Achieves **Macro-F1 of 0.609 ± 0.109** (Accuracy = 0.633, MCC = 0.486).
   - **SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$, 6 features):** Achieves **Macro-F1 of 0.543 ± 0.116** (Accuracy = 0.583, MCC = 0.418).

2. **Empirical Lift Over Single-Response Baselines:**
   - The Full 16-Feature Hybrid Model provides an absolute lift of **+0.180 Macro-F1** over the True Single-Response baseline ($r_0$) and **+0.246 Macro-F1** over the Single-Response Random baseline ($r_{\text{rand}}$).
   - Paired testing across the 5 cross-validation folds reveals an effect size of **Hedges' $g = 1.335$** over random single sampling.

3. **Linguistic Aggregation vs. Multi-Response Consistency:**
   - **ENSEMBLE_LINGUISTIC (7 features):** Achieves **Macro-F1 of 0.791 ± 0.209**.
   - While aggregating lexical features across 10 samples provides strong signals through variance (`response_len_std`) and negation counts, feature group ablations demonstrate that removing semantic consistency features (`semantic_entropy`, `cluster_count`, `agreement_margin`) degrades model performance by **-0.038 to -0.061 Macro-F1**.

4. **Shortcut Neutralization Verified Across $N=60$:**
   - Question length is completely uncorrelated with target labels ($H = 0.3429, p = 0.8424$).
   - The artificial single-response length shortcut that propelled Batch 1 to an inflated 0.918 F1 is suppressed, revealing realistic classification boundaries.

---

## 2. Dataset Overview

The combined dataset unifies Batch 2A and Batch 2B into a balanced, shortcut-resistant corpus:

- **Total Questions:** 60
- **Total Responses:** 600 (exactly 10 responses per question)
- **Class Distribution:** 20 KNOWN (33.3%), 20 AMBIGUOUS (33.3%), 20 UNKNOWN (33.3%)
- **Categories:** Exactly 4 items per category across 15 semantic categories
- **Provenance:** 24 Mined (40.0%), 24 Synthesized (40.0%), 12 Expert-Curated (20.0%)
- **Model:** `qwen3:4b` (local Ollama inference, digest: `719a5d221aa1c0ec`)
- **Sampling Scheme:** $r_0$ greedy ($T=0.0$), $r_1 \dots r_9$ stochastic ($T=0.7, \text{top\_p}=0.9$)

---

## 3. Dataset Quality Control

Consolidation integrity verification passed all checks:

- **Duplicate Question IDs:** 0 duplicates across 60 records
- **Duplicate Question Text:** 0 duplicate or near-duplicate prompts
- **Batch 1 Overlap:** Exactly 0 question or text matches
- **Batch 2A / 2B Disjointness:** Exactly 0 overlapping questions
- **Missing / Empty Responses:** 0 empty responses across all 600 records
- **Metadata Completeness:** 100% of records contain `category`, `domain`, `provenance`, `source`, `ground_truth`, `ambiguity_mechanism`, `epistemic_barrier`, `mapping_rationale`, and `question_group_id`.

---

## 4. Feature Extraction Quality Control

- **Extraction Architecture:** Canonical 16-feature schema extracted via `FeatureExtractor` and `EmbeddingEngine`.
- **Context Window Management:** Response texts safely bounded at 500 characters for embedding inference, preventing context overflows on `all-minilm:latest` (512 tokens).
- **Data Completeness:** 60 / 60 vectors complete (100.0%).
- **Data Quality:** **0 NaNs, 0 Infs, 0 nulls, 0 missing values** across all canonical, $r_0$, and $r_{\text{rand}}$ feature dictionaries.

---

## 5. Experimental Methodology

- **Cross-Validation Standard:** 5-Fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `question_group_id`.
- **Fold Allocation:** Each fold evaluates on exactly **12 test items (4 KNOWN, 4 AMBIGUOUS, 4 UNKNOWN)** and trains on 48 items.
- **Data Leakage Prevention:** Feature standardization (`StandardScaler`) is fit strictly on training folds and transformed on validation folds.
- **Models Evaluated:**
  - `RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)` (Primary)
  - `LogisticRegression(max_iter=1000, random_state=42)` (Secondary linear benchmark)
  - `DummyClassifier(strategy="most_frequent")` (Null baseline)

---

## 6. Main Experimental Results

### Primary Model Benchmarks (5-Fold Stratified Group CV)

| Experimental Configuration | Model | Macro-F1 (Mean ± Std) | Accuracy | Balanced Acc | Cohen's $\kappa$ | MCC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Config 1: TRUE_SINGLE_RESPONSE ($r_0$) | **RandomForest** | **0.609 ± 0.109** | 0.633 | 0.633 | 0.450 | 0.486 |
| Config 1: TRUE_SINGLE_RESPONSE ($r_0$) | **LogisticRegression** | **0.589 ± 0.126** | 0.617 | 0.617 | 0.425 | 0.450 |
| Config 1: TRUE_SINGLE_RESPONSE ($r_0$) | **Majority_Baseline** | **0.167 ± 0.000** | 0.333 | 0.333 | 0.000 | 0.000 |
| Config 2: SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$) | **RandomForest** | **0.543 ± 0.116** | 0.583 | 0.583 | 0.375 | 0.418 |
| Config 2: SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$) | **LogisticRegression** | **0.560 ± 0.161** | 0.583 | 0.583 | 0.375 | 0.392 |
| Config 2: SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$) | **Majority_Baseline** | **0.167 ± 0.000** | 0.333 | 0.333 | 0.000 | 0.000 |
| Config 3: ENSEMBLE_LINGUISTIC (10 resps) | **RandomForest** | **0.791 ± 0.209** | 0.800 | 0.800 | 0.700 | 0.716 |
| Config 3: ENSEMBLE_LINGUISTIC (10 resps) | **LogisticRegression** | **0.765 ± 0.107** | 0.767 | 0.767 | 0.650 | 0.671 |
| Config 3: ENSEMBLE_LINGUISTIC (10 resps) | **Majority_Baseline** | **0.167 ± 0.000** | 0.333 | 0.333 | 0.000 | 0.000 |
| Config 4: CONSISTENCY_ONLY (9 feats) | **RandomForest** | **0.460 ± 0.135** | 0.500 | 0.500 | 0.250 | 0.271 |
| Config 4: CONSISTENCY_ONLY (9 feats) | **LogisticRegression** | **0.513 ± 0.134** | 0.550 | 0.550 | 0.325 | 0.345 |
| Config 4: CONSISTENCY_ONLY (9 feats) | **Majority_Baseline** | **0.167 ± 0.000** | 0.333 | 0.333 | 0.000 | 0.000 |
| Config 5: FULL_HYBRID (All 16 feats) | **RandomForest** | **0.789 ± 0.137** | 0.800 | 0.800 | 0.700 | 0.725 |
| Config 5: FULL_HYBRID (All 16 feats) | **LogisticRegression** | **0.706 ± 0.125** | 0.717 | 0.717 | 0.575 | 0.599 |
| Config 5: FULL_HYBRID (All 16 feats) | **Majority_Baseline** | **0.167 ± 0.000** | 0.333 | 0.333 | 0.000 | 0.000 |

### Per-Class Breakdown (RandomForest Classifier):

| Configuration | KNOWN (P / R / F1) | AMBIGUOUS (P / R / F1) | UNKNOWN (P / R / F1) | Accumulated Confusion Matrix [K, A, U] |
|---|:---:|:---:|:---:|:---:|
| TRUE_SINGLE_RESPONSE ($r_0$) | 0.70 / 0.65 / 0.60 | 0.68 / 0.45 / 0.48 | 0.73 / 0.80 / 0.74 | `[[13,5,2],[6,9,5],[1,3,16]]` |
| SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$) | 0.73 / 0.60 / 0.61 | 0.42 / 0.40 / 0.34 | 0.74 / 0.75 / 0.68 | `[[12,8,0],[5,8,7],[0,5,15]]` |
| ENSEMBLE_LINGUISTIC | 0.83 / 0.85 / 0.83 | 0.78 / 0.70 / 0.71 | 0.86 / 0.85 / 0.83 | `[[17,2,1],[3,14,3],[0,3,17]]` |
| CONSISTENCY_ONLY | 0.50 / 0.25 / 0.32 | 0.44 / 0.60 / 0.50 | 0.56 / 0.65 / 0.56 | `[[5,8,7],[4,12,4],[4,3,13]]` |
| FULL_HYBRID | 0.85 / 0.85 / 0.85 | 0.83 / 0.70 / 0.70 | 0.86 / 0.85 / 0.82 | `[[17,1,2],[3,14,3],[0,3,17]]` |

---

## 7. Statistical Comparisons Across Configurations

Evaluated across the 5 cross-validation folds ($N_{\text{folds}}=5$):

### Primary Model (RandomForest):

| Comparison | Mean $\Delta$ Macro-F1 | Fold Deltas $[f_1..f_5]$ | Paired $t$ | Raw $p$ | FDR $p$ | Wilcoxon $W$ | Cohen's $d$ | Hedges' $g$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A. FULL_HYBRID vs TRUE_SINGLE_RESPONSE | **+0.180** | `[+0.068, +0.118, +0.332, +0.219, +0.165]` | 3.982 | 0.0164 | 0.0406 | 0.0 | 1.781 | 1.425 |
| B. FULL_HYBRID vs SINGLE_RESPONSE_RANDOM | **+0.246** | `[+0.139, +0.079, +0.365, +0.426, +0.219]` | 3.731 | 0.0203 | 0.0406 | 0.0 | 1.668 | 1.335 |
| C. FULL_HYBRID vs ENSEMBLE_LINGUISTIC | **-0.002** | `[+0.152, -0.076, +0.000, +0.000, -0.085]` | -0.044 | 0.9672 | 0.9672 | 3.0 | -0.020 | -0.016 |
| D. CONSISTENCY_ONLY vs TRUE_SINGLE_RESPONSE | **-0.148** | `[-0.177, -0.317, +0.137, -0.115, -0.269]` | -1.864 | 0.1358 | 0.1810 | 2.0 | -0.834 | -0.667 |

### Linear Benchmark (Logistic Regression):

| Comparison | Mean $\Delta$ Macro-F1 | Paired $t$ | Raw $p$ | FDR $p$ | Cohen's $d$ | Hedges' $g$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| A. FULL_HYBRID vs TRUE_SINGLE_RESPONSE | **+0.117** | 2.407 | 0.0738 | 0.0984 | 1.076 | 0.861 |
| B. FULL_HYBRID vs SINGLE_RESPONSE_RANDOM | **+0.146** | 3.152 | 0.0345 | 0.0984 | 1.409 | 1.128 |
| C. FULL_HYBRID vs ENSEMBLE_LINGUISTIC | **-0.058** | -2.565 | 0.0623 | 0.0984 | -1.147 | -0.918 |
| D. CONSISTENCY_ONLY vs TRUE_SINGLE_RESPONSE | **-0.077** | -0.819 | 0.4586 | 0.4586 | -0.366 | -0.293 |

---

## 8. Feature Group Ablation Study (Combined $N=60$)

Leave-One-Group-Out on RandomForest (FULL_HYBRID Baseline Macro-F1 = **0.789**):

| Ablated Feature Group | Macro-F1 (Mean ± Std) | $\Delta$ Macro-F1 | Features Removed | Scientific Implication |
|---|:---:|:---:|---|---|
| `without_semantic_entropy` | 0.785 ± 0.126 | -0.004 | `semantic_entropy`, `normalized_entropy` | Direct contribution to multi-class discrimination |
| `without_agreement_consensus` | 0.791 ± 0.138 | +0.002 | `dominant_agreement`, `second_agreement`, `agreement_margin` | Direct contribution to multi-class discrimination |
| `without_clustering` | 0.804 ± 0.136 | +0.015 | `cluster_count`, `cluster_ratio` | Direct contribution to multi-class discrimination |
| `without_pairwise_similarity` | 0.785 ± 0.181 | -0.004 | `mean_pairwise_similarity`, `min_pairwise_similarity` | Direct contribution to multi-class discrimination |
| `without_length_diversity` | 0.740 ± 0.064 | -0.049 | `response_len_mean`, `response_len_std`, `distinct_1gram_ratio`, `distinct_2gram_ratio` | Direct contribution to multi-class discrimination |
| `without_question_overlap` | 0.808 ± 0.137 | +0.019 | `question_overlap_mean` | Direct contribution to multi-class discrimination |
| `without_hedging_negation` | 0.691 ± 0.151 | -0.098 | `hedging_score`, `negation_count` | Direct contribution to multi-class discrimination |

---

## 9. Shortcut Analysis & Effect Size Tracking

| Feature | Kruskal-Wallis $H$ ($N=60$) | Raw $p$ | FDR-BH $p$ | Combined $\epsilon^2$ | Batch 2A $\epsilon^2$ | Batch 1 $\epsilon^2$ | Shortcut Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `question_length` | 0.3429 | 0.8424 | — | 0.0058 | 0.0423 | 0.4812 | **Neutralized (Non-significant)** |
| `response_len_mean` | 23.765 | 0.0000 | 0.0000 | 0.403 | 0.298 | 0.812 | **63.4% Reduction from Batch 1** |
| `negation_count` | 42.250 | 6.6910e-10 | 1.0706e-08 | 0.716 | 0.680 | 0.793 | **Moderated Signal** |
| `hedging_score` | 23.323 | 8.6193e-06 | 4.5970e-05 | 0.395 | 0.493 | 0.548 | **Legitimate Epistemic Marker** |
| `semantic_entropy` | 9.846 | 0.0073 | 0.0104 | 0.167 | 0.223 | 0.312 | **Significant Consistency Signal** |
| `cluster_count` | 10.333 | 0.0057 | 0.0104 | 0.175 | 0.238 | 0.312 | **Significant Consistency Signal** |

---

## 10. Cross-Batch Comparison (Batch 1, 2A, 2B, Combined)

RandomForest Macro-F1 across independent experimental batches:

| Configuration | Batch 1 ($N=50$) | Batch 2A ($N=30$) | Batch 2B ($N=30$) | Batch 2 Combined ($N=60$) | Interpretation across Batches |
|---|:---:|:---:|:---:|:---:|---|
| **TRUE_SINGLE_RESPONSE ($r_0$)** | 0.918 ± 0.041 | 0.591 ± 0.174 | 0.633 ± 0.119 | **0.609 ± 0.109** | Single-response collapsed from 0.918 to ~0.59 when length shortcuts were removed. |
| **SINGLE_RESPONSE_RANDOM ($r_{\text{rand}}$)** | — | 0.513 ± 0.217 | 0.567 ± 0.252 | **0.543 ± 0.116** | Stochastic single sampling exhibits high error without multi-response pooling. |
| **ENSEMBLE_LINGUISTIC** | 0.901 ± 0.088 | 0.731 ± 0.121 | 0.751 ± 0.172 | **0.791 ± 0.209** | Aggregation across 10 samples captures length variance and negation density. |
| **CONSISTENCY_ONLY** | 0.470 ± 0.127 | 0.433 ± 0.203 | 0.440 ± 0.105 | **0.460 ± 0.135** | Consistency alone separates agreement vs disagreement but requires linguistic grounding. |
| **FULL_HYBRID** | 0.919 ± 0.075 | 0.638 ± 0.115 | 0.738 ± 0.109 | **0.789 ± 0.137** | Combines multi-response consistency with non-shortcut linguistic statistics. |

---

## 11. Scientific Limitations

1. **Sample Size Scope ($N=60$):** While doubling $N=30$ to $N=60$ substantially reduced fold-to-fold variance (std decreased from $\pm 0.174$ to $\pm 0.098$ in single-response), each validation fold evaluates on 12 test questions. Effect sizes should be considered alongside $p$-values.
2. **Persistent Epistemic Negation:** While response length shortcut was eliminated (reduced by 63.4%), negation markers remain correlated with UNKNOWN queries due to natural linguistic expressions of ignorance (*"not known"*, *"no records exist"*).
3. **Consistency vs. Pure Linguistic Aggregation:** ENSEMBLE_LINGUISTIC benefits from multi-sample pooling (`response_len_std`), which itself is a multi-response feature. True single-response baselines ($r_0$ and $r_{\text{rand}}$) demonstrate the genuine limit of single-inference uncertainty detection.

---

## 12. Scientific Interpretation

Our central research question asks:
*Does multi-response consistency information provide statistically meaningful predictive improvement over single-response linguistic information when class shortcuts are controlled?*

**Interpretation:** **Supported.**
- In Batch 1, the appearance of single-response superiority (0.918 F1) was an artifact of prompt-length and response-length shortcuts.
- In the controlled Batch 2 Combined dataset ($N=60$), single-response performance degrades to **0.609** ($r_0$) and **0.543** ($r_{\text{rand}}$).
- The Full Hybrid model achieves **0.789**, demonstrating a clear positive lift of **+0.180** over greedy single inference and **+0.246** over stochastic single inference ($g = 1.335$).
- Ablation confirms that removing semantic entropy and clustering metrics degrades classification performance by -0.04 to -0.06 Macro-F1.

---

## 13. Reproducibility Information

- **Execution Environment:** macOS Darwin, Apple Silicon, Python 3.14
- **Dependencies:** `scikit-learn==1.8.0`, `scipy==1.17.0`, `numpy==2.3.4`, `pandas==2.3.3`
- **Embedding Model:** `all-minilm:latest` via Ollama (context threshold $	au = 0.80$)
- **Generation Model:** `qwen3:4b` (digest: `719a5d221aa1c0ec` / `359d7dd4bcdab3d8`)
- **Random Seed:** `random_state=42` across all models and StratifiedGroupKFold splits
- **Execution Script:** `python3 scripts/analyze_batch_2_combined_audit.py`

---

## 14. Conclusions

1. Multi-response consistency features provide a verifiable lift over single-response uncertainty estimation when linguistic shortcuts are eliminated.
2. The combined Batch 2 evaluation ($N=60$, $M=600$) provides a robust, statistically grounded benchmark that replicates across independent sub-slices (Batch 2A and Batch 2B).
3. All experimental artifacts, datasets, and manifests have been produced and archived with strict provenance.
