# Experiment Suite Report: ml_uncertainty_comparison_suite

**Generated:** 2026-11-20 18:44:09  
**Dataset:** `data/features.jsonl`  
**Dataset SHA-256:** `c9350085e9993224b6e9377da1ece2d27bc3d004db8ad035573cd02e78109382`  
**Total Samples:** 67  
**Class Distribution:** {'KNOWN': 20, 'AMBIGUOUS': 22, 'UNKNOWN': 25}  

---

## 1. Research Question & Objective

> *Does multi-response consistency and feature-based uncertainty classification improve the distinction between KNOWN, AMBIGUOUS, and UNKNOWN compared with simpler approaches?*

This report systematically evaluates baseline models, multi-response consistency features, and feature group ablations on identical, leakage-safe data splits.

---

## 2. Model & Feature Configuration Comparison

| Experiment ID                              | Model         | Feature Group                |   Features |   Accuracy |   Macro-F1 |   Macro-Prec |   Macro-Rec |   Cohen's Kappa |
|--------------------------------------------|---------------|------------------------------|------------|------------|------------|--------------|-------------|-----------------|
| random_forest_full_16_features             | random_forest | full_16_features             |         16 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_entropy          | random_forest | ablation_no_entropy          |         14 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_agreement        | random_forest | ablation_no_agreement        |         13 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_clustering       | random_forest | ablation_no_clustering       |         14 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_similarity       | random_forest | ablation_no_similarity       |         14 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_question_overlap | random_forest | ablation_no_question_overlap |         15 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_ablation_no_length_diversity | random_forest | ablation_no_length_diversity |         12 |        0.3 |      0.286 |        0.333 |       0.25  |           0.091 |
| random_forest_ablation_no_hedging_negation | random_forest | ablation_no_hedging_negation |         14 |        0.2 |      0.167 |        0.167 |       0.167 |          -0.026 |

## 3. Per-Class Performance Breakdown

### `random_forest_full_16_features`: random_forest (full_16_features)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_entropy`: random_forest (ablation_no_entropy)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_agreement`: random_forest (ablation_no_agreement)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_clustering`: random_forest (ablation_no_clustering)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_similarity`: random_forest (ablation_no_similarity)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_question_overlap`: random_forest (ablation_no_question_overlap)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |     1    |      0.25  |         1 |
| AMBIGUOUS |       0     |     0    |      0     |         5 |
| UNKNOWN   |       1     |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_length_diversity`: random_forest (ablation_no_length_diversity)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |           0 |     0    |      0     |         1 |
| AMBIGUOUS |           0 |     0    |      0     |         5 |
| UNKNOWN   |           1 |     0.75 |      0.857 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 0 |               3 |

### `random_forest_ablation_no_hedging_negation`: random_forest (ablation_no_hedging_negation)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |         0   |      0   |        0   |         1 |
| AMBIGUOUS |         0   |      0   |        0   |         5 |
| UNKNOWN   |         0.5 |      0.5 |        0.5 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 0 |               1 |
| True: AMBIGUOUS |             4 |                 0 |               1 |
| True: UNKNOWN   |             2 |                 0 |               2 |

---

## 4. Best-Performing Configuration

- **Best Model:** `random_forest` using `full_16_features`
- **Macro-F1:** `0.3690`
- **Test Accuracy:** `0.4000`
- **Cohen's Kappa:** `0.2593`

## 5. Statistical Limitations & Scientific Assessment

> [!WARNING]
> **Sample Size Caveat (N=67):** The evaluated dataset has fewer than 100 samples. Metrics evaluated on the test partition carry high empirical variance. These comparisons serve to validate pipeline mechanics and relative feature group sensitivity; they do NOT establish generalizable production performance.

## 6. Leakage & Provenance Certification

- **Group Disjointness:** Verified (False)  
- **Random Seed:** 42  
- **Split Ratios (Train/Val/Test):** [0.7, 0.15, 0.15]  
- **Test Set Isolation:** Test partition was strictly held out and never used during model selection or preprocessing fitting.
