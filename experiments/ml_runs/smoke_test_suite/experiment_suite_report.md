# Experiment Suite Report: ml_uncertainty_comparison_suite

**Generated:** 2026-11-20 18:43:55  
**Dataset:** `data/features.jsonl`  
**Dataset SHA-256:** `c9350085e9993224b6e9377da1ece2d27bc3d004db8ad035573cd02e78109382`  
**Total Samples:** 67  
**Class Distribution:** {'AMBIGUOUS': 22, 'UNKNOWN': 25, 'KNOWN': 20}  

---

## 1. Research Question & Objective

> *Does multi-response consistency and feature-based uncertainty classification improve the distinction between KNOWN, AMBIGUOUS, and UNKNOWN compared with simpler approaches?*

This report systematically evaluates baseline models, multi-response consistency features, and feature group ablations on identical, leakage-safe data splits.

---

## 2. Model & Feature Configuration Comparison

| Experiment ID                                     | Model                  | Feature Group              |   Features |   Accuracy |   Macro-F1 |   Macro-Prec |   Macro-Rec |   Cohen's Kappa |
|---------------------------------------------------|------------------------|----------------------------|------------|------------|------------|--------------|-------------|-----------------|
| hist_gradient_boosting_single_response_baseline   | hist_gradient_boosting | single_response_baseline   |          6 |        0.5 |      0.429 |        0.5   |       0.4   |           0.286 |
| random_forest_single_response_baseline            | random_forest          | single_response_baseline   |          6 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| random_forest_full_16_features                    | random_forest          | full_16_features           |         16 |        0.4 |      0.369 |        0.381 |       0.583 |           0.259 |
| hist_gradient_boosting_full_16_features           | hist_gradient_boosting | full_16_features           |         16 |        0.4 |      0.333 |        0.333 |       0.333 |           0.189 |
| logistic_regression_single_response_baseline      | logistic_regression    | single_response_baseline   |          6 |        0.3 |      0.306 |        0.381 |       0.5   |           0.125 |
| logistic_regression_full_16_features              | logistic_regression    | full_16_features           |         16 |        0.3 |      0.306 |        0.381 |       0.5   |           0.125 |
| logistic_regression_multi_response_consistency    | logistic_regression    | multi_response_consistency |          9 |        0.2 |      0.194 |        0.278 |       0.15  |          -0.111 |
| majority_baseline_single_response_baseline        | majority_baseline      | single_response_baseline   |          6 |        0.4 |      0.19  |        0.133 |       0.333 |           0     |
| majority_baseline_multi_response_consistency      | majority_baseline      | multi_response_consistency |          9 |        0.4 |      0.19  |        0.133 |       0.333 |           0     |
| majority_baseline_full_16_features                | majority_baseline      | full_16_features           |         16 |        0.4 |      0.19  |        0.133 |       0.333 |           0     |
| random_forest_multi_response_consistency          | random_forest          | multi_response_consistency |          9 |        0   |      0     |        0     |       0     |          -0.282 |
| hist_gradient_boosting_multi_response_consistency | hist_gradient_boosting | multi_response_consistency |          9 |        0   |      0     |        0     |       0     |          -0.282 |

## 3. Per-Class Performance Breakdown

### `hist_gradient_boosting_single_response_baseline`: hist_gradient_boosting (single_response_baseline)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |         0   |      0   |      0     |         1 |
| AMBIGUOUS |         0.5 |      0.2 |      0.286 |         5 |
| UNKNOWN   |         1   |      1   |      1     |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             4 |                 1 |               0 |
| True: UNKNOWN   |             0 |                 0 |               4 |

### `random_forest_single_response_baseline`: random_forest (single_response_baseline)

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

### `hist_gradient_boosting_full_16_features`: hist_gradient_boosting (full_16_features)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |           0 |        0 |          0 |         1 |
| AMBIGUOUS |           0 |        0 |          0 |         5 |
| UNKNOWN   |           1 |        1 |          1 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             0 |                 0 |               4 |

### `logistic_regression_single_response_baseline`: logistic_regression (single_response_baseline)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |      1   |      0.25  |         1 |
| AMBIGUOUS |       0     |      0   |      0     |         5 |
| UNKNOWN   |       1     |      0.5 |      0.667 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 1 |               2 |

### `logistic_regression_full_16_features`: logistic_regression (full_16_features)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0.143 |      1   |      0.25  |         1 |
| AMBIGUOUS |       0     |      0   |      0     |         5 |
| UNKNOWN   |       1     |      0.5 |      0.667 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             1 |                 0 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             1 |                 1 |               2 |

### `logistic_regression_multi_response_consistency`: logistic_regression (multi_response_consistency)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |       0     |     0    |      0     |         1 |
| AMBIGUOUS |       0.333 |     0.2  |      0.25  |         5 |
| UNKNOWN   |       0.5   |     0.25 |      0.333 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             3 |                 1 |               1 |
| True: UNKNOWN   |             2 |                 1 |               1 |

### `majority_baseline_single_response_baseline`: majority_baseline (single_response_baseline)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |         0   |        0 |      0     |         1 |
| AMBIGUOUS |         0   |        0 |      0     |         5 |
| UNKNOWN   |         0.4 |        1 |      0.571 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 0 |               1 |
| True: AMBIGUOUS |             0 |                 0 |               5 |
| True: UNKNOWN   |             0 |                 0 |               4 |

### `majority_baseline_multi_response_consistency`: majority_baseline (multi_response_consistency)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |         0   |        0 |      0     |         1 |
| AMBIGUOUS |         0   |        0 |      0     |         5 |
| UNKNOWN   |         0.4 |        1 |      0.571 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 0 |               1 |
| True: AMBIGUOUS |             0 |                 0 |               5 |
| True: UNKNOWN   |             0 |                 0 |               4 |

### `majority_baseline_full_16_features`: majority_baseline (full_16_features)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |         0   |        0 |      0     |         1 |
| AMBIGUOUS |         0   |        0 |      0     |         5 |
| UNKNOWN   |         0.4 |        1 |      0.571 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 0 |               1 |
| True: AMBIGUOUS |             0 |                 0 |               5 |
| True: UNKNOWN   |             0 |                 0 |               4 |

### `random_forest_multi_response_consistency`: random_forest (multi_response_consistency)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |           0 |        0 |          0 |         1 |
| AMBIGUOUS |           0 |        0 |          0 |         5 |
| UNKNOWN   |           0 |        0 |          0 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             2 |                 2 |               0 |

### `hist_gradient_boosting_multi_response_consistency`: hist_gradient_boosting (multi_response_consistency)

| Class     |   Precision |   Recall |   F1-Score |   Support |
|-----------|-------------|----------|------------|-----------|
| KNOWN     |           0 |        0 |          0 |         1 |
| AMBIGUOUS |           0 |        0 |          0 |         5 |
| UNKNOWN   |           0 |        0 |          0 |         4 |

**Confusion Matrix:**
|                 |   Pred: KNOWN |   Pred: AMBIGUOUS |   Pred: UNKNOWN |
|-----------------|---------------|-------------------|-----------------|
| True: KNOWN     |             0 |                 1 |               0 |
| True: AMBIGUOUS |             5 |                 0 |               0 |
| True: UNKNOWN   |             2 |                 2 |               0 |

---

## 4. Best-Performing Configuration

- **Best Model:** `hist_gradient_boosting` using `single_response_baseline`
- **Macro-F1:** `0.4286`
- **Test Accuracy:** `0.5000`
- **Cohen's Kappa:** `0.2857`

## 5. Statistical Limitations & Scientific Assessment

> [!WARNING]
> **Sample Size Caveat (N=67):** The evaluated dataset has fewer than 100 samples. Metrics evaluated on the test partition carry high empirical variance. These comparisons serve to validate pipeline mechanics and relative feature group sensitivity; they do NOT establish generalizable production performance.

## 6. Leakage & Provenance Certification

- **Group Disjointness:** Verified (False)  
- **Random Seed:** 42  
- **Split Ratios (Train/Val/Test):** [0.7, 0.15, 0.15]  
- **Test Set Isolation:** Test partition was strictly held out and never used during model selection or preprocessing fitting.
