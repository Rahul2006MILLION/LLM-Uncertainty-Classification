# Experimental Evaluation Pipeline Infrastructure Guide

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Milestone:** 7.5.5 — Experimental Evaluation Pipeline Infrastructure  
**Authoritative Protocol:** [docs/EXPERIMENTAL_PROTOCOL.md](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/docs/EXPERIMENTAL_PROTOCOL.md)  
**Implementation Modules:**  
- Engine: [src/evaluation/evaluation_pipeline.py](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/src/evaluation/evaluation_pipeline.py)  
- CLI: [scripts/run_experiment.py](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/scripts/run_experiment.py)  
- Configurations: `configs/experiments/`  

---

## 1. Scientific Status Declaration

> ### ⚠️ EMPIRICAL RESEARCH STATUS NOTICE:
> **NO REAL EXPERIMENTS HAVE BEEN CONDUCTED.**  
> - A finalized human gold benchmark does not currently exist.
> - Human annotation has not yet been executed (`Annotator A: 0/300`, `Annotator B: 0/300`).
> - The annotation blinding key remains quarantined.
> - Finalized gold labels have **NOT** been issued or inferred.
> - Real dataset splits and cross-validation assignments have not been generated.
> - No models have been trained or evaluated on candidate data.
> 
> In accordance with strict scientific integrity principles, the evaluation pipeline operates behind an **automated execution gate**: it strictly halts (exit code 2) if executed against candidate/unannotated data.

---

## 2. Experimental Framework & Architecture

The evaluation pipeline operationalizes the hypotheses ($H_1$ through $H_5$) formulated in `docs/EXPERIMENTAL_PROTOCOL.md` across three epistemic uncertainty classes:
- **`KNOWN` (Class 0):** Determinate, consensus-answerable factual questions yielding consistent LLM responses.
- **`AMBIGUOUS` (Class 1):** Polysemic, underspecified, or multi-answer questions resulting in multiple valid response clusters.
- **`UNKNOWN` (Class 2):** Inaccessible, unanswerable, or counterfactual queries eliciting hallucinations, refusals, or dispersed responses.

### Pipeline Execution Flow

```text
Finalized Gold Benchmark (uncertainty_benchmark_300_gold_v1.0.jsonl)
                  │
                  ▼
Strict Provenance & Manifest Validation (GoldDatasetValidator)
                  │
                  ▼
Validated Leakage-Safe Splits & CV (gold_split_v1.json, gold_cv5_v1.json)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
5-Fold StratifiedGroupKFold   Train / Val / Test Partition
(Fold Models, Mean ± Std)    (Train Fitting, Isolated Test Evaluation)
        │                   │
        └─────────┬─────────┘
                  ▼
Comprehensive Metric Calculation (Accuracy, Macro-F1, Balanced Acc, MCC, Kappa)
                  │
                  ▼
Standardized Results Schema & Publication Audit Report
```

---

## 3. Models & Baselines Supported

| Model Identifier | Architecture | Feature Vector | Purpose & Hypothesis Tested |
| :--- | :--- | :--- | :--- |
| **`majority_baseline`** | Prior Class Classifier | None (0 features) | Establishes the empirical performance floor; predicts majority class of training set. |
| **`simple_feature_baseline`** | Rule-Based Threshold | `normalized_entropy`, `dominant_agreement` (2) | Evaluates whether static heuristic cutoffs suffice without machine learning. |
| **`logistic_regression`** | Multinomial Logistic Regression | Canonical 16 (StandardScaler fitted on train) | Linear baseline; tests whether epistemic classes are linearly separable in feature space. |
| **`random_forest`** | Random Forest (100 trees, depth 6) | Canonical 16 | Non-linear ensemble baseline; provides Gini feature importance attribution. |
| **`xgboost`** | Gradient Boosted Decision Trees | Canonical 16 (`max_depth=4`, `lr=0.05`) | Proposed framework classifier under primary research investigation ($H_1$). |

### Feature Ablations (Ablations 1 – 6)
The pipeline supports feature subset evaluation via the `feature_set` configuration:
- `ablation_1_no_entropy`: Removes `semantic_entropy` and `normalized_entropy` (14 features).
- `ablation_2_no_agreement`: Removes dominant agreement, second agreement, margin, cluster count, ratio (11 features).
- `ablation_3_no_similarity`: Removes pairwise embedding distance metrics (14 features).
- `ablation_4_no_lexical_length`: Removes response length and distinct n-gram ratios (12 features).
- `ablation_5_no_hedging_negation`: Removes lexical hedging and negations (14 features).
- `ablation_6_no_overlap`: Removes question-response token overlap (15 features).

---

## 4. Strict Scientific Safeguards & Execution Gates

To prevent accidental data snooping, feature leakage, or invalid candidate-label evaluations:

1. **Gold Benchmark Provenance Gate:**
   - The dataset must declare `reviewer_agreement.annotation_status == "gold_benchmark"`.
   - Must contain verified `human_annotation_provenance`.
   - Datasets with `"needs_human_review"` or filenames matching `*candidate*` are unconditionally rejected.
2. **Preprocessing Leakage Prevention:**
   - Any feature scaling (e.g., `StandardScaler` for Logistic Regression) is fitted **strictly on the training partition** (`fit_transform(X_train)`), followed by `transform(X_val)` and `transform(X_test)`.
3. **Test-Set Isolation:**
   - The test partition ($\approx 15\%$, 45 questions) is never exposed during hyperparameter selection, threshold tuning, or cross-validation.
4. **Group Isolation:**
   - All questions belonging to the same `question_group_id` reside in the same partition.

---

## 5. Standardized Results Schema

Experiment execution outputs a machine-readable JSON record containing:
- `experiment_id`: Unique identifier.
- `timestamp`: UTC ISO timestamp.
- `dataset_version`: Version string of the gold benchmark.
- `gold_benchmark_sha256`: Cryptographic checksum of the input gold data.
- `model_name`: Model architecture.
- `feature_set`: Feature subset used.
- `random_seed`: Seed used for reproducibility.
- `sample_counts`: Exact train, val, and test sample counts.
- `validation_metrics`: Full classification metric dictionary on validation data.
- `test_metrics`: Full classification metric dictionary on held-out test data.
- `cross_validation`: 5-fold aggregate means, standard deviations, and class-wise scores.
- `feature_importances`: Gini or tree-gain importances (when supported by model).
- `execution_status`: `"COMPLETED"`.

---

## 6. CLI Usage

### Executing an Experiment (Post-Gold Benchmark Finalization)
```bash
python scripts/run_experiment.py \
    --config configs/experiments/proposed_xgboost.yaml \
    --output-dir experiments/results \
    --seed 42
```

### Dry-Run Verification (Audits configuration and data gates without training)
```bash
python scripts/run_experiment.py \
    --config configs/experiments/proposed_xgboost.yaml \
    --dry-run
```

### Safe Failure on Current Candidate Benchmark
When executed against the unannotated candidate benchmark:
```bash
python scripts/run_experiment.py --config configs/experiments/proposed_xgboost.yaml
```
**Output:**
```
==================================================
EXPERIMENTAL EVALUATION PIPELINE
==================================================
Status:
BLOCKED

Reason:
Finalized gold benchmark is unavailable or input dataset failed validation.
  - Candidate dataset rejected: filename 'uncertainty_benchmark_300_candidate_v1.1.jsonl' indicates candidate/draft data.
  - Line 1 (ID: draft_known_fact_001): Dataset is unannotated candidate data (annotation_status: 'needs_human_review')
  ...

Safe failure safeguard active: No models were trained. No experimental results were generated.
==================================================
```
(Exits with code 2).
