# Gold Benchmark Dataset Splitting & Cross-Validation Guide

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Milestone:** 7.5.4 — Gold Benchmark Split Generation Infrastructure  
**Authoritative Reference:** [docs/EXPERIMENTAL_PROTOCOL.md](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/docs/EXPERIMENTAL_PROTOCOL.md)  
**Execution Script:** [scripts/generate_gold_splits.py](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/scripts/generate_gold_splits.py)  

---

## 1. Scientific Status Declaration

> ### ⚠️ METHODOLOGICAL STATUS NOTICE:
> **A finalized gold benchmark does not currently exist.**  
> - Benchmark questions remain candidate drafts (`data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl`).
> - Human annotation has not yet been executed (`Annotator A: 0/300`, `Annotator B: 0/300`).
> - The annotation blinding key remains quarantined.
> - Finalized gold labels have **NOT** been generated or assigned.
> 
> In accordance with empirical research integrity standards, `scripts/generate_gold_splits.py` operates in **fail-safe mode**: it refuses to partition unannotated candidate data or fabricate splits. Real partition artifacts (`gold_split_v1.json`, `gold_cv5_v1.json`) will be generated strictly after Milestone 7.5.3 gold benchmark finalization.

---

## 2. Why Group-Aware Splitting is Essential

In standard machine learning, random $K$-fold cross-validation or uniform train/test splitting assumes all samples are independently and identically distributed ($i.i.d.$). 

In natural language and LLM uncertainty benchmarks, however, queries often exist as:
1. **Lexical Paraphrases:** (e.g., *"What is 2 + 2?"*, *"What is the sum of two and two?"*, *"What is the result when 2 is added to 2?"*).
2. **Polysemic Variants:** (e.g., *"Who plays the doctor in Doctor Who?"* across different eras).
3. **Adversarial Distractors:** Intended to test model robustness on identical underlying entities or premises.

If one paraphrase appears in the training partition while another paraphrase appears in the test partition:
- The classifier memorizes specific lexical tokens or semantic representations.
- Semantic consistency metrics (which are driven by entity clustering and pairwise embeddings) achieve artificially inflated evaluation scores.
- The evaluation measures **memorization of specific questions**, rather than the model's true capability to estimate epistemic uncertainty on genuinely novel prompts.

To eliminate this vulnerability, each question in the benchmark belongs to an indivisible connected component identified by `question_group_id` (152 total groups across the 300 questions). **All variants of a question must reside in exactly one partition.**

---

## 3. Partitioning Methodology

### A. 70 / 15 / 15 Train-Validation-Test Group Split

The held-out test split is designed for final model assessment, while the validation partition supports hyperparameter tuning and model selection:
- **Train ($70\%$):** $\approx 210$ questions ($\approx 106$ groups)
- **Validation ($15\%$):** $\approx 45$ questions ($\approx 23$ groups)
- **Test ($15\%$):** $\approx 45$ questions ($\approx 23$ groups)

#### Allocation Algorithm
Because group sizes are integers (varying from 1 to 5 questions per group), uniform percentage cuts cannot be made across individual questions. Instead, `GroupStratifiedSplitter` implements a deterministic, multi-objective greedy optimization:
1. Groups are ordered deterministically by total sample count descending, with seeded hash tie-breaking on the group ID.
2. For each group $g$, the allocation evaluates placing $g$ into partition $p \in \{\text{train}, \text{val}, \text{test}\}$.
3. The chosen partition minimizes the global quadratic deviation across total partition size and per-class targets ($N_{\text{KNOWN}}, N_{\text{AMBIGUOUS}}, N_{\text{UNKNOWN}}$):
$$\mathcal{L}(p) = \sum_{p' \in \{\text{train}, \text{val}, \text{test}\}} \left[ \left(\frac{C_{p'} - T_{p'}}{T_{p'}}\right)^2 + \sum_{c \in \mathcal{C}} 2 \cdot \left(\frac{C_{p',c} - T_{p',c}}{T_{p',c}}\right)^2 \right]$$
4. This ensures group atomicity while preserving the exact 33.3% class balance across partitions.

---

### B. 5-Fold StratifiedGroupKFold Cross-Validation

To maximize data efficiency and ensure robust statistical hypothesis testing, the framework implements **5-Fold StratifiedGroupKFold**:
- Each fold consists of:
  - **Training:** 4 folds ($\approx 80\%$, $\approx 240$ samples)
  - **Validation:** 1 fold ($\approx 20\%$, $\approx 60$ samples)
- **Group Isolation:** No group appears in both the training set and the validation set for any fold.
- **Completeness:** Every single benchmark record is assigned to the validation set of **exactly one fold** across the 5 iterations.

---

## 4. Independent Leakage Detection Suite

`LeakageValidator` provides independent verification that runs post-splitting (and can also audit existing artifacts):

| Verification Check | Target Constraint | Enforcement Mechanism |
| :--- | :--- | :--- |
| **Train/Val ID Disjointness** | $\text{IDs}_{\text{train}} \cap \text{IDs}_{\text{val}} = \emptyset$ | Set intersection on question IDs |
| **Train/Test ID Disjointness** | $\text{IDs}_{\text{train}} \cap \text{IDs}_{\text{test}} = \emptyset$ | Set intersection on question IDs |
| **Val/Test ID Disjointness** | $\text{IDs}_{\text{val}} \cap \text{IDs}_{\text{test}} = \emptyset$ | Set intersection on question IDs |
| **Train/Val Group Disjointness** | $\text{Groups}_{\text{train}} \cap \text{Groups}_{\text{val}} = \emptyset$ | Set intersection on `question_group_id` |
| **Train/Test Group Disjointness** | $\text{Groups}_{\text{train}} \cap \text{Groups}_{\text{test}} = \emptyset$ | Set intersection on `question_group_id` |
| **Val/Test Group Disjointness** | $\text{Groups}_{\text{val}} \cap \text{Groups}_{\text{test}} = \emptyset$ | Set intersection on `question_group_id` |
| **Normalized Text Collision** | $\text{Text}_{\text{train}} \cap \text{Text}_{\text{test}} = \emptyset$ | Normalized string matching across splits |
| **CV Fold Group Disjointness** | $\text{Groups}_{\text{train\_fold}} \cap \text{Groups}_{\text{val\_fold}} = \emptyset$ | Verified independently for each of the 5 folds |
| **CV Validation Partition Exhaustion** | $\sum_{k=1}^5 \mathbb{I}(x \in V_k) = 1, \forall x$ | Verification that every sample appears in val once |

Any violation immediately aborts artifact emission and flags the error in the audit report.

---

## 5. Required Gold Benchmark Schema & Provenance

The split generator enforces strict provenance checks on the input JSONL file before proceeding:

1. **Top-Level Record Validation:**
   - `id`: Non-empty canonical question ID (`draft_known_fact_001` or final ID).
   - `question`: Natural language prompt string.
   - `question_group_id`: Connected component ID.
   - `final_gold_label`: The authoritative ground-truth label in `{"KNOWN", "AMBIGUOUS", "UNKNOWN"}`.
   - `benchmark_version`: Version string (e.g. `"1.0"`).
2. **Human Annotation Provenance:**
   - Must contain `human_annotation_provenance` object detailing `annotator_a`, `annotator_b`, `is_unanimous`, `adjudication_required`, and adjudication record where applicable.
3. **Reviewer Agreement Status:**
   - `reviewer_agreement.annotation_status` must equal `"gold_benchmark"`.
   - Records with `"needs_human_review"` are strictly rejected as draft/candidate data.
4. **Anti-Leakage Safeguard:**
   - Any record containing model predictions, entropy values, cluster counts, or extracted feature fields is rejected.

---

## 6. Output Artifacts Specification

When provided with a verified gold benchmark, the pipeline generates:

1. **`data/experiments/gold_split_v1.json`**:
   - Split metadata, random seed, target ratios.
   - Explicit question ID lists for `train`, `val`, and `test`.
   - Complete group-to-partition mapping.
   - Per-partition class distributions.
2. **`data/experiments/gold_cv5_v1.json`**:
   - 5-fold assignments, random seed, fold sample and group counts.
   - Per-fold `train_ids` and `val_ids`.
   - Inverted index: `{question_id: fold_index}` mapping.
3. **`reports/dataset_split_audit_v1.json`**:
   - Complete audit trail, input file checksums, leakage check results, and execution warnings.
4. **`reports/DATASET_SPLIT_AUDIT_REPORT.md`**:
   - Human-readable publication table with split statistics and verification checkmarks.

---

## 7. Execution CLI Commands

### Generating Splits on Finalized Gold Benchmark
```bash
python scripts/generate_gold_splits.py \
    --dataset data/benchmarks/uncertainty_benchmark_300_gold_v1.0.jsonl \
    --manifest data/benchmarks/gold_benchmark_manifest_v1.0.json \
    --seed 42 \
    --output-dir data/experiments \
    --reports-dir reports
```

### Auditing Pre-Existing Split Artifacts
```bash
python scripts/generate_gold_splits.py \
    --validate-existing \
    --dataset data/benchmarks/uncertainty_benchmark_300_gold_v1.0.jsonl \
    --split-file data/experiments/gold_split_v1.json \
    --cv-file data/experiments/gold_cv5_v1.json
```

### Safe Failure on Current Candidate Benchmark
When executed against the unannotated candidate benchmark:
```bash
python scripts/generate_gold_splits.py --dataset data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl
```
**Output:**
```
==================================================
GOLD BENCHMARK SPLIT GENERATION
==================================================
Status:
BLOCKED

Reason:
Input dataset is not a finalized gold benchmark or failed verification.
  - Candidate dataset rejected: filename 'uncertainty_benchmark_300_candidate_v1.1.jsonl' indicates candidate/draft data.
  - Line 1 (ID: draft_known_fact_001): Dataset is unannotated candidate data (annotation_status: 'needs_human_review')
  - Line 1 (ID: draft_known_fact_001): Missing 'human_annotation_provenance' - dataset lacks required human ground-truth provenance
  - Line 1 (ID: draft_known_fact_001): Missing authoritative 'final_gold_label'
  ...

Safe failure safeguard active: Split and CV artifacts were NOT generated.
==================================================
```
(Exits with return code 2).

---

## 8. Scientific Limitations of Group Stratification

1. **Integer Knapsack Variance:** Group-aware stratification operates at the group level. If an entity group contains an odd number of items or mixes question types, achieving exactly 70.00% or equal fold counts is mathematically bounded by integer programming constraints.
2. **Small Class Representation:** If a rare category has fewer groups than cross-validation folds ($G_c < K$), that class cannot appear in every validation fold without violating group isolation. The pipeline explicitly audits and reports these occurrences.
3. **Reproducibility Dependency:** All split operations must use `np.random.default_rng(seed)` with fixed tie-breaking hashes. Re-running the pipeline with different seeds will yield different group assignments while maintaining zero leakage.
