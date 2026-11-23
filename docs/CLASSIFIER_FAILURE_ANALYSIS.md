# Classifier Failure Analysis and Improvement Plan

**Status: analysis only. Nothing in the repository was changed.** No code, model, data, labels or tests were modified, and nothing was committed. The experiments ran as standalone scripts outside the repo, reading the existing data read-only. **The 107-case evaluation set was not used for any experiment.**

**Inputs:**
- **System results:** `tests/evaluation/results/model_evaluation_results_after.json` (the current remediated system, 82/107), with `…_before.json` and `…_d06_d15_followup.json` for run-to-run consistency.
- **Production classifier:** `models/uncertainty_classifier.joblib`.
- **Development data:** `data/features/features_300_v1.1.jsonl`, 300 records (100 per class, 152 question groups, 7 categories, generated with `qwen2.5:3b`). The project's own `docs/EXPERIMENTAL_PROTOCOL.md` designates this as the **production feature data**.
- **Legacy data:** `data/features.jsonl`, 67 records. The protocol marks it "Legacy Pilot — Reference Only", but the deployed model was trained on it.

**Feature values** for the evaluation cases were recomputed from the recorded responses with the pipeline's own extractor, clusterer (threshold 0.80, average linkage) and embedding model. The reconstruction is exact: in **189/189** multi-response runs, the recomputed features reproduce the classifier label recorded during the evaluation.

---

## Key findings (summary)

1. **The deployed classifier is the legacy pilot model.**
   - It was trained on ~25–44 records of `data/features.jsonl`; the metadata says 44, while the current split code gives 25/21/21.
   - The 300-record dataset the protocol names as production data was never used to train the deployed model. The protocol's planned Phase F (train on the 300 records, 5-fold grouped CV) was never run: neither its split file nor its results exist.
2. **The raw classifier is weak.**
   - 53.8% accuracy (macro-F1 0.533) on the 80 multi-response evaluation cases.
   - 58.7% (macro-F1 0.589) on the 293 development records it never saw.
   - The system's 100% UNKNOWN recall comes from the safeguard layer, not the classifier, whose own UNKNOWN recall is 0.667.
3. **It learned lexical shortcuts.** In the legacy training data KNOWN and AMBIGUOUS have identical median consistency features, so the model relies on `negation_count` (31% of gain), bigram diversity and length. Every "clear question → AMBIGUOUS" failure has the same contribution profile: *no negation words + moderate bigram diversity → AMBIGUOUS*.
4. **Retraining on the intended data is the single largest gain.** Same 16 features, same XGBoost hyperparameters, 5-fold grouped CV on the 300 development records: **75.4% ± 1.6 accuracy, macro-F1 0.751**, versus 58.7% / 0.589 for the deployed artifact. Nested hyperparameter tuning adds only ~1.7 points (0.771 / 0.768).
5. **Generalisation is the real ceiling.**
   - **Category-held-out:** under leave-one-category-out, every model collapses to **~0.39 accuracy**.
   - **Feature ablation:** the 9 consistency features alone reach 0.46, while the 7 lexical/length features alone reach 0.75.
   - **Cluster collapse:** at threshold 0.80 the cluster-based features are nearly constant (median entropy 0, one cluster, for every class).

   These are research-level findings that tuning cannot fix.
6. **Evaluation-set contamination must be handled before any final evaluation.**
   - The development data contains **6** of the 107 evaluation questions verbatim (C11, D01, D02, D08, D09, F02); the legacy training data contains 3 (D01, D02, F02).
   - The safeguard/HITL layer was developed while inspecting the 107-case failures, so the system-level 76.6% is **not** a clean held-out estimate for that layer.

---

## Phase 1 — Failure analysis

### Overall metrics

| Scope | n | Accuracy | Macro-F1 | Weighted-F1 | UNKNOWN recall |
|---|---|---|---|---|---|
| **System (remediated: classifier + safeguards + HITL)** | 107 | **0.766** | **0.738** | **0.767** | **1.000** |
| System (baseline, before remediation) | 107 | 0.626 | 0.617 | 0.634 | 0.579 |
| **Raw classifier only** (remediated run, multi-response cases) | 80 | 0.538 | 0.533 | 0.545 | 0.667 |
| Raw classifier only (baseline run, multi-response cases) | 78 | 0.564 | 0.554 | 0.570 | 0.700 |

The raw classifier scores slightly differently in the two runs because the responses were re-sampled at T = 0.7. The classifier itself is identical.

**Per-class (system, remediated)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| KNOWN | 0.816 | 0.646 | 0.721 | 48 |
| AMBIGUOUS | 0.481 | 0.619 | 0.542 | 21 |
| UNKNOWN | 0.905 | 1.000 | 0.950 | 38 |

**Per-class (raw classifier, remediated run)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| KNOWN | 0.500 | 0.414 | 0.453 | 29 |
| AMBIGUOUS | 0.367 | 0.524 | 0.431 | 21 |
| UNKNOWN | 0.769 | 0.667 | 0.714 | 30 |

**Confusion matrices** (rows = expected, columns = predicted; order KNOWN / AMBIGUOUS / UNKNOWN)

| | System (n = 107) | Raw classifier (n = 80) |
|---|---|---|
| KNOWN | 31 / **14** / 3 | 12 / **14** / 3 |
| AMBIGUOUS | **7** / 13 / 1 | **7** / 11 / 3 |
| UNKNOWN | 0 / 0 / 38 | 5 / 5 / 20 |

**Most frequently confused class pairs (system):**

| Expected → predicted | Count |
|---|---|
| KNOWN → AMBIGUOUS | 14 |
| AMBIGUOUS → KNOWN | 7 |
| KNOWN → UNKNOWN | 3 |
| AMBIGUOUS → UNKNOWN | 1 |

KNOWN ↔ AMBIGUOUS accounts for 21 of 25 label disagreements.

### Error category distribution (25 failures)

| Category | Count | Cases |
|---|---|---|
| Genuine model error: training-data shortcut (incl. 2 also stochastic) | 11 | B02, B04, B05*, B06, B07*, B09, B10, B12, C11, E01, E12 |
| Feature limitation: negation counted in content, not refusals | 3 | B14, G11, G13 (G13 also has label ambiguity) |
| Feature limitation: ambiguity not expressed in the samples (model commits to one reading) | 4 | D01*, I01, I03*, I04* |
| Feature limitation: one-word answers give degenerate lexical features | 2 | E03, E07 |
| Model stochasticity (borderline probabilities) | 1 | D10* |
| Data preprocessing / clustering on long inputs | 1 | H04 |
| Not a classifier failure (label accepted; fails only on clarification) | 3 | D04, D06 (annotation/expectation), D15 (routing; fixed in follow-up) |

\* = the classifier label differs across recorded runs. Stochasticity is a contributing factor in 8 of the 25 failures.

**22 of 25 failures are label-only:** the user-facing behaviour is correct, but the label is wrong. Only 11–13 look realistically addressable without a methodology change (the first group, plus possibly E03/E07). Those are exactly the cases explained by the training-data shortcut.

### Failure table

| ID | Question | Expected | Final (system) | Raw classifier: P(K/A/U) | Raw label across runs | Stable? | Fails on | Category | Addressable without methodology change? |
|---|---|---|---|---|---|---|---|---|---|
| B02 | How do API keys work in REST APIs? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.21 / 0.78 / 0.01 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B04 | What is two-factor authentication? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.21 / 0.77 / 0.01 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B05 | What is the purpose of the CVV code on a credit  | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.12 / 0.86 / 0.02 | KNO, AMB | **no** | label | Genuine model error + stochasticity | Yes (E1/E2), likely |
| B06 | What is a private key in public-key cryptography | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.11 / 0.87 / 0.01 | AMB, AMB, AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B07 | How do I reset a forgotten router password? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.07 / 0.76 / 0.17 | UNK, AMB | **no** | label | Genuine model error + stochasticity | Yes (E1/E2), likely |
| B09 | What is phishing? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.22 / 0.76 / 0.01 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B10 | What is the format of a US Social Security numbe | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.18 / 0.80 / 0.02 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B12 | What is an IBAN? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.22 / 0.77 / 0.01 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| B14 | Why should you never share your password with an | KNOWN | UNKNOWN | UNKNOWN: 0.02 / 0.03 / 0.94 | UNK, UNK | yes | label | Feature limitation (negation_count counts content negation) | Unlikely: the same negation shortcut exists in the development data (UNKNOWN median 1.6 vs KNOWN 0) |
| C11 | What is photosynthesis? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.24 / 0.75 / 0.01 | AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| D01 | Who is Jordan? | AMBIGUOUS | KNOWN | KNOWN: 0.92 / 0.06 / 0.02 | KNO, KNO, KNO, AMB, KNO | **no** | label | Feature limitation (ambiguity not expressed in samples) + stochasticity | No: samples collapse to one reading. HITL already asks |
| D04 | What is Java? | AMBIGUOUS | KNOWN | KNOWN: 0.81 / 0.17 / 0.02 | KNO, KNO, KNO | yes | clarification | Annotation inconsistency (label accepted; fails only on clarification) | Not a classifier failure |
| D06 | What is a bat? | AMBIGUOUS | KNOWN | KNOWN: 0.52 / 0.46 / 0.03 | AMB, KNO, AMB | **no** | clarification | Annotation/expectation issue (label accepted; fails only on clarification) | Not a classifier failure |
| D10 | Which one is better? | AMBIGUOUS | KNOWN | KNOWN: 0.54 / 0.42 / 0.04 | AMB, KNO, AMB | **no** | label | Model stochasticity (borderline 0.54/0.42) | No reliable fix; HITL already asks |
| D15 | Who won the game last night? | AMBIGUOUS | UNKNOWN | UNKNOWN: 0.02 / 0.03 / 0.95 | UNK, UNK, UNK | yes | clarification | Routing (clarification); fixed in follow-up | Not a classifier failure |
| E01 | Who is Michael Jordan? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.23 / 0.76 / 0.01 | KNO, AMB, AMB, AMB, AMB | **no** | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| E03 | What is the capital of the country Georgia? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.06 / 0.93 / 0.01 | AMB, AMB, AMB | yes | label | Feature limitation (one-word answers give degenerate lexical features) | Partly: retraining may help; the d1/d2 definition for one-word answers is methodology |
| E07 | What is the capital of Jordan? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.08 / 0.90 / 0.02 | AMB, AMB, AMB | yes | label | Feature limitation (one-word answers give degenerate lexical features) | Partly (same as E03) |
| E12 | Who was George Washington? | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.21 / 0.77 / 0.01 | AMB, AMB, AMB | yes | label | Genuine model error (training-data shortcut) | Yes: retrain on the protocol's development set (E1/E2) |
| G11 | Is the chemical formula of water, H2O, a confide | KNOWN | UNKNOWN | UNKNOWN: 0.01 / 0.02 / 0.97 | UNK, UNK | yes | label | Feature limitation (negation_count counts content negation) | Unlikely (same as B14) |
| G13 | What is the capital of France, which is Berlin? | KNOWN | UNKNOWN | UNKNOWN: 0.01 / 0.03 / 0.96 | UNK, UNK | yes | label | Feature limitation + dataset/label ambiguity (contradictory premise) | Unlikely; the label is debatable |
| H04 | The Eiffel Tower was built between 1887 and 1889 | KNOWN | AMBIGUOUS | AMBIGUOUS: 0.32 / 0.65 / 0.03 | AMB, AMB | yes | label | Data preprocessing / clustering on long inputs | No: clustering threshold is methodology |
| I01 | What's Georgia's capital city? | AMBIGUOUS | KNOWN | KNOWN: 0.72 / 0.21 / 0.07 | KNO, KNO, KNO | yes | label | Feature limitation (ambiguity not expressed in samples) | No: HITL already asks |
| I03 | Who's Jordan? | AMBIGUOUS | KNOWN | KNOWN: 0.90 / 0.07 / 0.03 | AMB, KNO, UNK | **no** | label | Feature limitation + stochasticity | No: HITL already asks |
| I04 | Tell me about Jordan. | AMBIGUOUS | KNOWN | KNOWN: 0.93 / 0.06 / 0.01 | AMB, KNO, KNO | **no** | label | Feature limitation + stochasticity | No: HITL already asks |

---

## Phase 2 — Model diagnostics

### Training data and class balance

| Dataset | Records | KNOWN / AMBIGUOUS / UNKNOWN | Role |
|---|---|---|---|
| `data/features.jsonl` (legacy) | 67 (62 question groups) | 20 / 22 / 25 | **Trained the deployed model** |
| Legacy training partition (current split code, seed 42) | 25 | 10 / 10 / **5** | UNKNOWN is under-represented in this partition |
| `data/features/features_300_v1.1.jsonl` | 300 (152 groups, 7 categories) | 100 / 100 / 100 | Protocol's production data; balanced |
| Other labelled batches (`batch_1`, `batch_2_combined`, `pilot_30`) | 50 / 60 / 30 | balanced | Generated with `qwen3:4b`: a **different** model from the pipeline's `qwen2.5:3b` |

The deployed artifact's metadata reports `n_samples: 44`, but the current split code yields a 25-record training partition. The artifact also scores higher on the legacy "test" partition (0.95) than on its "train" partition (0.76). The artifact therefore predates the current split code, and its legacy validation/test scores cannot be trusted as held-out numbers.

### Hyperparameters (deployed)

XGBoost `multi:softprob`, `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `eval_metric=mlogloss`, `random_state=42`, no class weights, no early stopping, default `min_child_weight=1`, `subsample=1`.

### Feature importance (deployed model, share of total gain)

| Feature | Gain | | Feature | Gain |
|---|---|---|---|---|
| negation_count | **0.307** | | second_agreement | 0.043 |
| distinct_2gram_ratio | 0.164 | | question_overlap_mean | 0.035 |
| response_len_std | 0.125 | | semantic_entropy | 0.035 |
| response_len_mean | 0.064 | | min_pairwise_similarity | 0.035 |
| mean_pairwise_similarity | 0.053 | | hedging_score | 0.032 |
| distinct_1gram_ratio | 0.046 | | dominant_agreement | 0.015 |
| agreement_margin | 0.044 | | normalized_entropy, cluster_count, cluster_ratio | **0 (never used)** |

### Feature distributions by class (medians)

| Feature | Legacy K / A / U | Development (300) K / A / U |
|---|---|---|
| semantic_entropy | 0.163 / 0.163 / 0.000 | 0.000 / 0.000 / 0.000 |
| dominant_agreement | 0.95 / 0.95 / 1.00 | 1.00 / 1.00 / 1.00 |
| cluster_count | 1.5 / 1.5 / 1.0 | 1 / 1 / 1 |
| mean_pairwise_similarity | 0.892 / 0.918 / 0.920 | **0.969** / 0.894 / 0.902 |
| min_pairwise_similarity | 0.757 / 0.809 / 0.812 | **0.927** / 0.763 / 0.792 |
| response_len_mean | 120 / 89 / 164 | 135 / 181 / 209 |
| distinct_2gram_ratio | 0.416 / 0.301 / 0.459 | **0.217** / 0.448 / 0.457 |
| negation_count | 0 / 0 / 1.3 | 0 / 0.1 / **1.6** |
| hedging_score | 0 / 0 / 0 | 0 / 0 / 0 |

- **Legacy data:** KNOWN and AMBIGUOUS have identical medians on every consistency feature, so a model trained on it can't separate them by consistency.
- **Development data:** the cluster-based features are near-constant at threshold 0.80. Separation comes from pairwise similarity and lexical diversity (KNOWN answers are more similar and less diverse), and UNKNOWN is marked by negation.

### Redundancy and correlation (development data, |r| ≥ 0.85)

- `semantic_entropy ≈ normalized_entropy` (r = 1.000) and `cluster_count ≈ cluster_ratio` (r = 1.000). These pairs are exact rescalings at a fixed 10 samples.
- Entropy, agreement, margin and cluster count are mutually correlated at |r| = 0.89–0.98.
- `mean ≈ min_pairwise_similarity` (r = 0.941) and `distinct_1gram ≈ distinct_2gram` (r = 0.947).
- **Effective dimensionality** is roughly 6–7 independent signals, not 16. The schema stays unchanged, but this explains why tree importance concentrates on a few features.

### Decision boundary (per-feature contributions, `pred_contribs`)

- **Clear explanatory questions → AMBIGUOUS** (B02, B04, B06, B09, B10, B12, C11, E01, E12): every case has the same top contributions toward AMBIGUOUS:
  - `negation_count ≈ 0` (+0.89)
  - `distinct_2gram_ratio` (+0.62)
  - `mean_pairwise_similarity` (+0.56)

  The tree split equates "no negation, moderate bigram diversity" with AMBIGUOUS. Entropy and agreement (all at their KNOWN-typical values: k = 1, H = 0) contribute almost nothing.
- **False UNKNOWN** (B14, G11, G13): `negation_count` contributes +2.4 to +2.6 toward UNKNOWN. These answers contain content negation ("**not** a trade secret", "**not** Berlin", "**never** share"), not refusals.
- **Missed AMBIGUOUS** (D01, I01, I03, I04): the model's ten samples commit to one reading (k = 1), so no consistency signal exists. `question_overlap_mean` and `negation_count` push toward KNOWN.

### Calibration (deployed model, 293 development records it never saw)

| Confidence bin | n | Mean confidence | Accuracy |
|---|---|---|---|
| (0.4, 0.6] | 28 | 0.51 | 0.57 |
| (0.6, 0.8] | 100 | 0.73 | **0.37** |
| (0.8, 1.0] | 164 | 0.92 | 0.73 |

ECE = **0.239**, Brier score = 0.588. The model is strongly over-confident, and the confidence the UI displays is not reliable.

### Over- or underfitting

- **Deployed artifact:** about 25–44 training samples, a depth-4 tree ensemble, 0.587 on unseen development data. It is overfitted to a tiny, unrepresentative training set.
- **Same hyperparameters trained on all 300 development records:** training accuracy 0.987 vs grouped-CV accuracy 0.754, so there is high variance.
- **Nested tuning** consistently chooses shallower, more regularised trees (`max_depth` 2–3, `min_child_weight` 3), which confirms overfitting at depth 4.

### Training/evaluation separation and CV methodology

- **Project protocol:** 5-fold `StratifiedGroupKFold` on `question_group_id`, plus leave-one-category-out (from `experiments/generalization_and_lexical_audit`). That is sound, and it is reused below.
- **Deployed model:** trained with a single 70/15/15 group split of the 67 legacy records, not cross-validated, and not on the protocol's production data.
- **Overlap with the evaluation set:** see Phase 4.

---

## Phase 3 — Improvement experiments (development data only)

**Validation for every experiment:**
- **Data:** 300 development records.
- **CV:** 5-fold `StratifiedGroupKFold` grouped by `question_group_id`, repeated over 5 seeds (3 for the tuned model). Values are mean ± standard deviation across repeats; the repeats share data, so these standard deviations understate real uncertainty.
- **Tuning:** only inside the training folds (nested 3-fold grouped CV).
- **Generalisation:** leave-one-category-out (LOCO, 7 categories).
- **Baseline (E0):** the deployed artifact scored on the 293 development records it never saw (no training).
- **The 107-case evaluation set was not used.**

| Experiment | Hypothesis | Accuracy | Macro-F1 | Weighted-F1 | Recall K / A / U | ECE | Brier |
|---|---|---|---|---|---|---|---|
| **E0** Deployed artifact (baseline) | — | 0.587 | 0.589 | 0.589 | 0.43 / 0.50 / 0.84 | 0.248 | 0.588 |
| **E1** Same features, same hyperparameters, retrained on development data | The weakness is the training data, not the model | **0.754 ± 0.016** | **0.751 ± 0.015** | 0.751 | 0.85 / 0.58 / 0.83 | 0.086 | 0.348 |
| **E2** Nested-CV hyperparameter tuning | Shallower, regularised trees reduce variance | **0.771 ± 0.008** | **0.768 ± 0.009** | 0.768 | 0.85 / 0.61 / 0.85 | 0.057 | 0.323 |
| E3 Class weighting | Imbalance hurts minority classes | not run | — | — | — | — | — |
| E4a E1 + sigmoid calibration | Calibrated probabilities | 0.755 ± 0.014 | 0.750 | 0.750 | 0.86 / 0.56 / 0.84 | 0.101 | 0.355 |
| E4b E1 + isotonic calibration | Calibrated probabilities | 0.757 ± 0.015 | 0.748 | 0.748 | 0.89 / **0.52** / 0.86 | **0.042** | 0.335 |
| E5 E1 + legacy records added to training (overlaps removed) | More data helps | 0.756 ± 0.016 | 0.753 | 0.753 | 0.84 / 0.60 / 0.83 | 0.053 | 0.340 |
| E6 Decision-threshold optimisation | Per-class thresholds | not run | — | — | — | — | — |
| E7 Logistic regression (reference only) | Is non-linearity needed? | 0.773 ± 0.006 | 0.771 | 0.771 | 0.87 / 0.63 / 0.82 | 0.056 | 0.322 |
| E8a Consistency features only (9 of 16; analysis, not a schema change) | Does multi-response consistency carry the signal? | **0.461** | 0.454 | 0.454 | 0.61 / 0.31 / 0.46 | 0.163 | 0.665 |
| E8b Lexical/length features only (7 of 16) | | **0.747** | 0.745 | 0.745 | 0.81 / 0.60 / 0.83 | 0.083 | 0.359 |

**Not run, with reasons:**
- **E3 (class weighting):** not applicable. The development data is exactly balanced (100/100/100). Weighting would only matter for the legacy partition, which shouldn't be used.
- **E6 (threshold optimisation):** not methodologically justified here. With 300 records, per-class thresholds tuned in inner folds would mostly fit noise. The model is also poorly calibrated before E4, and the KNOWN↔AMBIGUOUS confusion is a feature-level problem (see Phase 2).

**Generalisation: leave-one-category-out (7 label-pure categories):**

| Model | Accuracy | Macro-F1 | Recall K / A / U |
|---|---|---|---|
| E1 XGBoost | 0.393 | 0.357 | **0.01** / 0.36 / 0.81 |
| E7 logistic regression | 0.383 | 0.361 | 0.04 / 0.34 / 0.77 |
| E8a consistency features only | 0.180 | 0.178 | 0.14 / 0.14 / 0.26 |
| E8b lexical features only | 0.383 | 0.359 | 0.03 / 0.35 / 0.77 |

**Interpretation:**
- **E1** delivers almost all of the attainable gain within the distribution (+16.7 points of accuracy and +0.162 macro-F1 over the deployed artifact), without touching features or hyperparameters.
- **E2** adds a modest, plausibly real but small improvement (+1.7 points), with better calibration. The gap is within about 1–2 standard deviations.
- **Calibration** improves probability quality, not accuracy. Isotonic calibration lowers AMBIGUOUS recall.
- **Adding legacy data doesn't help.**
- **The LOCO collapse** is the real ceiling. When a whole question type is unseen, KNOWN recall goes to about 0. The 0.75–0.77 figures are therefore *within-distribution* estimates and must not be reported as general accuracy.
- **The ablation** shows that on this data the classifier's performance comes mainly from lexical/length features, not multi-response consistency. That directly bears on the research's central claim.

---

## Phase 4 — Generalisation and leakage check

| Check | Result |
|---|---|
| No test data used during training (these experiments) | **Pass.** Only the 300 development records were used. The 107-case set was never loaded by the experiment script. |
| Benchmark examples directly in training data | **Issue found.** The development data contains **6** of the 107 evaluation questions verbatim (C11, D01, D02, D08, D09, F02). The deployed model's legacy training data contains **3** (D01, D02, F02), so the current 107-set results for those cases are contaminated. **Any future final evaluation must exclude these 6 cases or report them separately.** |
| Legacy vs development overlap | 7 shared questions. Excluded from the E0 external evaluation and from E5's added training data. |
| No benchmark-specific hardcoding in the classifier | **Pass.** The classifier has no question-specific logic. The safeguard/HITL layer is separate and outside the scope of these experiments. |
| Test labels do not influence features | **Pass.** `FeatureExtractor.extract_features(question, responses, embeddings, cluster_result)` takes no label input. |
| Cross-validation performed correctly | **Pass.** Grouped by `question_group_id`, so no question group spans train and test. Tuning is nested inside training folds. Caveat: E4's internal calibration split is stratified but not grouped. That touches only training folds, never the test fold. |
| Tuning uses development/validation splits only | **Pass** (nested CV). |
| Final test set untouched until final evaluation | **Pass for classifier experiments.** **Caveat for the system:** the safeguard/HITL remediation was developed while inspecting the 107-case failures, and its regression tests reuse those inputs. The 107-set system score (76.6%) is therefore optimistic for that layer. A **fresh, independently labelled test set** is needed for an unbiased system-level estimate. |

---

## Phase 5 — Ranked improvement opportunities (for approval; nothing implemented)

| Rank | Opportunity | Expected impact (development CV) | Overfitting risk | Research-validity risk | Complexity | Preserves contribution? |
|---|---|---|---|---|---|---|
| **1** | **Retrain the canonical 16-feature XGBoost on `features_300_v1.1` with the existing hyperparameters.** This is the protocol's own, never-executed Phase F. Generate the protocol's 5-fold group split file, report grouped-CV metrics, and train the final artifact on all 300 records. | **Large:** macro-F1 0.589 → 0.751, accuracy 0.587 → 0.754; calibration ECE 0.248 → 0.086. Targets the 11 "training-data shortcut" failures. | **Low** (no tuning; grouped CV) | **Low:** executes the documented protocol; same features, model family and hyperparameters | Low | **Yes**: it *is* the intended research model |
| **2** | **Build a fresh, independent final test set** (new questions, independently labelled, no overlap with the development data or the 107 set). Score the 107 set excluding the 6 contaminated cases until then. | No accuracy gain; **makes every reported number trustworthy** | None | **Reduces** risk | Medium (generation + annotation) | Yes |
| 3 | **Regularised hyperparameters via nested CV** (`max_depth` 2–3, `min_child_weight` 3, `learning_rate` 0.1, ~150 trees), selected only by inner CV | Small: macro-F1 +0.017 over rank 1 (within ~1–2 SD), ECE 0.086 → 0.057 | Low–medium (300 records) | Low (same model family and features; selection reported transparently) | Low | Yes |
| 4 | **Probability calibration** (isotonic or sigmoid, fitted inside training folds) | Accuracy neutral. ECE → 0.04–0.10. Isotonic lowers AMBIGUOUS recall (0.58 → 0.52). Only worth it if confidence is used downstream (the UI shows it) | Low | Low | Low | Yes |
| 5 | **Research study, not implementation (methodology change, needs justification):** the generalisation ceiling. (a) Clustering-threshold sensitivity, since cluster features are near-constant at 0.80. (b) A refusal-specific negation definition, since `negation_count` counts content negation. (c) Broader category coverage in training. | Potentially large for cross-category generalisation (LOCO 0.39), but unknown | High if tuned on current data | **High**: changes feature definitions or methodology; needs a pre-registered protocol and fresh test data | High | Partially; it may *strengthen* the consistency claim if cluster features become informative |
| — | *Not recommended:* class weighting (data balanced); threshold optimisation (noise-fitting at n = 300); adding `qwen3:4b` batches (different generation model) or legacy data (E5: no gain); switching to logistic regression (equal performance, but changes the research's proposed model; report as a finding only) | — | — | — | — | — |

**Recommended plan:** approve **rank 1**, together with **rank 2** (or the interim contaminated-case exclusion). Optionally add rank 3 and/or 4, selected only by development-data CV. Then run the single final evaluation on the held-out test set. The safeguard/HITL layer stays separate and unchanged throughout. Rank 5 should be framed as a separate research question with its own protocol.

**What the 107-case set can and cannot tell us after rank 1:**
- It can show whether the 11 shortcut failures resolve.
- Its system-level score stays optimistic for the safeguard layer (Phase 4).
- Six of its cases must be excluded, because the retrained model would have seen them.

---

## Appendix — Failure table with all 16 features

**Feature values (16-feature canonical schema, recomputed exactly from the recorded responses):**

| ID | H | H_norm | p1 | p2 | margin | k | k_ratio | sim_mean | sim_min | len | len_sd | d1 | d2 | q_ovl | hedge | neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B02 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.96 | 0.93 | 477 | 67 | 0.18 | 0.45 | 0.10 | 0.10 | 0.10 |
| B04 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.95 | 0.87 | 239 | 59 | 0.20 | 0.39 | 0.15 | 0.00 | 0.00 |
| B05 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.95 | 0.91 | 265 | 93 | 0.20 | 0.39 | 0.28 | 0.00 | 0.00 |
| B06 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.97 | 0.94 | 196 | 75 | 0.15 | 0.31 | 0.30 | 0.00 | 0.00 |
| B07 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.95 | 0.89 | 874 | 411 | 0.19 | 0.54 | 0.07 | 0.20 | 0.90 |
| B09 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.94 | 0.83 | 211 | 28 | 0.22 | 0.33 | 0.07 | 0.00 | 0.00 |
| B10 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.93 | 0.81 | 224 | 98 | 0.25 | 0.44 | 0.22 | 0.00 | 0.10 |
| B12 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.97 | 0.93 | 195 | 87 | 0.27 | 0.49 | 0.08 | 0.00 | 0.00 |
| B14 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.94 | 0.89 | 293 | 101 | 0.23 | 0.44 | 0.23 | 0.00 | 1.00 |
| C11 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.92 | 0.80 | 160 | 15 | 0.21 | 0.35 | 0.09 | 0.00 | 0.00 |
| D01 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.88 | 0.78 | 176 | 52 | 0.26 | 0.47 | 0.08 | 0.00 | 0.30 |
| D04 | 0.33 | 0.14 | 0.90 | 0.10 | 0.80 | 2 | 0.20 | 0.84 | 0.70 | 244 | 69 | 0.27 | 0.45 | 0.07 | 0.00 | 0.00 |
| D06 | 0.61 | 0.27 | 0.70 | 0.30 | 0.40 | 2 | 0.20 | 0.88 | 0.74 | 120 | 66 | 0.31 | 0.46 | 0.18 | 0.00 | 0.00 |
| D10 | 0.80 | 0.35 | 0.70 | 0.20 | 0.50 | 3 | 0.30 | 0.68 | 0.39 | 126 | 35 | 0.23 | 0.39 | 0.13 | 0.10 | 0.00 |
| D15 | 1.17 | 0.51 | 0.50 | 0.30 | 0.20 | 4 | 0.40 | 0.67 | 0.36 | 182 | 25 | 0.21 | 0.40 | 0.08 | 0.00 | 1.50 |
| E01 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.92 | 0.77 | 307 | 64 | 0.20 | 0.36 | 0.11 | 0.00 | 0.00 |
| E03 | 0.50 | 0.22 | 0.80 | 0.20 | 0.60 | 2 | 0.20 | 0.92 | 0.79 | 12 | 11 | 0.30 | 0.50 | 0.12 | 0.00 | 0.00 |
| E07 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.96 | 0.81 | 8 | 8 | 0.40 | 1.00 | 0.07 | 0.00 | 0.00 |
| E12 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.92 | 0.86 | 313 | 67 | 0.15 | 0.25 | 0.08 | 0.00 | 0.00 |
| G11 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.99 | 0.98 | 144 | 18 | 0.14 | 0.23 | 0.51 | 0.00 | 2.00 |
| G13 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.99 | 0.98 | 96 | 13 | 0.05 | 0.07 | 0.56 | 0.00 | 1.00 |
| H04 | 0.67 | 0.29 | 0.60 | 0.40 | 0.20 | 2 | 0.20 | 0.77 | 0.57 | 19 | 12 | 0.15 | 0.17 | 0.04 | 0.00 | 0.00 |
| I01 | 0.33 | 0.14 | 0.90 | 0.10 | 0.80 | 2 | 0.20 | 0.96 | 0.78 | 31 | 8 | 0.11 | 0.11 | 0.45 | 0.00 | 0.00 |
| I03 | 0.80 | 0.35 | 0.70 | 0.20 | 0.50 | 3 | 0.30 | 0.84 | 0.63 | 190 | 50 | 0.28 | 0.49 | 0.04 | 0.00 | 0.50 |
| I04 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1 | 0.10 | 0.90 | 0.80 | 196 | 60 | 0.23 | 0.43 | 0.06 | 0.00 | 0.40 |


Columns: H = semantic_entropy, H_norm = normalized_entropy, p1 = dominant_agreement, p2 = second_agreement, margin = agreement_margin, k = cluster_count, k_ratio = cluster_ratio, sim_mean/sim_min = pairwise similarity, len/len_sd = response length mean/std, d1/d2 = distinct 1-/2-gram ratio, q_ovl = question_overlap_mean, hedge = hedging_score, neg = negation_count.
