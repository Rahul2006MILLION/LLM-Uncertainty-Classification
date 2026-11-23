# Production Retraining Experiment

**Question:** if the deployed classifier is retrained on the project's intended 300-record production dataset, with the same features, implementation and hyperparameters, what changes?

**Scope and safeguards:**
- **Isolated experiment.** The production model (`models/uncertainty_classifier.joblib`), the feature pipeline, the safeguard/HITL layer, the datasets, the labels and the 107-case benchmark were **not modified**. SHA-256 checksums of the production model, its metadata, both datasets, the benchmark definition and the recorded benchmark results are byte-identical before and after each run.
- **Not committed.**

**Reproduce:**

```
.venv/bin/python experiments/production_retraining/run_retraining_experiment.py
```

- The run is deterministic: a second run reproduced every number.
- Ollama (`all-minilm`) is needed only for the semantic overlap check and for recomputing benchmark features from recorded responses.
- **Outputs:**
  - `experiments/production_retraining/results/retraining_results.json` (all numbers below)
  - `models/candidates/uncertainty_classifier_f300_candidate.joblib` (+ `_metadata.json`, marked `promoted: false`)

**Models compared:**
- **A:** the deployed model, trained on the legacy pilot data (`data/features.jsonl`, metadata `n_samples` = 44).
- **B:** the candidate, trained on `data/features/features_300_v1.1.jsonl` after leakage exclusions (**280** records).

---

## 1. Dataset validation

| Check | Result |
|---|---|
| Expected records | **300** records, 152 question groups, 7 categories |
| Duplicate IDs / duplicate questions (normalised text) | **0 / 0** |
| Label distribution | KNOWN 100 / AMBIGUOUS 100 / UNKNOWN 100; `label` and `gold_label` agree on every record; no invalid labels |
| Missing or invalid feature values (NaN, inf, non-numeric, ratio features outside [0, 1]) | **0** |
| Canonical 16-feature schema | Every record has exactly the 16 `FEATURE_NAMES` keys, no more and no fewer |
| Clustering threshold recorded in rows | 0.80 for all 300 (matches the pipeline config) |
| Categories | determinate_factual 50, concept_definition 50, genuine_ambiguity 60, subjective_questions 40, private_inaccessible_info 35, unknowable_future_events 35, false_premises 30 |

### Overlap with the 107-case benchmark and exclusions

- **Detection:**
  - **Exact matches** on normalised question text: 6.
  - **Semantic near-duplicates:** `all-minilm` cosine ≥ 0.90, 18 pairs. All 18 were reviewed and are genuine paraphrases.
- **Rule:** exclude the **whole question group** of any matched record, because groups hold paraphrase variants of the same question.

| Excluded group | Records | Matched benchmark cases | Match type |
|---|---|---|---|
| grp_def_photosynthesis | 2 (KNOWN) | C11 | exact |
| grp_poly_jordan_who | 2 (AMBIGUOUS) | D01, I03 | exact / paraphrase (0.977) |
| grp_poly_georgia_capital | 3 (AMBIGUOUS) | D02, I01, I02, E03, E04 | exact / paraphrases (0.91–0.97) |
| grp_subj_prog_lang | 2 (AMBIGUOUS) | D08, I07 | exact / paraphrase (0.975) |
| grp_subj_best_athlete | 2 (AMBIGUOUS) | D09 | exact |
| grp_priv_user_thoughts | 2 (UNKNOWN) | F02 | exact |
| grp_capital_france | 3 (KNOWN) | C01, G08 | paraphrase ("official capital city of France", 0.92) |
| grp_highest_mountain | 2 (KNOWN) | E10 | paraphrase (0.906) |
| grp_poly_springfield_state | 2 (AMBIGUOUS) | D07 | paraphrase (0.923) |

- **Result:** 9 groups and **20 records** excluded. **280 records** remain (KNOWN 93 / AMBIGUOUS 89 / UNKNOWN 98).
- **E03 and E04:** these ("capital of the *country* / *US state* of Georgia") are disambiguated variants rather than identical questions. The group was excluded conservatively.
- **Legacy overlap:** 4 of the 280 retained records also appear in the legacy data the deployed model was trained on. They are removed from every A-vs-B comparison ("fair" subset, n = 276).

## 2. Training protocol

| Item | Value |
|---|---|
| Classifier | `src/classification/classifier.py::UncertaintyClassifier(model_type="xgboost")`, the existing implementation, unchanged |
| Hyperparameters | Production defaults: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42`, `eval_metric=mlogloss`. **No tuning.** |
| Features | The canonical 16 (`FEATURE_NAMES`), as pre-extracted in the dataset |
| Cross-validation | The project's own `scripts/generate_gold_splits.py::FiveFoldCrossValidator`, imported unchanged: 5-fold `StratifiedGroupKFold`, group = `question_group_id`, seed 42 |
| Variance across splits | Repeated with seeds 42–46 |
| Generalisation | Leave-one-category-out (7 categories) |
| Final candidate | Trained on all 280 retained records; saved to `models/candidates/` |

## 3. Model comparison: cross-validation (identical held-out records)

A was scored on exactly the same held-out records as B, excluding the 4 records A may have seen (n = 276). A is a fixed artifact, so its fold variance reflects fold composition only.

| Metric | **A** Deployed (legacy-trained) | **B** Candidate (production data) |
|---|---|---|
| Training samples | 44 (metadata) | 280 (≈ 224 per CV fold) |
| Accuracy | 0.605 | **0.764** |
| Macro-F1 | 0.602 | **0.757** |
| Weighted-F1 | 0.609 | **0.761** |
| Precision K / A / U | 0.51 / 0.43 / 0.89 | **0.74 / 0.68** / 0.85 |
| Recall K / A / U | 0.42 / 0.55 / 0.84 | **0.80 / 0.59 / 0.89** |
| **UNKNOWN recall** | 0.84 | **0.89** |
| Per-fold macro-F1 (mean ± SD) | 0.597 ± 0.055 | **0.756 ± 0.058** |
| Per-fold UNKNOWN recall (mean ± SD) | 0.839 ± 0.168 | 0.889 ± 0.149 |
| Repeated CV, seeds 42–46 (all 280 records) | — | acc 0.781 ± 0.021, macro-F1 **0.775 ± 0.021** |

Per-fold macro-F1 (fair subset):

| Fold | n | A | B |
|---|---|---|---|
| 0 | 55 | 0.512 | 0.692 |
| 1 | 55 | 0.573 | 0.701 |
| 2 | 55 | 0.642 | 0.848 |
| 3 | 55 | 0.669 | 0.793 |
| 4 | 56 | 0.592 | 0.744 |

**Significance** (paired, same 276 records):
- B is better in **5 of 5 folds**.
- Records only B gets right: 69. Records only A gets right: 25. Exact McNemar **p = 6.3 × 10⁻⁶**.
- Bootstrap 95% CI for the macro-F1 gain: **[+0.091, +0.220]**. This resamples records, not groups, so it is somewhat optimistic.

**Confusion matrices (cross-validation, fair subset; rows = expected, columns = predicted K / A / U)**

| | A Deployed | B Candidate |
|---|---|---|
| KNOWN | 38 / **52** / 1 | **73** / 16 / 2 |
| AMBIGUOUS | 31 / 48 / 9 | 23 / **52** / 13 |
| UNKNOWN | 5 / 11 / 81 | 3 / 8 / **86** |

The deployed model's dominant error (KNOWN → AMBIGUOUS, 52 of 91) falls to 16 of 91. AMBIGUOUS remains the hardest class for B (recall 0.59).

## 4. Generalisation

### 4.1 Three kinds of results, kept separate

| Evaluation | What it measures | A | B |
|---|---|---|---|
| **Grouped cross-validation** (§3) | New questions from categories seen in training | 0.602 macro-F1 (A never trained on this dataset, so this is cross-dataset for A) | **0.757** |
| **Category-held-out** (leave-one-category-out) | An entirely unseen question *type* | n/a (A is fixed) | **0.361 macro-F1, 0.404 accuracy** |
| **107-case benchmark** (§4.4) | **Not an independent test set** (see below) | 0.766 system accuracy | 0.860 system accuracy |

### 4.2 Category-held-out (B)

| | Value |
|---|---|
| Accuracy / macro-F1 / weighted-F1 | 0.404 / 0.361 / 0.369 |
| Recall K / A / U | **0.01** / 0.37 / 0.81 |
| Confusion (K / A / U rows) | KNOWN 1 / **86** / 6 · AMBIGUOUS 35 / 33 / 21 · UNKNOWN 2 / 17 / 79 |

When either KNOWN category is held out, B labels almost all of it AMBIGUOUS. The two KNOWN categories (determinate facts, concept definitions) look different in the feature space, so one doesn't transfer to the other. The UNKNOWN categories transfer well (recall 0.81). This collapse is not specific to B: every model tried in the earlier analysis scored about 0.38–0.39 here (`docs/CLASSIFIER_FAILURE_ANALYSIS.md`). It reflects the features and categories, not the retraining.

### 4.3 Performance by question category

| Category | Label | n (train pool) | B, grouped CV | B, category held out | A (fair, same records) |
|---|---|---|---|---|---|
| determinate_factual | KNOWN | 45 | **0.93** | 0.02 | 0.73 |
| concept_definition | KNOWN | 48 | **0.69** | 0.00 | 0.13 |
| genuine_ambiguity | AMBIGUOUS | 53 | 0.51 | 0.26 | **0.58** |
| subjective_questions | AMBIGUOUS | 36 | **0.72** | 0.53 | 0.50 |
| unknowable_future_events | UNKNOWN | 35 | **0.97** | 0.91 | 0.94 |
| private_inaccessible_info | UNKNOWN | 33 | **0.82** | 0.73 | 0.81 |
| false_premises | UNKNOWN | 30 | **0.83** | 0.77 | 0.73 |

**Categories with few training examples** (false_premises 30, private_inaccessible_info 33, unknowable_future_events 35, subjective_questions 36):
- **Under CV:** B is at or above A on all four.
- **Held out entirely:** the UNKNOWN categories still score 0.73–0.91, so the UNKNOWN signal (mainly negation and refusal wording) transfers across categories. subjective_questions drops to 0.53.
- **The clearest weakness:** genuine_ambiguity, the largest category, is where B is *below* A under CV (0.51 vs 0.58).

### 4.4 The 107-case benchmark (not an independent test set)

**This benchmark is not an independent test:**
- the safeguard/HITL layer was developed against its failures;
- 16 of its cases are, or paraphrase, development questions (excluded from training here, and listed in the "clean" rows below);
- 3 of its questions are in A's training data.

**How the replay works:**
- B's labels are produced offline from features recomputed from the **recorded** responses of the remediated run.
- They pass through the same override order as `experiments/ask.py` (criterion-ambiguity, then inaccessibility). Fast-path cases are unaffected by the classifier.
- **Validated:** for A, the replay reproduces the recorded raw labels (80/80) and the recorded final labels (80/80).
- **Not replayed:** HITL clarification decisions.

| Scope | n | A acc / macro-F1 | B acc / macro-F1 | Recall K / A / U (A → B) |
|---|---|---|---|---|
| Raw classifier, multi-response cases | 80 | 0.537 / 0.533 | **0.625 / 0.626** | 0.41→0.66 / 0.52→0.67 / **0.67→0.57** |
| Raw classifier, clean (overlaps removed) | 66 | 0.545 / 0.525 | **0.636 / 0.629** | 0.40→0.68 / 0.58→0.75 / 0.66→0.55 |
| System labels (classifier + safeguards), all | 107 | 0.766 / 0.738 | **0.860 / 0.840** | 0.65→0.79 / 0.62→0.76 / 1.00→1.00 |
| System labels, clean | 93 | 0.785 / 0.718 | **0.882 / 0.839** | 0.66→0.82 / 0.58→0.75 / 1.00→1.00 |

**System confusion (all 107; rows = expected K / A / U):**

| | KNOWN | AMBIGUOUS | UNKNOWN |
|---|---|---|---|
| A | 31 / 14 / 3 | 7 / 13 / 1 | 0 / 0 / 38 |
| B | **38 / 7** / 3 | 3 / **16** / 2 | 0 / 0 / 38 |

**System label changes A → B (18):**
- **14 fixed:** B02, B04, B05, B06, B07, E01, E12, H04 (clear questions no longer AMBIGUOUS); D01, D04, D06, D10, I03, I04 (ambiguous questions now AMBIGUOUS).
- **4 newly wrong:**
  - D03 "What is Mercury?" (→ KNOWN)
  - D07 "Where is Springfield?" (→ KNOWN)
  - D16 "How tall is he?" (→ UNKNOWN)
  - E02 "Who is Jordan Peterson?" (→ AMBIGUOUS)

**One concern:** B's *raw* UNKNOWN recall on the benchmark is lower (0.67 → 0.57). B labels several private-data requests AMBIGUOUS: A01, A05, A08, A10, A15, G07, G10, G12, H03, I05. The safeguard layer still makes all of them UNKNOWN (system recall 1.00), so B depends more on the safeguards for this class than A did.

## 5. Feature analysis

| Measure | A Deployed | B Candidate |
|---|---|---|
| Top feature by gain | negation_count **0.307** | negation_count **0.344** |
| Top feature by mean \|contribution\| | negation_count **0.548** | negation_count **0.415** |
| Next most used features (by contribution) | question_overlap 0.096, distinct_2gram 0.090, mean_similarity 0.089 | distinct_1gram 0.148, question_overlap 0.121, response_len_std 0.083, min_similarity 0.076 |
| Consistency-group share: gain / contribution | 0.225 / 0.139 | 0.372 / 0.115 |
| Linguistic-group share: gain / contribution | 0.775 / 0.861 | 0.628 / 0.885 |
| Grouped-CV permutation importance (macro-F1 drop) | — | negation_count **0.299**, distinct_1gram 0.113, question_overlap 0.034, min_similarity 0.016, all others ≤ 0.003 |
| Ablation (grouped CV): consistency features only | — | **0.442** macro-F1 |
| Ablation: linguistic features only | — | **0.761** macro-F1 (the full model is 0.757) |

**Answers to the investigation questions:**
- **Does `negation_count` still dominate?** **Yes.** It's the top feature in B by every measure. Removing its information costs about 0.30 macro-F1.
- **Do consistency features contribute meaningful predictive signal?** **Not measurably on this data.** `cluster_count` has 17% of B's gain, but permuting it changes held-out macro-F1 by about 0; its gain comes from splits on rare values. Consistency features alone reach 0.44, and adding them to the linguistic features changes macro-F1 from 0.761 to 0.757.
- **Do linguistic features dominate?** **Yes:** 86–89% of the contribution mass for both models.
- **Shortcuts?** Both models rely on surface-text proxies: negation words, lexical diversity, question overlap. Retraining fixed the *training-data* problem (B no longer labels unanimous explanations AMBIGUOUS), not the *feature* problem. False UNKNOWNs from content negation ("not a trade secret") remain likely; the benchmark's B14, G11 and G13 are still UNKNOWN for B.

## 6. Leakage analysis

| Check | Status |
|---|---|
| Benchmark questions or paraphrases in training | **Removed:** 9 groups / 20 records (exact + semantic ≥ 0.90, whole groups) |
| Records the deployed model may have seen, in A-vs-B comparisons | **Removed:** 4 records (fair subset n = 276) |
| CV group isolation | The project's `FiveFoldCrossValidator` (grouped by `question_group_id`) |
| Hyperparameter tuning | **None**, so no selection on any test data |
| Labels used in feature computation | No. Features were pre-extracted; `FeatureExtractor` takes no label |
| 107-case benchmark used for training or selection | **No.** It was used only for the clearly labelled, non-independent replay in §4.4 |
| Benchmark independence | **Not independent** (see §4.4); no independence claim is made |

## 7. Limitations

- **Small data:** 280 training records, and ~55 per CV fold. Fold SD is ~0.055 macro-F1. The bootstrap CI resamples records, not groups.
- **Within-distribution only:** the cross-validation figures (0.76–0.78) don't hold for unseen question types (0.36–0.40 category-held-out).
- **Feature shortcuts persist:** negation and lexical diversity, unchanged by design (schema frozen).
- **Benchmark caveats:** the replay is label-level only. HITL clarification decisions were not re-run, and the benchmark is not independent.
- **Overfitting:** B's training accuracy is 0.993 vs 0.76 in CV, with production hyperparameters held fixed on purpose (no tuning in this experiment).
- **Single configuration:** one generation model (`qwen2.5:3b`), one clustering threshold (0.80).

## 8. Recommendation

**Recommend promoting the candidate (B), gated on one live verification step.**

**Why:**
- **Every like-for-like comparison favours B:** grouped CV on identical unseen records (+0.155 macro-F1, 5/5 folds, p = 6 × 10⁻⁶), and the benchmark replay (system 0.766 → 0.860; 0.785 → 0.882 on uncontaminated cases).
- **It is the protocol's intended model:** same features, implementation and hyperparameters, trained on the documented production dataset. So it improves the system without changing the research method.
- **It removes the deployed model's dominant failure:** unanimous explanations labelled AMBIGUOUS.

**Conditions and caveats before and after promotion:**
1. **Gate:** run the live 107-case evaluation with B swapped in, so the HITL decisions are verified, not just labels. Report the 16 overlapping cases separately.
2. **Expect** 4 new label errors (D03, D07, D16, E02) and lower raw UNKNOWN recall. The safeguard layer must stay in place: B relies on it more than A for private-data requests.
3. **Don't claim** cross-category generalisation. Report category-held-out performance (~0.40) next to the CV figures.
4. **Keep the deployed artifact** as a rollback. The candidate is stored separately and isn't wired into the application.
5. **Follow-ups** (separate decisions, outside this experiment): a fresh, independent test set, and the feature-level questions (negation definition, clustering threshold) raised in `docs/CLASSIFIER_FAILURE_ANALYSIS.md`.
