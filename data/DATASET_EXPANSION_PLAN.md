# Benchmark Expansion and Epistemic Annotation Plan: 300 Questions

**Project**: LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Target Benchmark Size**: 300 Balanced Ground-Truth Questions  
**Status**: Plan & Annotation Template (Evaluation Protocol Frozen; No Synthetic Self-Labeling)  
**Date**: October 2026  

---

## 1. Inspection of Existing Data Schema and Limitations

### 1.1 Existing Dataset Artifacts
- **`data/schema.py` (`QuestionRecord`)**:
  - Fields: `id`, `question`, `label` (`KNOWN`, `AMBIGUOUS`, `UNKNOWN`), `domain`, `subcategory`, `source`, `license`, `ground_truth_answers`, `interpretations`, `annotation_type`, `annotation_rationale`, `metadata`.
  - *Limitation*: Does not explicitly track multi-annotator review, individual reviewer disagreements, or explicit question-group links across adversarial perturbations.
- **`data/raw/generated_responses.jsonl`**:
  - Fields: `id`, `question`, `ground_truth_label`, `domain`, `source`, `model`, `temperature`, `top_p`, `n_responses` (10), `responses` (list of text, elapsed time, eval tokens).
  - *Current Size*: 68 records (representing 67 unique questions across 42 groups).
- **`data/features.jsonl`**:
  - Extracted 16 multi-response features (`semantic_entropy`, `normalized_entropy`, `dominant_agreement`, `second_agreement`, `agreement_margin`, `cluster_count`, `cluster_ratio`, `mean_pairwise_similarity`, `min_pairwise_similarity`, `response_len_mean`, `response_len_std`, `distinct_1gram_ratio`, `distinct_2gram_ratio`, `question_overlap_mean`, `hedging_score`, `negation_count`).
  - *Limitation*: All records were constructed assuming $N=10$ multi-response clustering.

---

## 2. Expanded Annotation Schema

To support defensible scientific evaluation, a new schema template is established at [`data/annotation_template.json`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/annotation_template.json). Every entry in the 300-question benchmark must provide the following eight required fields:

| Field Name | Type | Description | Epistemic Function |
| :--- | :--- | :--- | :--- |
| **`question`** | `string` | The exact natural language query string presented to the system. | Input stimulus. |
| **`intended_interpretation`** | `string` | The contextual reading under which answerability is analyzed. | Disambiguates whether a question is tested under its primary common reading or an adversarial nuance. |
| **`gold_label`** | `enum` | Must be strictly one of `KNOWN`, `AMBIGUOUS`, or `UNKNOWN`. | Evaluative ground truth. |
| **`rationale`** | `string` | Rigorous justification for why the question belongs to this class. | Auditable human explanation. |
| **`answerability_evidence`** | `object` | Authoritative citations for KNOWN, enumerated valid interpretations for AMBIGUOUS, or proof of epistemic barrier for UNKNOWN. | Prevents unsupported or speculative label assignments. |
| **`source`** | `string` | Origin dataset, encyclopedia standard, scientific law, or curated adversarial design. | Provenance tracking. |
| **`question_group_id`** | `string` | Cluster identifier linking paraphrases, domain entity variations, and adversarial pairs. | **Prevents data leakage**; enforces disjoint group splits. |
| **`reviewer_agreement`** | `object` | Records number of annotators, independent labels, Cohen's/Fleiss' $\kappa$, adjudicator ID, and resolution rationale. | Documents genuine human consensus and disagreement. |

---

## 3. Benchmark Category Quotas (Total: 300 Questions)

The proposed benchmark is balanced across the 3 epistemic classes (100 `KNOWN`, 100 `AMBIGUOUS`, 100 `UNKNOWN`) across 7 distinct operational categories:

```
Total Benchmark: 300 Questions
├── KNOWN (100 Questions)
│   ├── Category 1: Determinate Factual Questions (Quota: 50)
│   └── Category 2: Established Concept Definitions (Quota: 50)
├── AMBIGUOUS (100 Questions)
│   ├── Category 3: Subjective Questions without Objective Metrics (Quota: 40)
│   └── Category 4: Genuine Entity Polysemy & Linguistic Ambiguity (Quota: 60)
└── UNKNOWN (100 Questions)
    ├── Category 5: Private or Inaccessible Information (Quota: 35)
    ├── Category 6: Unknowable Future Events (Quota: 35)
    └── Category 7: False Premises & Presupposition Failures (Quota: 30)
```

### Detailed Category Descriptions:

1. **Category 1: Determinate Factual Questions (Quota: 50, Target: `KNOWN`)**
   - *Scope*: Closed-domain arithmetic, geographic consensus facts, historical events with settled consensus, international standards (SI units, ISO codes), chemistry/physics constants.
   - *Criteria*: The answer is singular, verifiable against authoritative references, and invariant across reputable sources.

2. **Category 2: Established Concept Definitions (Quota: 50, Target: `KNOWN`)**
   - *Scope*: Standard academic disciplines, medical conditions, technical protocols, legal terms of art, standard algorithms.
   - *Criteria*: The definition has universal or textbook consensus, even if phrasing varies slightly across sources.

3. **Category 3: Subjective Questions without Single Objective Metric (Quota: 40, Target: `AMBIGUOUS`)**
   - *Scope*: Unconstrained superlatives ("greatest country", "best programming language", "most beautiful painting", "most successful person").
   - *Criteria*: The query demands a value judgment or normative preference where no singular objective metric is specified in the prompt.

4. **Category 4: Genuine Entity Polysemy & Linguistic Ambiguity (Quota: 60, Target: `AMBIGUOUS`)**
   - *Scope*: Ambiguous entity names ("capital of Georgia", "president of Lincoln", "Who played Spider-Man?"), temporal ambiguity without reference frame ("Who is the prime minister?"), polysemous terminology.
   - *Criteria*: Multiple prominent, non-conflicting answers exist depending on unstated context, and all interpretations are legitimate.

5. **Category 5: Private or Inaccessible Information (Quota: 35, Target: `UNKNOWN`)**
   - *Scope*: Epistemic opacity, unrecorded historical minutiae ("What did Socrates whisper as he died?"), private personal credentials (safe codes, private keys, passwords), unrecorded conversations, internal mental states.
   - *Criteria*: The fact exists or existed in reality, but no record or trace is accessible in public human knowledge.

6. **Category 6: Unknowable Future Events (Quota: 35, Target: `UNKNOWN`)**
   - *Scope*: Distant chaotic systems (micro-weather in 2095), stochastic financial outcomes (closing stock price next decade), unheld lottery drawings, unpredictable political elections.
   - *Criteria*: The event has not occurred; physical laws or informational barriers make exact determination fundamentally impossible.

7. **Category 7: False Premises & Presupposition Failures (Quota: 30, Target: `UNKNOWN`)**
   - *Scope*: Queries asking for properties of nonexistent entities ("King of France in 2024", "melting point of dry water", "capital of Atlantis", "diameter of the third moon of Mars").
   - *Criteria*: The question presumes an existential premise that is objectively false. The requested value does not exist.

---

## 4. Question Groups, Paraphrases, and Adversarial Perturbations

### 4.1 Group Architecture
To evaluate robustness rather than surface memorization, the benchmark is organized into **~75 Question Groups** (averaging 4 questions per group):

1. **Canonical Formulation**: Natural baseline phrasing.
2. **Syntactic Paraphrase**: Structural alteration (passive/active, inverted clause, formal vs. colloquial) without semantic change.
3. **Lexical Paraphrase**: Synonym substitution and alternative entity mentions.
4. **Adversarial Boundary Perturbation**: A targeted modification designed to flip or test the class boundary.

### 4.2 Non-Trivial Adversarial Examples (No Superficial Templating)

| Group ID | Base Question & Label | Adversarial Perturbation | New Label | Epistemic Mechanism |
| :--- | :--- | :--- | :---: | :--- |
| `grp_prog_lang_01` | "What is the best programming language?" (`AMBIGUOUS`) | "Which programming language has the highest share of questions on Stack Overflow in 2023?" | **`KNOWN`** | Adding an explicit, measurable empirical metric transforms an unconstrained subjective query into a determinate fact. |
| `grp_french_head_01` | "What was the name of the King of France in 2024?" (`UNKNOWN`) | "Who was the official head of state of France in 2024?" | **`KNOWN`** | Removing the false monarchical presupposition reveals the valid constitutional office (President Emmanuel Macron). |
| `grp_georgia_cap_01` | "What is the capital of Georgia?" (`AMBIGUOUS`) | "What is the capital of the sovereign nation of Georgia located in the Caucasus region?" | **`KNOWN`** | Disambiguating the entity to the sovereign nation eliminates geographical polysemy $\to$ Tbilisi. |
| `grp_moon_diameter_01`| "What is the diameter of Phobos, the moon of Mars?" (`KNOWN`) | "What is the diameter of the third natural moon of Mars?" | **`UNKNOWN`** | Mars has only two moons (Phobos and Deimos). Adding "third" introduces a false existential presupposition. |
| `grp_socrates_01` | "Where was Socrates put on trial in 399 BC?" (`KNOWN`) | "What were the exact unrecorded thoughts of Socrates during the final ten minutes before drinking hemlock?" | **`UNKNOWN`** | Shifting from public historical trial records to unrecorded private mental cognition introduces an impenetrable epistemic barrier. |

---

## 5. Strict Labeling Rules and Difficult Boundary Decisions

### 5.1 The Core Distinction: `AMBIGUOUS` vs. `UNKNOWN`

This boundary is the most frequent source of annotator confusion. Annotators must follow this formal operational test:

```
                                  [Query Analyzed]
                                         │
                 Does any valid, legitimate answer exist in reality
                        or human knowledge under any reading?
                                    /         \
                                 [YES]        [NO]
                                  /             \
            Is there a single unique           Is it because of:
            answer across all valid            - Private/unrecorded state?
            readings/criteria?                 - Chaotic future contingency?
                 /        \                    - False existential premise?
              [YES]       [NO]                           │
               /            \                            ▼
           ▼                 ▼                       UNKNOWN
         KNOWN           AMBIGUOUS
                     (Multiple valid answers
                     can be listed/clarified)
```

- **Rule 1 (AMBIGUOUS requires plural answerability)**: A question is `AMBIGUOUS` **if and only if** multiple distinct, plausible, knowable answers exist, and the query fails to specify criteria or context to select among them. A well-informed response could answer by saying: *"If you mean X, the answer is A; if you mean Y, the answer is B."*
- **Rule 2 (UNKNOWN denotes epistemic barrier or nonexistence)**: A question is `UNKNOWN` **if and only if** no determinate answer exists, cannot be verified by any observer, or the premise is false. A well-informed response cannot provide a list of answers; it must declare that the answer cannot be known or that the presupposition fails.

### 5.2 Boundary Case Reference Table

| Query | Ambiguous? | Unknown? | Correct Gold Label | Adjudication Rationale |
| :--- | :---: | :---: | :---: | :--- |
| *"Who is the president?"* | Yes | No | **`AMBIGUOUS`** | Many countries, corporations, and universities have presidents. Plural legitimate answers exist depending on unstated entity scoping. |
| *"Who was the president of the United States on February 30, 2020?"* | No | Yes | **`UNKNOWN`** | February 30 does not exist in any calendar. Presupposition failure $\to$ `UNKNOWN`. |
| *"What is the greatest invention in human history?"* | Yes | No | **`AMBIGUOUS`** | Subjective superlative without criteria. Plural legitimate historical candidates exist (wheel, printing press, antibiotics). |
| *"What was the exact temperature at the center of the Earth on July 4, 1776 at 12:00 PM?"* | No | Yes | **`UNKNOWN`** | Core temperature has physical existence, but direct measurement at that exact timestamp is epistemically inaccessible. |
| *"What is the population of Springfield?"* | Yes | No | **`AMBIGUOUS`** | Over 30 recognized cities named Springfield exist in the United States alone. |
| *"What will the winning Powerball lottery numbers be next Wednesday?"* | No | Yes | **`UNKNOWN`** | Random future stochastic event. No current information determines the outcome. |

---

## 6. Disjoint Group Partitioning & Test Set Freezing Protocol

### 6.1 Group-Aware Splitting Rules
To prevent data contamination, splitting must occur strictly at the **`question_group_id` level**, never at the individual question level:
1. All paraphrases, syntactic variants, and adversarial pairs within a `question_group_id` reside in **the same split** (100% group isolation).
2. The split ratio across groups is **60% Train, 20% Validation, 20% Test** (approximately 45 groups Train, 15 groups Val, 15 groups Test).
3. The group allocation is stratified such that each split contains an exact balanced distribution of the 7 subcategories.

### 6.2 Test Set Freezing Protocol
1. **Cryptographic Checksumming**: Once the 300 questions are annotated and adjudicated, compute SHA-256 hashes of `train_benchmark.jsonl`, `val_benchmark.jsonl`, and `test_benchmark.jsonl`.
2. **Read-Only Vault**: The `test_benchmark.jsonl` file is written to a designated evaluation directory and locked with read-only filesystem permissions (`chmod 444`).
3. **No Iterative Peeking**: The test set must NOT be inspected during prompt engineering, feature selection, classifier hyperparameter tuning, or threshold adjustment. It is evaluated exactly once when reporting final benchmark metrics.

---

## 7. Human Review Protocol & Recording Disagreements

### 7.1 Multi-Annotator Requirements
- Every question must be independently annotated by at least **two qualified human reviewers**.
- Reviewers are provided the raw question string and the intended interpretation, but are **blinded to model predictions**.
- **Model predictions must NEVER be used as default gold labels.**

### 7.2 Inter-Annotator Agreement & Adjudication
- Annotators record their independent labels and notes in the `reviewer_agreement.individual_labels` list.
- If both annotators agree: `inter_annotator_agreement = "unanimous"`.
- If annotators disagree: `inter_annotator_agreement = "disputed_adjudicated"`. A third senior reviewer (Adjudicator) examines the rationale and answerability evidence, assigns the final gold label, and documents the scientific reasoning in `adjudication_rationale`.
- Inter-annotator reliability is quantified using Cohen's $\kappa$ (pairwise) and Fleiss' $\kappa$ (across all annotators). Questions with persistent annotator uncertainty are revised or discarded before freezing.

---

## 8. Summary of Created Artifacts

1. **[`data/annotation_template.json`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/annotation_template.json)**:
   - JSON Schema draft-07 compliant specification.
   - Exact definitions for all 8 required fields.
   - 6 concrete, fully annotated exemplar records covering all key categories and difficult boundary adjudications.
   - Blank annotation template entry ready for human annotator ingestion.
2. **[`data/DATASET_EXPANSION_PLAN.md`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/DATASET_EXPANSION_PLAN.md)**:
   - Complete scientific plan documenting current limitations, schema expansion, balanced quotas, adversarial perturbations, operational decision trees, group-disjoint splitting, and review protocols.
