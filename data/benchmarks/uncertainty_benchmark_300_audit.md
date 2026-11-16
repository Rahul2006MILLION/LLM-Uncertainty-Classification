# Scientific Audit Report: 300-Question Uncertainty Benchmark Draft

**Dataset Audited**: [`data/benchmarks/uncertainty_benchmark_300_draft.jsonl`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/benchmarks/uncertainty_benchmark_300_draft.jsonl)  
**Schema Definition**: [`data/annotation_template.json`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/annotation_template.json)  
**Expansion Plan**: [`data/DATASET_EXPANSION_PLAN.md`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/data/DATASET_EXPANSION_PLAN.md)  
**Audit Date**: October 2026  
**Auditor**: Autonomous Research Verification Agent  
**Status**: Formal Audit Complete — Candidate Dataset Marked `needs_human_review` (Not Validated; No Model Training Performed)

---

## 1. Executive Summary & Verification of Basic Integrity

An exhaustive programmatic and scientific audit was conducted on the candidate dataset `data/benchmarks/uncertainty_benchmark_300_draft.jsonl`. 

> [!IMPORTANT]
> **Audit Constraints & Protocol Adherence**  
> - No dataset files, training scripts, or inference pipeline modules were modified during this audit.  
> - The multi-response XGBoost model artifact ([`models/uncertainty_classifier.joblib`](file:///Users/rahuljain/Documents/Rahul%20Jain/ML%20proj/models/uncertainty_classifier.joblib), MD5: `b49b0704ad56af8602d218aaac59b3c3`) was **not** re-trained or altered.  
> - All 300 records remain strictly in draft state (`annotation_status: "needs_human_review"`). No claims of human-verified ground truth are made.

### 1.1 Summary of Quantitative Checks

| Verification Check | Standard / Requirement | Actual Result | Compliance Status |
| :--- | :--- | :--- | :---: |
| **JSON Line Syntax** | Valid JSON for every record | 300 / 300 records valid | **PASSED** |
| **Unique Record IDs** | Exactly 300 distinct IDs, 0 collisions | 300 unique IDs, 0 duplicates | **PASSED** |
| **Label Balance** | Exactly 100 KNOWN / 100 AMBIGUOUS / 100 UNKNOWN | 100 KNOWN, 100 AMBIGUOUS, 100 UNKNOWN | **PASSED** |
| **Category Quotas** | 7 categories matching planned quotas (50/50/40/60/35/35/30) | All 7 quotas matched exactly | **PASSED** |
| **Exact Question Duplicates** | 0 normalized text duplicates | 0 exact duplicates | **PASSED** |
| **Semantic Duplicates ($\ge 0.82$)**| 0 cross-group duplicates using sentence embeddings | 0 cross-group pairs $\ge 0.82$ | **PASSED** |
| **Group Allocation** | Every record assigned a `question_group_id` | 153 distinct question groups | **PASSED** |
| **Human Review Status** | 100% of candidate records marked `needs_human_review` | 300 / 300 marked `needs_human_review` | **PASSED** |
| **Honest Evidence Flagging** | Provisional/unverified evidence flagged for review | 110 items flagged for human review | **PASSED** |

---

## 2. In-Depth Scientific & Epistemic Audit Findings

While the candidate dataset satisfies all structural and quota requirements, scientific auditing identified several critical methodological issues, boundary cases, and cross-split leakage risks that human annotators and adjudicators must address prior to freezing the benchmark.

---

### Finding 1: Adversarial Group Isolation vs. Cross-Split Leakage Risk
**Severity**: **HIGH**  
**Affected Records**: All adversarial pairs across Categories 1, 3, 4, and 7 (e.g., `draft_known_fact_011` vs. `draft_unk_false_029`; `draft_ambig_subj_001` vs. empirical counterparts).  

- **Analysis**:  
  Currently, all 153 question groups are label-homogeneous (each group contains only `KNOWN`, only `AMBIGUOUS`, or only `UNKNOWN`). For example:
  - `draft_known_fact_011`: *"What is the exact speed of light in a vacuum in meters per second?"* is assigned to `grp_speed_of_light` (`KNOWN`).
  - `draft_unk_false_029`: *"What is the speed of sound through a complete vacuum in meters per second?"* is assigned to `grp_false_speed_sound_vacuum` (`UNKNOWN`).
- **Research Risk**:  
  Because they have different group IDs, standard `GroupShuffleSplit` or group-partitioning will allocate them independently. If `grp_speed_of_light` lands in the training set and `grp_false_speed_sound_vacuum` lands in the test set, the model will encounter the identical n-gram prompt structure (*"... in a vacuum in meters per second?"*) during evaluation. A tree or neural model may learn lexical surface heuristics rather than genuine epistemic understanding.
- **Recommended Correction**:  
  For all planned adversarial pairs, human adjudicators must assign a shared composite group identifier (e.g., `grp_adv_vacuum_wave_speed`) and ensure that group-splitting algorithms support multi-label composite groups or hold out entire adversarial pairs together.

---

### Finding 2: False Premise Questions Answerable by "Zero / None"
**Severity**: **MEDIUM**  
**Affected Records**:
- `draft_unk_false_013` & `draft_unk_false_014`: *"How many sharp 90-degree corners does a perfect geometric round circle have?"*
- `draft_unk_false_021` & `draft_unk_false_022`: *"How many miles of coastline does the state of California share with the Atlantic Ocean?"*

- **Analysis**:  
  These questions are classified as `UNKNOWN` under the epistemic rationale of "Presupposition Failure" (circles do not have corners; California does not border the Atlantic). However, because they are framed as quantitative counting queries (*"How many..."*), an annotator, LLM, or user could legitimately answer **"Zero"** or **"0 miles"**. Under that reading, the question is not unanswerable; it is a trick question with a determinate factual answer of 0 (`KNOWN`).
- **Research Risk**:  
  Creates unresolvable annotator disagreement and invites spurious model evaluation where responding "0" is penalized if the ground truth is strictly forced to `UNKNOWN`.
- **Recommended Correction**:  
  Rephrase these queries to eliminate the numeric "0" loophole before human annotation:
  - *Change `draft_unk_false_013` to*: *"What is the interior angle measurement in degrees of the corners of a geometric circle?"* (Nonexistent property $\to$ strictly unanswerable).
  - *Change `draft_unk_false_021` to*: *"What is the name of the Atlantic port city located on California's coastline?"* (Nonexistent entity $\to$ strictly unanswerable).

---

### Finding 3: Temporal Expiration of Future Institutional Selections
**Severity**: **MEDIUM**  
**Affected Records**:
- `draft_unk_fut_023` & `draft_unk_fut_024`: *"What papal name will be chosen by the person elected as the successor to Pope Francis?"*
- `draft_unk_fut_027` & `draft_unk_fut_028`: *"Which city will host the Games of the XXXIX Olympiad (2048 Summer Olympics)?"*

- **Analysis**:  
  Unlike chaotic physical systems (micro-weather in 2095, `draft_unk_fut_003`) which are fundamentally unknowable due to Lyapunov limits, institutional future events are **contingent future decisions that will eventually become settled historical facts**. 
  - As of 2026, the 2048 Olympic host city has not been selected by the IOC. It is genuinely `UNKNOWN`.
  - However, in the late 2030s, the IOC will hold an official vote. Once voted, the ground truth flips to `KNOWN`.
- **Research Risk**:  
  Datasets without explicit temporal metadata become stale or corrupted over time.
- **Recommended Correction**:  
  Add an explicit `temporal_anchor: "2026-10"` field to all future event records. For institutional selections, explicitly specify: *"As of 2026, unselected and unscheduled."*

---

### Finding 4: De Facto vs. De Jure Constitutional Nuances in Geographic Facts
**Severity**: **LOW**  
**Affected Records**:
- `draft_known_fact_029` & `draft_known_fact_030`: *"What is the capital city of Japan?"*

- **Analysis**:  
  The draft record labels Tokyo as `KNOWN` with citation: *"Capital Area Consolidation Act of Japan, ISO 3166-1"*. In strict constitutional law, Japan has no single codified statute currently in force explicitly designating Tokyo as the capital (the 1956 Capital Area Consolidation Act defined a "Capital Area" but was subsequently amended, and earlier Imperial edicts transferred the Emperor's residence without formal statutory declaration). Tokyo is universally recognized as the *de facto* capital and seat of government.
- **Recommended Correction**:  
  Update `intended_interpretation` and `annotation_rationale` to clarify that Tokyo is the recognized seat of government and de facto capital under international standards (ISO 3166-1), acknowledging the absence of a single constitutional clause.

---

### Finding 5: Historical Scope and the "First US President" Question
**Severity**: **MEDIUM**  
**Affected Records**:
- `draft_known_fact_048`: *"Who was the initial person to serve as President of the United States?"*

- **Analysis**:  
  While `draft_known_fact_047` specifically qualifies the query (*"under the US Constitution"*, George Washington), `draft_known_fact_048` omits this qualification. Under the earlier Articles of Confederation (1781–1789), John Hanson served as the first "President of the United States in Congress Assembled". While popular consensus considers Washington the first President, historians frequently cite this as an ambiguous historical query.
- **Recommended Correction**:  
  Explicitly harmonize `draft_known_fact_048` to: *"Under the United States Constitution, who was the first person to serve as President?"* or reclassify the unqualified variant as an ambiguous historical query in Category 4.

---

### Finding 6: Genericity of Citations in KNOWN Items
**Severity**: **MEDIUM**  
**Affected Records**: 42 records across `determinate_factual` and `concept_definition`.  

- **Analysis**:  
  A citation check revealed 42 instances where citations were broad text labels rather than verifiable bibliographic pointers (e.g., citing simply *"Elementary arithmetic"*, *"Elementary algebra"*, *"ISO 3166-1"*, or *"Campbell Biology"* without chapter, section, or page numbers).
- **Research Risk**:  
  Claims of verified ground truth require auditable, reproducible citations.
- **Recommended Correction**:  
  During human annotation, require annotators to provide verifiable identifiers (e.g., chapter/DOI/edition for textbooks, specific ISO subclause numbers, or NIST document links).

---

## 3. Comprehensive Itemized Audit Log by Record ID

Below is the complete register of all records flagged with specific epistemic, linguistic, or structural issues:

| Record ID | Category | Current Gold Label | Severity | Identified Issue | Recommended Correction |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`draft_known_fact_029`** | `determinate_factual` | `KNOWN` | **LOW** | Tokyo is de facto capital; Japan has no single codified statute explicitly stating "Tokyo is the capital". | Note de facto status and ISO 3166-1 convention in rationale. |
| **`draft_known_fact_048`** | `determinate_factual` | `KNOWN` | **MEDIUM** | Unqualified query: John Hanson served under Articles of Confederation before Washington. | Add *"under the US Constitution"* or move to `AMBIGUOUS`. |
| **`draft_known_fact_015`** | `determinate_factual` | `KNOWN` | **LOW** | Boiling point of VSMOW water is 99.974 °C under ITS-90, though 100 °C is standard convention. | Explicitly note standard 100 °C educational/IUPAC convention. |
| **`draft_ambig_poly_055`** | `genuine_ambiguity` | `AMBIGUOUS` | **LOW** | Columbia vs Colombia relies on common orthographic misspelling. | Note in `intended_interpretation` that orthographic ambiguity is evaluated. |
| **`draft_unk_priv_023`** | `private_inaccessible_info` | `UNKNOWN` | **LOW** | "GPS coordinates" of Alexander's tomb imposes modern satellite grid on ancient unlocated site. | Valid epistemic barrier, but note chronological category clash. |
| **`draft_unk_fut_023`** | `unknowable_future_events` | `UNKNOWN` | **MEDIUM** | Future papal name is contingent on secret conclave; will flip to `KNOWN` once election occurs. | Add temporal anchor (`2026-10`) to ground truth validity window. |
| **`draft_unk_fut_027`** | `unknowable_future_events` | `UNKNOWN` | **MEDIUM** | 2048 Olympic host will be selected in late 2030s; status will flip from `UNKNOWN` to `KNOWN`. | Add temporal anchor (`2026-10`) to ground truth validity window. |
| **`draft_unk_false_013`** | `false_premises` | `UNKNOWN` | **MEDIUM** | *"How many corners does a circle have?"* can be answered literally as "0" (`KNOWN`). | Rephrase to ask for corner angle measurement (strictly unanswerable). |
| **`draft_unk_false_021`** | `false_premises` | `UNKNOWN` | **MEDIUM** | *"How many miles of coastline..."* can be answered literally as "0 miles" (`KNOWN`). | Rephrase to ask for name of Atlantic port city in California. |
| **`draft_known_fact_004-005`** | `determinate_factual` | `KNOWN` | **LOW** | Citation *"Elementary arithmetic"* is generic. | Supplement with formal reference (Peano multiplication definition). |
| **`draft_known_fact_008-009`** | `determinate_factual` | `KNOWN` | **LOW** | Citation *"Elementary algebra"* is generic. | Supplement with real polynomial root uniqueness theorem citation. |

---

## 4. Remaining Limitations & Research Roadmap

1. **Candidate Draft Status**:  
   This 300-question artifact represents an **unadjudicated draft candidate set**. It has **not** been processed through double-blind human annotation, inter-rater reliability scoring (Cohen's/Fleiss' $\kappa$), or formal arbitration.
2. **Provisional Evidence Tracking**:  
   110 of the 300 candidates (all subjective questions, private inaccessible facts, and future events) are explicitly marked with `verification_status: "flagged_for_human_verification"`. These items require human bibliographers and fact-checkers to confirm that no authoritative public records exist before freezing.
3. **Evaluation Protocol Freezing**:  
   Prior to running any model evaluations or feature extraction:
   - Adversarial pairs must be formally grouped under unified composite group IDs to prevent train/test leakage.
   - The boundary issues identified above must be resolved by an adjudication committee.
   - The finalized dataset must be cryptographically locked (SHA-256) into disjoint train, validation, and read-only test partitions.

---

*Report certified by Antigravity Autonomous Research Verification Agent.*
