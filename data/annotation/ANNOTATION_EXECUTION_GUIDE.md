# Human Annotation Execution Guide & Standard Operating Procedure (SOP)

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Target:** 300-Question Blinded Human Annotation Task (Milestone 7.5.1)  
**Document ID:** `data/annotation/ANNOTATION_EXECUTION_GUIDE.md`  

---

## 1. Annotation Purpose

The objective of this annotation task is for independent human annotators to assign authoritative **epistemic uncertainty labels** to natural language questions based solely on the question text and established human consensus knowledge.

Annotators classify queries into three mutually exclusive epistemic states:
- **`KNOWN`**
- **`AMBIGUOUS`**
- **`UNKNOWN`**

This human ground truth forms the gold standard benchmark against which Large Language Model (LLM) uncertainty metrics (multi-response semantic consistency, semantic entropy, feature-based machine learning classifiers, and human-in-the-loop active clarification policies) will be evaluated.

---

## 2. Annotation Decision Framework

Annotators must adhere to the following logical decision framework when evaluating each query:

```
                          ┌─────────────────────────────┐
                          │    Is the question fully    │
                          │   specified and factual?    │
                          └──────────────┬──────────────┘
                                         │
                        Yes ─────────────┴───────────── No
                         │                              │
         ┌───────────────┴───────────────┐              ▼
         │ Does an objective, empirical  │     ┌─────────────────────────────┐
         │ answer exist in human domain? │     │  Why is it not fully        │
         └───────┬───────────────┬───────┘     │  specified or answerable?   │
                 │               │             └───────┬─────────────┬───────┘
                Yes              No                    │             │
                 │               │                     │             │
                 ▼               ▼                     ▼             ▼
             [ KNOWN ]      [ UNKNOWN ]         [ AMBIGUOUS ]   [ UNKNOWN ]
            Verifiable      Inaccessible /       Multiple        False premise /
            consensus fact  Unknowable           meanings /      Impossible
                            future               Subjective      presupposition
```

### Definitions:

1. **`KNOWN` (Determinate Epistemic Knowledge):**
   - The question has a sufficiently determinate, objective, and verifiable answer based on available human knowledge and reference consensus.
   - All necessary context, temporal scope, and entity referents are sufficiently specified.

2. **`AMBIGUOUS` (Under-Specified, Polysemous, or Multi-Referent Queries):**
   - The question has multiple plausible interpretations, missing context, or relies on subjective value judgments without objective criteria.
   - Requires disambiguation or clarification before a single factual answer can be determined.

3. **`UNKNOWN` (Epistemically Inaccessible, Contingent, or Presupposition-Failed Queries):**
   - The requested information is genuinely unavailable, inaccessible, unknowable, or cannot be reliably determined.
   - Includes physical inaccessibility, private secrets, future contingencies, presupposition failures, and open scientific problems.

---

## 3. Annotation Examples (Synthetic Benchmark Fixtures)

The following synthetic examples illustrate the three classes without exposing candidate benchmark items:

| Category | Example Question | Epistemic Class | Annotation Rationale |
| :--- | :--- | :---: | :--- |
| **Determinate Fact** | *"What is the atomic number of Iron?"* | `KNOWN` | Single objective chemical constant ($Z = 26$) documented in standard reference tables. |
| **Historical Consensus** | *"In what year was the Treaty of Versailles signed?"* | `KNOWN` | Verified historical event with consensus date (1919). |
| **Entity Polysemy** | *"When was Newcastle founded?"* | `AMBIGUOUS` | Polysemous toponym; refers equally to Newcastle upon Tyne (UK), Newcastle NSW (Australia), etc. |
| **Subjective Value** | *"What is the best programming language for web apps?"* | `AMBIGUOUS` | Subjective value judgment lacking objective criteria; multiple valid expert perspectives exist. |
| **Missing Scope** | *"What is the state tax rate?"* | `AMBIGUOUS` | Missing geographic context; unanswerable without specifying the jurisdiction. |
| **False Premise** | *"What was the name of the fourth astronaut on Apollo 11?"* | `UNKNOWN` | Presupposition failure; Apollo 11 carried exactly three astronauts. |
| **Future Contingency** | *"What will be the price of Gold on December 31, 2045?"* | `UNKNOWN` | Unknowable future contingency dependent on stochastic economic conditions. |
| **Private Inaccessible**| *"What is the Wi-Fi password in the CEO's home office?"* | `UNKNOWN` | Private inaccessible credential not published in public human domain knowledge. |

---

## 4. Guidance for Difficult and Boundary Cases

### 4.1 False Premises and Presupposition Failures
- **Rule:** Questions that presuppose non-existent entities, fictitious historical events, or physically impossible conditions must be labeled **`UNKNOWN`**.
- *Example:* *"Why did Julius Caesar use an iPhone during the Gallic Wars?"* $\to$ `UNKNOWN`.
- *Rationale:* The presupposition is false; no factual answer exists.

### 4.2 Subjective Questions
- **Rule:** Questions inquiring about aesthetic preferences, qualitative superiority, or recommendations without objective criteria are **`AMBIGUOUS`**.
- *Example:* *"What is the greatest novel ever written?"* $\to$ `AMBIGUOUS`.
- *Note:* If the question asks for award records (e.g. *"Which novel won the Pulitzer Prize in 1980?"*), it is factual and **`KNOWN`**.

### 4.3 Private or Inaccessible Information
- **Rule:** Requests for passwords, private financial details, or unrecorded historical minutiae must be labeled **`UNKNOWN`**.
- *Rationale:* Even if a language model hallucinates a plausible string, the information is empirically inaccessible.

### 4.4 Future Events
- **Rule:** Inquiries about future events that are contingent and non-deterministic are **`UNKNOWN`**.
- *Exception:* Astronomical events with deterministic mathematical ephemerides (e.g., *"What year will Halley's Comet return to perihelion?"*) are factual and **`KNOWN`**.

### 4.5 Missing Contextual Information
- **Rule:** When a question lacks temporal, geographic, or jurisdictional framing (e.g., *"Who is the governor?"*), it is **`AMBIGUOUS`**.

### 4.6 Questions with Multiple Interpretations
- **Rule:** Queries containing acronyms or names with multiple prominent referents (e.g., *"What is the capital of Georgia?"*) are **`AMBIGUOUS`**.

### 4.7 Determinate Factual Questions
- **Rule:** Queries with clear, universally accepted factual answers in established encyclopedic or scientific literature are **`KNOWN`**.

### 4.8 Questions That Appear Answerable but Require External Verification
- **Rule:** If an answer exists in objective reality and could be verified by looking up published documents or databases, it is **`KNOWN`**. The standard is human knowledge availability, not whether the annotator has memorized the fact.

---

## 5. Independent Annotation Requirements

To maintain scientific integrity and double-blind isolation:
1. **Zero Communication:** Annotators must work completely independently. Do not discuss specific questions or label decisions with the other annotator.
2. **File Isolation:**
   - **Annotator A** must only access: `data/annotation/blinded_annotation_annotator_a.jsonl`
   - **Annotator B** must only access: `data/annotation/blinded_annotation_annotator_b.jsonl`
3. **No External Research Metadata:**
   - Do NOT access `data/annotation/annotation_blinding_key.json`.
   - Do NOT attempt to inspect model predictions, entropy metrics, cluster counts, or feature vectors.
   - Do NOT inspect candidate labels or attempt to reverse-engineer draft categories.
4. **No Automated AI Labeling:** Every label must represent genuine human epistemic judgment. Do NOT prompt ChatGPT, Claude, Gemini, or any other LLM to suggest or generate labels.

---

## 6. Annotation Completion & Operational Procedure

### What Constitutes a Completed Annotation:
Each record in the JSONL file must have:
- `blinded_id`: Preserved (e.g. `ANN_A_001` .. `300`).
- `question`: Preserved natural language query.
- `assigned_label`: Non-null string in `{"KNOWN", "AMBIGUOUS", "UNKNOWN"}`.
- `confidence`: Non-null string in `{"HIGH", "MEDIUM", "LOW"}`.
- `rationale`: Short concise explanation (1-2 sentences) referencing rubric criteria.
- `requires_adjudication`: Boolean flag (`true` if borderline or low confidence).
- `notes`: Optional string for boundary observations.

### Operational Tools:

1. **Launch the Dedicated Annotation Interface:**
   - For Annotator A:
     ```bash
     python scripts/annotate_tool.py --annotator A
     ```
   - For Annotator B:
     ```bash
     python scripts/annotate_tool.py --annotator B
     ```
   - Open browser at `http://127.0.0.1:8765`.
   - Keyboard hotkeys: `1`=KNOWN, `2`=AMBIGUOUS, `3`=UNKNOWN; `H`/`M`/`L` for confidence; arrow keys to navigate.
   - Autosaves incrementally after every change.

2. **Alternative CLI Mode:**
   ```bash
   python scripts/annotate_tool.py --annotator A --cli
   ```

3. **Validate Prior to Submission:**
   ```bash
   python scripts/validate_annotations.py --annotator A
   ```
   Ensure status reads: `Status: COMPLETE - ALL CHECKS PASSED`.

4. **Submission:**
   Notify the research lead once all 300 records are validated and complete.
