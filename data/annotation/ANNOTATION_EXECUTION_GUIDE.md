# Human Annotation Execution Guide & Standard Operating Procedure (SOP)

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Target:** 300-Question Double-Blind Human Annotation Task (Milestone 7.5.1)  
**File Location:** `data/annotation/ANNOTATION_EXECUTION_GUIDE.md`  

---

## 1. Annotation Objective & Research Scope

The scientific purpose of this annotation task is to establish an authoritative, unbiased, and empirically validated **gold standard** for 300 natural language questions. 

In this research project, questions are evaluated on their fundamental **epistemic uncertainty** under human consensus knowledge. We categorize questions into three mutually exclusive epistemic states:
- **`KNOWN`**
- **`AMBIGUOUS`**
- **`UNKNOWN`**

This gold-standard dataset will serve as the ground truth against which Large Language Model (LLM) consistency metrics, semantic entropy, feature-based machine learning classifiers, and Human-in-the-Loop (HITL) active clarification policies are benchmarked.

---

## 2. Epistemic Label Definitions

Each question must be assigned exactly one of the three epistemic uncertainty classes:

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

1. **`KNOWN` (Determinate Epistemic Knowledge):**
   - A single, objective, verifiable factual answer exists in established human knowledge or consensus reference sources.
   - The query specifies all necessary context, entities, and scope needed for a definitive answer.

2. **`AMBIGUOUS` (Under-Specified, Polysemous, or Multi-Referent Queries):**
   - The query permits multiple legitimate, competing factual interpretations, distinct entity referents, or subjective evaluation metrics.
   - Without additional context or clarification, no single factual response is uniquely correct.

3. **`UNKNOWN` (Epistemically Inaccessible, Contingent, or Presupposition-Failed Queries):**
   - The query is fundamentally unanswerable because the empirical information is physically inaccessible, relates to contingent future events, involves private secrets, or presupposes facts/entities that do not exist (presupposition failure).

---

## 3. Step-by-Step Decision-Making Procedure

For every question presented:
1. **Analyze Query Pragmatics:** Read the question carefully. Identify the core entity, the requested property/relation, and any temporal or geographical scope.
2. **Evaluate Presupposition Validity:** Does the question presuppose an entity or condition that is false or fictitious? If yes $\to$ assign `UNKNOWN`.
3. **Assess Information Inaccessibility:** Does the question ask for private credentials, real-time unrecorded data, or future events? If yes $\to$ assign `UNKNOWN`.
4. **Inspect Referents & Subjectivity:** Does the question refer to an entity with multiple common interpretations without disambiguating context? Or does it ask for subjective preferences ("best", "favorite") without objective criteria? If yes $\to$ assign `AMBIGUOUS`.
5. **Verify Single Factual Consensus:** If the question has a clear, established, consensus answer in encyclopedic or academic knowledge $\to$ assign `KNOWN`.
6. **Assign Confidence & Rationale:** Select your confidence level (`HIGH`, `MEDIUM`, `LOW`) and briefly explain the rationale by referencing the rubric criteria.

---

## 4. `KNOWN` Criteria & Archetypes

A query is `KNOWN` if and only if it satisfies all of the following:
- Clear, unambiguous entity references.
- Objective truth criteria independent of personal taste.
- Documented in authoritative reference works (scientific literature, standard encyclopedias, official records).

### Archetypes:
- **Scientific & Mathematical Facts:** *"What is the atomic number of Gold?"*, *"What is the derivative of $\sin(x)$?"*
- **Established Historical Events:** *"In what year was the Magna Carta signed?"*, *"Who was the first president of the United States?"*
- **Well-Defined Geographical Facts:** *"What is the official currency of Japan?"*, *"What is the capital of France?"*
- **Established Cultural & Linguistic Definitions:** *"What is the definition of photosynthesis?"*, *"Who authored Pride and Prejudice?"*

---

## 5. `AMBIGUOUS` Criteria & Archetypes

A query is `AMBIGUOUS` if any single answer would be incomplete or arbitrary without clarification:

### Archetypes:
- **Entity Polysemy / Name Clashes:** *"When was the University of Washington founded?"* vs. *"When was Washington founded?"* (George Washington? Washington State? Washington D.C.?).
- **Temporal / Geographic Context Absence:** *"Who is the current prime minister?"* (Which country?), *"What is the capital of Georgia?"* (US State or Eurasian country?).
- **Subjective Value Judgments:** *"What is the best movie of all time?"*, *"Which programming language is superior?"* (No single objective ground truth exists).
- **Scope Ambiguity:** *"How long does it take to get a passport?"* (Routine vs. expedited? US vs. UK?).

---

## 6. `UNKNOWN` Criteria & Archetypes

A query is `UNKNOWN` if no human or AI possesses or could possess the factual answer:

### Archetypes:
- **Presupposition Failures / Counterfactual Fallacies:**
  - *"What is the boiling point of liquid vacuum?"*
  - *"Who was the female US president elected in 1980?"* (No female US president existed in 1980).
  - *"What is the chemical formula of phlogiston gas?"*
- **Unknowable Future Contingencies:**
  - *"What will be the exact closing price of Apple stock on December 1, 2035?"*
  - *"Who will win the FIFA World Cup in 2042?"*
- **Private / Epistemically Inaccessible Data:**
  - *"What is the master PIN code for the server room of Acme Corp?"*
  - *"What exact words did Abraham Lincoln whisper to his wife at 7:15 AM on April 10, 1862?"*
- **Unsolved Scientific Open Problems:**
  - *"Is $P$ strictly equal to $NP$?"*
  - *"What is the exact mass of dark matter particles in the universe?"*

---

## 7. Handling False Premises and Presupposition Failures

> **CRITICAL RULE:** Questions with false premises MUST be classified as **`UNKNOWN`**, NOT `AMBIGUOUS` or `KNOWN`.

- If a user asks *"Why did Alexander the Great build the Eiffel Tower?"*, do NOT treat this as ambiguous. Alexander the Great did not build the Eiffel Tower; the premise is false.
- A question cannot have a factual affirmative answer when its existential presupposition fails. Label: **`UNKNOWN`**. Rationale: *"Presupposition failure; the presupposed entity/event never occurred."*

---

## 8. Handling Private and Inaccessible Information

- Requests for private keys, personal passwords, unrecorded historical minutiae, or unreleased internal memos must be classified as **`UNKNOWN`**.
- Even if an LLM generates a plausible-sounding string or hallucination, the information is epistemically unavailable. Label: **`UNKNOWN`**.

---

## 9. Handling Future Predictions

- Any query regarding future contingencies that cannot be computed from deterministic laws of nature (e.g. election outcomes, stock prices, weather on a distant future day) is fundamentally **`UNKNOWN`**.
- Deterministic astronomical events with established ephemerides (e.g., *"On what date will the next total solar eclipse occur in North America?"*) have deterministic answers and are **`KNOWN`**.

---

## 10. Handling Subjective Questions

- Questions asking for recommendations, aesthetic judgments, or qualitative superiority without explicit objective criteria (e.g., *"What is the greatest song ever recorded?"*) are **`AMBIGUOUS`**.
- If a subjective question asks for historical consensus or award records (e.g., *"Which film won the Academy Award for Best Picture in 1994?"*), it is factual and therefore **`KNOWN`**.

---

## 11. Confidence Scoring Guidelines

Record your certainty in the assigned label:
- **`HIGH`:** Clear-cut, canonical example meeting all explicit criteria of the rubric with zero doubt.
- **`MEDIUM`:** Reasonable interpretation based on rubric, but alternative interpretations or borderline boundary conditions exist.
- **`LOW`:** Highly difficult boundary query; difficult to disentangle between two classes.

---

## 12. When to Flag for Adjudication (`requires_adjudication`)

Check the **Flag for Adjudication** box (`requires_adjudication = true`) whenever:
1. You feel torn between two labels (e.g., whether a term is commonly understood by default or requires disambiguation).
2. The question relies on specialized technical terminology whose ambiguity depends on specific domain context.
3. You assign `confidence = "LOW"`.

Flagging an item does not invalidate your choice; it ensures the item will be independently audited during the adjudication phase.

---

## 13. How to Record Rationales

Brief, concise rationales (1-2 sentences) are required for every question.
- **Good KNOWN rationale:** *"Single determinate factual answer established in standard astronomical reference data."*
- **Good AMBIGUOUS rationale:** *"Query is polysemous; refers equally to the musical instrument or the computing framework without context."*
- **Good UNKNOWN rationale:** *"Presupposition failure; assumes an emperor of Antarctica which does not exist."*

---

## 14. What Annotators Must NOT Access

To protect the double-blind integrity of the research:
- ❌ **DO NOT** consult or speak with the other annotator about specific questions.
- ❌ **DO NOT** search for or attempt to view candidate labels or metadata.
- ❌ **DO NOT** inspect model outputs, LLM responses, or entropy measurements.
- ❌ **DO NOT** inspect or access `data/annotation/annotation_blinding_key.json`.
- ❌ **DO NOT** use AI assistants (ChatGPT, Claude, Gemini, etc.) to generate labels. Every label must reflect human judgment.

---

## 15. Operational Annotation Tools & Submission Procedure

### Using the Dedicated Local Web Interface (Recommended)

1. **Launch the interface for your assigned sheet:**
   - **Annotator A:**
     ```bash
     python scripts/annotate_tool.py data/annotation/blinded_annotation_annotator_a.jsonl
     ```
   - **Annotator B:**
     ```bash
     python scripts/annotate_tool.py data/annotation/blinded_annotation_annotator_b.jsonl
     ```
2. Open your web browser at: `http://127.0.0.1:8765`.
3. Use the keyboard shortcuts (`1`, `2`, `3` for labels; `H`, `M`, `L` for confidence; `Arrow Keys` for navigation).
4. Progress is **automatically saved** to your JSONL file after every change.
5. You may stop and resume at any time.

### Using the Interactive CLI Mode (Alternative)

If you prefer terminal-only work:
```bash
python scripts/annotate_tool.py data/annotation/blinded_annotation_annotator_a.jsonl --cli
```

### Pre-Submission Verification

Before submitting your completed file, run the validation tool:
```bash
python scripts/validate_annotations.py data/annotation/blinded_annotation_annotator_a.jsonl --expected-annotator A
```
Confirm that:
- Total records: **300**
- Fully valid records: **300**
- Unannotated records: **0**
- Validation status: **PASSED**

### Submission Procedure

Once verified:
1. Submit your completed file (`blinded_annotation_annotator_a.jsonl` or `b.jsonl`) to the research lead.
2. The research lead will record the file's SHA-256 hash.
3. Once both files are collected, the blinding key will be unsealed, agreement calculated, and the adjudication docket generated.
