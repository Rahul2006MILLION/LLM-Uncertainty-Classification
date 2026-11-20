# Human Annotator Guide & Operational Instructions

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Target:** 300-Question Blinded Human Annotation Task (Milestone 7)  
**File:** `data/annotation/ANNOTATOR_INSTRUCTIONS.md`  

---

## 1. Introduction & Mission

Thank you for participating as an expert human annotator in this research project. Your task is to independently classify **300 natural language questions** into three fundamental epistemic uncertainty classes:

1. **`KNOWN`** — A single, objective, verifiable factual answer exists in authoritative human knowledge.
2. **`AMBIGUOUS`** — The query has multiple legitimate, competing interpretations, referents, or subjective evaluation metrics.
3. **`UNKNOWN`** — The query is fundamentally unanswerable because the information is inaccessible, future-contingent, non-existent, or scientifically undetermined.

---

## 2. Strict Blinding Protocol

To ensure absolute scientific rigor and avoid confirmation bias:
1. **No External Labels:** You have been provided a blinded sheet where all existing candidate labels, question group IDs, and category indicators have been removed.
2. **Independent Classification:** Do NOT collaborate, consult, or share responses with the other annotator during the annotation phase.
3. **No Model Outputs:** You will evaluate only the raw natural language question. You must NOT inspect model outputs, LLM responses, or entropy measurements.
4. **Isolated File Assignment:**
   - **Annotator A:** Works exclusively on `data/annotation/blinded_annotation_annotator_a.jsonl`.
   - **Annotator B:** Works exclusively on `data/annotation/blinded_annotation_annotator_b.jsonl`.

*(Note: The question sequences for Annotator A and Annotator B have been independently randomized using distinct seeds to prevent order and fatigue bias).*

---

## 3. Annotation Sheet Structure & Fields

Each line in your assigned annotation file is a JSON object with the following fields:

```json
{
  "blinded_id": "ANN_A_001",
  "question": "What is the most important invention in human history?",
  "assigned_label": null,
  "confidence": null,
  "rationale": "",
  "requires_adjudication": false,
  "notes": ""
}
```

### Required Fields to Populate:

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `assigned_label` | String | `"KNOWN"`, `"AMBIGUOUS"`, `"UNKNOWN"` | Your classification according to `ANNOTATION_RUBRIC.md`. |
| `confidence` | String | `"HIGH"`, `"MEDIUM"`, `"LOW"` | Your confidence in this classification. |
| `rationale` | String | Non-empty text | Brief sentence explaining your reasoning (e.g., *"Subjective superlative without criteria"*, *"Single verifiable geographic capital"*, *"Future contingent date"*). |
| `requires_adjudication` | Boolean | `true` or `false` | Set to `true` if you feel the item is on a borderline between two classes and warrants panel review. |
| `notes` | String | Optional text | Any observations regarding grammar, unusual interpretations, or alternative readings. |

---

## 4. Step-by-Step Annotation Workflow

1. **Review the Rubric:** Thoroughly read `data/annotation/ANNOTATION_RUBRIC.md` before starting.
2. **Open Your Assigned File:** Open your designated file (`blinded_annotation_annotator_a.jsonl` or `b.jsonl`) in an editor.
3. **Process Each Query Linearly:**
   - Read the `question`.
   - Apply the Rubric Decision Tree:
     - Check for presupposition failure or logical impossibility $\to$ `UNKNOWN`.
     - Check for private secret, future event, or inaccessible history $\to$ `UNKNOWN`.
     - Check for value judgment, unconstrained superlative, or name collision $\to$ `AMBIGUOUS`.
     - Check for single verifiable fact $\to$ `KNOWN`.
   - Set `assigned_label` to `"KNOWN"`, `"AMBIGUOUS"`, or `"UNKNOWN"`.
   - Set `confidence` (`"HIGH"`, `"MEDIUM"`, or `"LOW"`).
   - Enter a concise `rationale`.
   - If borderline, set `"requires_adjudication": true`.
4. **Validate JSON Syntax:** Ensure each line remains valid JSON on a single line. Do not add trailing commas or delete the `blinded_id`.

---

## 5. Decision Rules for Common Dilemmas

- **"Is it Ambiguous or Unknown?"**
  - If distinct, legitimate answers exist that could each satisfy the user if clarified (e.g. *"What is the capital of Georgia?"* $\to$ Atlanta or Tbilisi), it is **`AMBIGUOUS`**.
  - If NO answer exists anywhere in the world because it is private, future, or unrecorded (e.g. *"What did George Washington eat for breakfast on April 4, 1772?"*), it is **`UNKNOWN`**.
- **"Is it Known or Ambiguous if there are multiple ways to phrase the answer?"**
  - Phrasing differences (e.g., *"4"*, *"four"*, *"equals four"*) are semantic paraphrases of the **same single answer**. This is **`KNOWN`**.
  - Truly distinct factual answers (e.g., *"Georgia"* referring to the US State vs the Sovereign Nation) are distinct entities. This is **`AMBIGUOUS`**.
- **"What about questions with false premises?"**
  - Questions like *"What is the boiling point of pure vacuum?"* or *"Who was the female US president in 1995?"* presuppose a condition that is false. Always classify these as **`UNKNOWN`**.

---

## 6. Submission & Quality Control

Upon completing all 300 records:
1. Run a JSON schema validation check against `data/annotation/annotation_schema.json`.
2. Verify that:
   - Exactly 300 records exist.
   - Zero null values remain in `assigned_label` or `confidence`.
   - Every record has a populated `rationale`.
3. Submit the completed file to the Adjudication Coordinator.
