# Human Adjudication Protocol & Dispute Resolution Framework

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Scope:** Resolution of Annotator Disagreements on the 300-Question Benchmark  
**File:** `data/annotation/ADJUDICATION_PROTOCOL.md`  

---

## 1. Purpose & Scope

The adjudication protocol defines the authoritative, structured procedure for resolving all classification disagreements between Annotator A and Annotator B, as well as queries flagged with `requires_adjudication = true`.

This protocol guarantees that final benchmark gold labels are epistemically sound, fully traceable, and backed by explicit rationales and evidentiary citations.

---

## 2. Trigger Conditions for Adjudication

An item enters the adjudication pipeline if ANY of the following criteria are met:

1. **Direct Label Disagreement:** $y_A \ne y_B$ (e.g., Annotator A assigns `KNOWN`, Annotator B assigns `AMBIGUOUS`).
2. **Explicit Escalation Flag:** Either annotator marked `"requires_adjudication": true`, regardless of whether their labels coincided ($y_A = y_B$).
3. **Low Confidence Flag:** Both annotators assigned the same label, but both recorded `"confidence": "LOW"`.
4. **Targeted Diagnostic Outlier:** The item belongs to the high-priority diagnostic review set (`data/annotation/targeted_diagnostic_review.jsonl`) and reveals empirical contradictions between LLM behavior and human consensus.

---

## 3. Tiered Adjudication Workflow

```
               [Annotator A & Annotator B Complete Sheets]
                                   │
                                   ▼
                   [Inter-Annotator Agreement Check]
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
    Unanimous Agreement                           Disagreement OR
    (y_A = y_B AND no flag)                 Flagged (y_A != y_B or flag=True)
            │                                             │
            ▼                                             ▼
    [Provisional Consensus]                       [Tier 1: Arbiter C Review]
                                                          │
                                         ┌────────────────┴────────────────┐
                                         ▼                                 ▼
                               Majority Reached (2/3)              Three-Way Split OR
                                   (e.g., C matches A)             Persistent Epistemic Contestation
                                         │                                 │
                                         ▼                                 ▼
                             [Adjudication Documented]          [Tier 2: Senior Panel Review]
                                         │                                 │
                                         └────────────────┬────────────────┘
                                                          ▼
                                             [Final Gold-Label Approval]
```

---

## 4. Tier 1: Blind Third-Arbiter Review (Arbiter C)

1. **Blinded Extraction:**  
   Items requiring adjudication are extracted into `data/annotation/adjudication_input.jsonl` using synthetic identifier `ARB_xxx`. Original labels from Annotator A and B are strictly hidden.
2. **Independent Classification:**  
   A senior research arbiter (Arbiter C), thoroughly trained on `ANNOTATION_RUBRIC.md`, independently classifies each query and writes a rationale.
3. **Resolution Rule (Majority Consensus):**  
   - If $y_C = y_A$, the majority label is $y_A$ (2-to-1).
   - If $y_C = y_B$, the majority label is $y_B$ (2-to-1).
   - The majority label is accepted as the provisional gold label, provided Arbiter C's rationale validates the criteria in the Rubric.

---

## 5. Tier 2: Senior Adjudication Panel Escalation

An item is escalated to the Senior Adjudication Panel under any of the following conditions:
- **Three-Way Disagreement:** Annotator A assigns `KNOWN`, Annotator B assigns `AMBIGUOUS`, Arbiter C assigns `UNKNOWN`.
- **Systematic Ambiguity:** The query hinges on an unsettled philosophical or domain-specific boundary (e.g., unproven mathematical conjectures, borderline historical records, presupposition subtleties).
- **Benchmark Group Inconsistency:** The resolution threatens to create an inconsistent label across members of the same question group (`question_group_id`).

### Panel Composition & Quorum
- Panel consists of the Lead PI / Senior Author and two Senior Researchers.
- Quorum requires unanimous consensus after inspecting authoritative reference material.

### Panel Deliberation Protocol:
1. **Bibliographic & Evidentiary Audit:**  
   Inspect encyclopedic standards (Britannica, ISO standards, scientific consensus, statutory legal texts).
2. **Delineation of Interpretations:**  
   For contested `AMBIGUOUS` vs `KNOWN`: Determine whether the secondary interpretation is genuinely salient in common public discourse or merely pedantic/contrived.
3. **Epistemic Barrier Audit:**  
   For contested `UNKNOWN` vs `KNOWN`: Confirm whether verifiable records exist anywhere in public archives or if the fact is lost/unrecorded.
4. **Binding Written Decision:**  
   The Panel authors a binding adjudication statement detailing:
   - Final Gold Label (`KNOWN`, `AMBIGUOUS`, or `UNKNOWN`).
   - Definitive Rationale citing external evidence or proof of inaccessibility.
   - Evidence Type (`consensus_reference`, `definitional_standard`, `delineated_interpretations`, `inaccessibility_proof`, `future_contingency`, or `presupposition_failure`).

---

## 6. Audit Trail & Manifest Preservation

Every adjudicated item must produce a permanent audit record in `data/annotation/adjudication_log.jsonl` containing:
- `question_id` (original candidate ID)
- `question`
- `annotator_a_label`, `annotator_a_rationale`, `annotator_a_confidence`
- `annotator_b_label`, `annotator_b_rationale`, `annotator_b_confidence`
- `arbiter_c_label`, `arbiter_c_rationale` (if invoked)
- `final_gold_label`
- `adjudication_tier` (`tier_1_majority` or `tier_2_senior_panel`)
- `formal_justification`
- `timestamp`

**CRITICAL SAFEGUARD:** Under no circumstances are existing candidate benchmark files (`uncertainty_benchmark_300_candidate_v1.1.jsonl`) or manifest files overwritten until the Adjudication Log is fully signed off by the Lead PI.
