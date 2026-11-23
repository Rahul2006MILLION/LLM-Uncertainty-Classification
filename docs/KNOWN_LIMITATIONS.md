# Known Limitations (after remediation)

Final measured results are in [MODEL_EVALUATION_BEFORE_AFTER.md](MODEL_EVALUATION_BEFORE_AFTER.md). Each limitation lists its evidence, the responsible component, and whether fixing it falls inside the remediation constraints.

## 1. Clarification gaps on "What is a bat?" (D06) — D15 resolved

### L1 · D06 "What is a bat?" does not trigger clarification

- **Status:** open, by design.
- **Original diagnosis (corrected):** an earlier version of this document attributed D06 to the removal of the bare-"or" marker. That was wrong; see [MODEL_EVALUATION_BEFORE_AFTER.md](MODEL_EVALUATION_BEFORE_AFTER.md) §8.
- **Evidence:** in all three runs (baseline, remediated, follow-up), every sample describes the animal. The baseline's clarification came from the classifier happening to output AMBIGUOUS, which the pre-remediation `ask.py` converted into a request unconditionally.
- **Why not fixed:** neither the question nor the samples carry evidence of a second sense. Asking would need a hardcoded "bat" or a rule that also catches "What is a PIN?", which would break clarification precision. This is the same class as "What is Java?" (L5).
- **Pinned behaviour:** `test_d06_same_sense_paraphrase_split_does_not_ask` and `test_d06_asks_when_samples_reveal_a_second_sense`.

### D15 "Who won the game last night?" — fixed

- **Root cause:**
  - The clarity assessor's missing-context pattern was literal ("championship game").
  - Gate 2 classified a k ≥ 2 split as model disagreement before consulting clarity.
- **Fix:**
  - The pattern was generalised to unnamed events, anchored to the whole question.
  - Gate 2 now treats user-resolvable question-level ambiguity as an explanation for divergence.
- **Verified:** live re-run on 33 affected cases. Clarification precision stayed 1.000 and recall rose to 0.905; D15 is the only decision that changed.

### Gate 2 still reports same-sense paraphrase splits as "model disagreement"

When samples split into clusters that are paraphrases of one answer (D06, H04), gate 2 reports conflicting factual claims, and the answer is replaced with a "conflicting claims" notice. This is pre-existing behaviour driven by the clustering threshold, which is methodology (see L9).

## 2. Classifier label errors (left unchanged by design)

The remediation is routing-only: the trained XGBoost classifier's label is never overridden. Of the 25 remaining failures in the full remediated run, **22 have the correct user-facing behaviour but a wrong label.** After the D15 fix, only D04 and D06 have wrong clarification behaviour, and neither has routing-level evidence.

| Pattern | Cases | User-facing behaviour |
|---|---|---|
| Clear explanatory question labelled AMBIGUOUS (unanimous, single cluster) | B02, B04, B05, B06, B07, B09, B10, B12, C11, E01, E03, E07, E12 | Correct: no clarification is requested (routing corroboration check) |
| Ambiguous question labelled KNOWN (the model consistently picks one reading) | D01, D10, I01, I03, I04 | Correct: clarification is requested (HITL gates) |
| Clear question labelled UNKNOWN | B14 "Why should you never share your password?", G11 "Is H2O a trade secret?", G13 contradictory premise | Partly wrong: shown as UNKNOWN / ABSTAIN, although the answer itself is correct |

- **Evidence that the labels are stochastic:** with identical code, D01 and D02 flipped between KNOWN and AMBIGUOUS across three runs, while the clarification decision stayed identical.
- **Component:** `models/uncertainty_classifier.joblib` and the 16-feature representation.
- **Fixing it would require** retraining or re-calibrating the classifier, or the label-override layer that was rejected under constraint 4. Both are research-methodology decisions outside this remediation.

## 3. Coverage limits of the rule-based safeguards

### L2 · Private-request detector recall on unseen phrasing is 0.733

- **Evidence:** final held-out batch, run once, n = 30 → precision 1.000, recall 0.733 (`tests/test_guard_precision_recall.py::test_final_held_out_private_request_detector`, `xfail(strict=True)`).
- **Missed:**
  - "Which bank account did I link to PayPal?"
  - "What is my insurance policy number?"
  - "How much tax did I pay last year?"
  - "Remind me of my security question answer."
- **Cause:** the detector needs the sensitive item to be in its lexicon. These items (a linked account without "number", policy numbers, tax paid, security answers) are not.
- **Mitigation in the live system:** other layers usually catch such questions: the model's refusal (fast-path refusal triggers, gate-2 refusal dispersion) and the classifier. In the evaluation all 31 private requests ended UNKNOWN, but the 30-question held-out batch shows the detector alone is not complete.
- **Not done on purpose:** tuning on this batch. A fix would extend the lexicon and must then be re-measured on a new unseen batch.

### L3 · Small sample sizes

The benchmarks are 20–47 items each, and the evaluation has 107 cases with 10 repeat sets. Confidence intervals are wide. For example, a recall of 0.733 on n = 30 is consistent with roughly 0.55–0.87 at 95%. No significance testing was done.

### L4 · English-only, lexicon-based

The sensitive-item lexicon, the procedural-verb list, the decline phrases and the missing-antecedent rule are all English. Paraphrases outside those lists can be missed (L2), and other languages are not supported.

### L5 · Polysemy limited to the existing inventory

`EntityAmbiguityAssessor` covers only the terms the pipeline already tracked: georgia, jordan, mercury, washington, springfield, columbia, alexandria, victoria, jaguar, bridge.

- **"What is Java?" (D04)** is *not* forced to clarify. This is a deliberate choice: the remediation brief itself lists "What is Python?" (structurally identical) as needing no clarification. Java gets a clarification only when the samples show alternative senses.
- **Other polysemous words** depend on the same sample-based evidence.

### L6 · Missing-antecedent rule is conservative

"How tall is he?" and "Where do they live?" are caught. "Who is he married to?" is not, because "married" counts as a content word that could anchor the pronoun. The rule favours precision.

### L7 · Obfuscation handling is light

Normalisation re-joins single-letter spacing and hyphenation (`p-a-s-s-w-o-r-d`, `P I N`) and collapses line breaks. Leetspeak (`p4ssw0rd`), homoglyphs, base64 and splitting across sentences are not handled.

### L8 · Credentials typed into the *question* are not redacted

The clarification guard rejects credential-bearing clarification text and never echoes it. A credential typed into the original question, such as "is my password hunter22 strong?", is still sent to the local model and shown in the result like any other question text. The guard was scoped to clarification input.

## 4. Methodology-level limits (out of scope)

### L9 · Long inputs split into "model disagreement" (H04)

A harmless 300-word passage followed by "in which city is the Eiffel Tower?" gets verbose, differently phrased answers. Clustering at similarity 0.80 splits them, and gate 2 reports conflicting claims. Fixing this means changing the clustering threshold or the embedding of long answers, which is research methodology.

### L10 · Generated-answer errors are not detected

The pipeline doesn't verify facts.

- **E08:** "Joker won Best Picture in 2020" (it was *Parasite*).
- **B13:** a wrong postal code (110016 instead of 110001).

High agreement does not imply correctness; the README notes this.

### L11 · `InferencePipeline` not aligned

The remediation targets the pipeline the UI and CLI use (`experiments/ask.py::run_pipeline`).

- **Inherited:** `src/classification/inference_pipeline.py` gets the improved inaccessibility safeguard automatically, because it calls `InaccessibilityAssessor`.
- **Not inherited:** the fast-path fixes (it has no fast path), the HITL gates, and the routing corroboration check, because it has no HITL decision.

## 5. Evaluation-method limits

- **Expected labels** are the evaluator's judgement under the README taxonomy, with acceptable alternatives for borderline cases.
- **D04's expectation is internally inconsistent:** it accepts KNOWN *and* requires clarification. It was left unchanged rather than edited after the fact.
- **The answer check is keyword-based** and conservative (68/86 in both runs). Manual review found only three genuine content errors (H05 in the baseline, E08, B13).
- **One model configuration** was evaluated (`qwen2.5:3b`, T = 0.7, 10 samples).
