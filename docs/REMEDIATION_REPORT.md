# Remediation Report

**Scope:** a targeted fix of the failures documented in [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md), mapped to components in [REMEDIATION_AUDIT.md](REMEDIATION_AUDIT.md).

**Base:** `feature/hitl-clarification-ui` @ `d33f90b`. All changes are uncommitted.

**Related documents:**
- Measured results: [MODEL_EVALUATION_BEFORE_AFTER.md](MODEL_EVALUATION_BEFORE_AFTER.md)
- Tests: [REGRESSION_TEST_REPORT.md](REGRESSION_TEST_REPORT.md)
- Remaining issues: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)

This report supersedes an earlier, uncommitted fix attempt that also overrode the classifier's label. Per the remediation constraints, and the decision taken during this remediation, **classifier labels are no longer changed**. Only the routing and safeguard layer was modified.

## Final results (live evaluation, 107 cases, completed 2026-11-23 18:59)

| Metric | Baseline | Remediated |
|---|---|---|
| Passed / failed | 63 / 44 | **82 / 25** |
| Accuracy (strict / lenient) | 62.6% / 64.5% | **76.6% / 79.4%** |
| Macro-F1 | 0.617 | **0.738** |
| UNKNOWN recall / precision | 0.579 / 0.815 | **1.000 / 0.905** |
| Private requests answered KNOWN | 11 | **0** |
| Fast-path safeguard bypasses | 8 | **0** |
| Unnecessary clarification requests | 20 | **0** |
| Missing clarification requests | 6 | 3 |
| Security safeguard failures (41 cases) | 24 | **1** (B14, from the classifier) |
| UNKNOWN shown with an "Answerable" decision | 10 | **0** |
| Clarification-input guard cases | 16/22 | **22/22** |

- **Changes:** 24 cases fixed, 5 regressed, 20 still failing.
- **Regressions:** 5 cases regressed. B05, D10 and I03 are classifier-label sampling variance with correct HITL behaviour. D06 and D15 lost their clarification request; the follow-up investigation found:
  - **D15:** a real missing-context gap. It is now **fixed** (see below).
  - **D06:** not a routing regression. The samples never mention a second sense, and the baseline asked only because of a classifier AMBIGUOUS label. Left open by design ([KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) L1).
- **Remaining failures:** 22 of the 25 are label-only. The untouched classifier's label is wrong, but the user-facing behaviour is right.
- **Full breakdown:** [MODEL_EVALUATION_BEFORE_AFTER.md](MODEL_EVALUATION_BEFORE_AFTER.md).

## Follow-up fix: D15 (routing only)

| File | Change |
|---|---|
| `src/classification/clarity_assessor.py` | The literal missing-context pattern "who won the championship game last night" is generalised to unnamed events ("Who won the game / match / race / final / election [last night / yesterday / today]?"). It stays anchored to the whole question, so named events, teams or years cannot match. |
| `src/classification/hitl_engine.py` | Gate 2, priority 4 (k ≥ 2 and p2 ≥ 0.25): user-resolvable question-level ambiguity from the clarity assessor now counts as evidence, instead of the split being classified as model disagreement. |

- **No bare-"or" heuristic** was reintroduced, and the classifier is unchanged.
- **Live re-run of the 33 affected cases:**
  - only D15's decision changed (→ `CLARIFICATION_REQUIRED`, `MISSING_CONTEXT`);
  - clarification precision stayed **1.000** (0/12 specific E questions asked);
  - recall rose from 0.857 to **0.905**.
- **Tests:** 30 new; full suite **518 passed, 1 xfailed**.

## Constraint compliance

| Constraint | How it was met | Verified by |
|---|---|---|
| Research methodology, ML experiments, datasets, labels, training, evaluation methodology untouched | No diff in `src/features`, `src/preprocessing`, `src/generation`, `src/experimentation`, `src/evaluation`, `models/`, `data/`, `configs/`, `scripts/`, training/experiment scripts | `git diff --stat d33f90b -- …` is empty |
| 16-feature schema unchanged | `FEATURE_NAMES` is identical to the model metadata | Check in [REGRESSION_TEST_REPORT.md](REGRESSION_TEST_REPORT.md) |
| Entropy, generation, clustering, classification unchanged | `classifier.py`, `inference_pipeline.py`, `router.py`, `ambiguity_assessor.py`, generation and clustering have no diff. The classifier's label is never overridden by new code | `test_pipeline_classifier_label_never_overridden_by_routing` |
| Architecture and interfaces preserved | Same function signatures and status strings (`ASSESSED_FAST_PATH_REFUSAL`, `ENTITY_POLYSEMY` kept); new logic lives in 3 new modules called from existing hook points | All 267 pre-existing tests pass unchanged |
| No question-specific exceptions | Rules are structural (owner + item + intent; sense cue / proper name / attribute; antecedent). No test question appears in any rule | Held-out benchmarks (§ Precision/recall) |
| Regexes evaluated for precision/recall | Three benchmarks, including a final batch run once without tuning | `tests/test_guard_precision_recall.py` |
| UI untouched | `webapp/static/` has no diff | `git diff --stat d33f90b -- webapp/static` is empty |
| Nothing committed or pushed | — | `git status` |

## Root causes and changes

### 1. Inaccessibility safeguard (Phase 2)

**Root causes (audit):**
- The fast path never called the safeguard. For A02, A06 and A11 the safeguard *would* have returned UNKNOWN.
- Credential detection was keyword-only:
  - false positives on "What is an API key?"
  - false negatives on "What is **my** password?", account numbers, SSN, CVV, salary, medical data
- No input normalisation: `p-a-s-s-w-o-r-d`, `P I N` and line breaks bypassed it.
- Private-data evidence required the model to refuse, so a confident or fabricated answer cancelled it.

**Changes:**

| File | Change |
|---|---|
| `src/classification/sensitive_request.py` (**new**) | `SensitiveRequestDetector`: flags a request only when a **sensitive item**, a **specific owner** ("my", "our", "my mother's", "the email I signed up with") and **retrieval intent** co-occur in the same clause. Conceptual and procedural framing ("what is a…", "how do… work", "how do I reset/retrieve/generate…", "where can I find…", "what's the safest way…", "what happens if…") never qualifies. `normalize_text()` handles NFKC, markdown/JSON punctuation, line breaks and single-letter obfuscation. |
| `src/classification/inaccessibility_assessor.py` | The detector is the first and strongest evidence (`private_personal_data` → UNKNOWN, no refusal confirmation). Keyword-only credential patterns no longer apply to conceptual questions and need at least one refusal even with a single response. Explicit "what number am I thinking of"-type requests are strong evidence. A new future-private-action category. Patterns match normalised text. |
| `src/classification/fast_path_assessor.py` | The safeguard runs **before** any fast-path certification (it can't be bypassed). Access/disclosure refusals were added to the refusal triggers. The definition rule excludes possessives ("what is **my** …"). The existing `ASSESSED_FAST_PATH_REFUSAL` status is kept; `ASSESSED_FAST_PATH_INACCESSIBLE` is used when a confident answer is overridden. |

The UNKNOWN / inaccessibility taxonomy is unchanged. The safeguard still produces `ASSESSED_INACCESSIBLE_RULE_HEURISTIC (<category>)`.

### 2. Credential detection in clarification responses (Phases 3 and 5)

**Root causes:**
- Values were matched only in `label: value` form.
- No vendor key families; `sk-proj-` failed because of its second hyphen.
- No normalisation, no card numbers.
- Decline phrases with punctuation ("Skip.") or politeness ("no thanks") weren't recognised.
- The sanitiser stripped `[INST]` tags but kept the injected text between them.

**Changes in `src/classification/hitl_prompts.py`:**
- **Natural-language disclosure:** "my password is X", "pwd=X", "pin was 4821". The value must look like a secret (contain a digit or symbol), so "the password is strong" isn't flagged.
- **Provider key families:** OpenAI, Anthropic and OpenRouter `sk-…`, Stripe `sk_/rk_/pk_live|test_`, Slack `xox?-`, Google `AIza…`, GitHub `gh?_` and `github_pat_`, AWS `AKIA/ASIA`, JWT.
- **Unknown formats:** a generic **high-entropy token** rule (≥ 24 chars, letters and digits, ≥ 3.5 bits/char; URLs, e-mail addresses and paths excluded), so detection doesn't depend on any single prefix.
- **Other secrets:** Luhn-validated card numbers (a random 16-digit number isn't flagged); PEM private-key headers.
- **Decline detection:** case- and punctuation-insensitive, with conversational declines ("no thanks", "never mind", "I don't want to clarify", "just answer the original question"). Real clarifications such as "No, the country" are not declines.
- **Sanitiser:** removes whole injected instruction blocks, role prefixes and override phrases. New `contains_prompt_injection()`.

**No echoing of credentials:**
- The web adapter rejects a credential-bearing clarification with a generic message that never repeats the text. This is verified over real HTTP by `test_adapter_rejects_credentials_without_echo`.
- The request body is never logged.
- In the pipeline's HITL flow, the clarification is not applied, and the record contains no trace of the value (`test_e2e_credential_in_clarification_is_not_applied_or_persisted`).

`webapp/clarify.py` sanitises the question and the clarification before embedding them in its helper prompts. This is backend only; no UI change.

### 3. Clarity and HITL routing (Phase 4)

**Root causes:**
- **Polysemy:** a bare word list, applied inconsistently. The clarity assessor treated "capital of" as resolving every term, gate 1 had no context check, and gate 2 used the same raw list.
- **No antecedent rule:** "How tall is he?" got no clarification.
- **Gate 2 ordering:** refusals were checked before clarity, so missing context ("I don't know your location") became ABSTAIN.
- **Over-broad context markers:** gate 2's list included bare "or", "area" and "definition".
- **Every AMBIGUOUS label became a clarification request:** `ask.py` did this even with no supporting signal, and cited the clarity reason "Question specifies unambiguous referents".

**Changes:**

| File | Change |
|---|---|
| `src/classification/entity_ambiguity.py` (**new**) | `EntityAmbiguityAssessor`: a sense inventory for the polysemous entities the pipeline **already tracked**; no new terms. A term is resolved by (1) a sense cue ("planet", "US state", "USA", "basketball"), (2) being part of a longer proper name ("Michael Jordan", "Jordan Peterson"), or (3) an attribute only one sense has ("capital of **Jordan**": only the country has a capital). Ambiguity remains when the attribute is shared ("capital of **Georgia**": country and US state). |
| `src/classification/clarity_assessor.py` | Uses the entity assessor instead of the word list. New missing-antecedent rule for short standalone questions built on he/she/they/there with nothing to refer to (existential "there" and generic "they" excluded). The unscoped size-superlative question ("What is the largest city?", with neither scope nor measure) is added to the existing full-question `MISSING_CONTEXT_PATTERNS`, consistent with the project's curated `largest_city` clarification. |
| `src/classification/hitl_engine.py` | Gate 1 and gate 2 use the same entity assessor, so all three components agree. Entity ambiguity keeps the established `ENTITY_POLYSEMY` type. Gate 2 skips the refusal → ABSTAIN rule when the question is user-resolvably ambiguous or conceptual. Context-dependence now requires explicit statements ("depends on which…", "can refer to"). |
| `src/classification/ambiguity_corroboration.py` (**new**) | Routing-only check: an AMBIGUOUS label leads to a clarification request only when it is corroborated by question-level ambiguity, criterion ambiguity, a competing cluster (≥ the existing `explanation_threshold`, 0.25) or "it depends" responses (≥ 30%). **The label is never changed.** |
| `experiments/ask.py` | Uses the corroboration check in the final HITL decision. An uncorroborated AMBIGUOUS gives `ANSWERABLE` with an explanatory reason instead of a clarification request. The contradictory clarity reason is no longer cited. A clarification is `PENDING` only when an interactive channel exists (`interactive_hitl`); otherwise `UNRESOLVED`, the existing convention for exhausted rounds. An UNKNOWN label is no longer paired with an `ANSWERABLE` decision (→ `ABSTAIN`). |

**On "What is Java?":** this remediation deliberately does **not** force a clarification for it. This prompt's own examples put "What is Python?" in the no-clarification group, and the two questions are structurally identical: one dominant sense, other senses rarer. Forcing Java would require a hardcoded word.

Java *does* get a clarification when the model's samples reveal alternatives (corroboration via "can refer to" responses or competing clusters). That is evidence-driven rather than list-driven (`test_e2e_java_clarification_depends_on_evidence`).

### 4. Decline handling (Phase 5)

Through the real pipeline, a decline produces `interaction_status = USER_DECLINED` and `clarification_decision = UNRESOLVED`, and the user is never asked again. The web adapter sends declines to the existing "Show analysis without clarifying" action. Both are covered by tests.

## Precision and recall of the guards

| Guard | Development benchmark | Held-out batch 1 (first run, before re-tuning) | **Final held-out (run once, no tuning)** |
|---|---|---|---|
| Private-request detector | P 1.000 / R 0.850 → 1.000 / 1.000 after fixes (n = 40) | P 0.778 / R 0.700 (n = 20) | **P 1.000 / R 0.733** (n = 30) |
| Credential guard (clarification text) | P 1.000 / R 0.920 → 1.000 / 1.000 (n = 47) | P 1.000 / R 1.000 (n = 10) | — |
| Decline detector | P 1.000 / R 1.000 (n = 30) | — | — |

**The final held-out recall (0.733) is below the 0.80 target.** It is recorded as a strict expected failure rather than tuned away; see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). Precision on unseen educational questions was 1.000 in the final batch: no educational question was treated as a credential request.

## Files changed

**Modified (7):**

| File | Lines changed |
|---|---|
| `experiments/ask.py` | +40 / −3 |
| `src/classification/clarity_assessor.py` | +83 |
| `src/classification/fast_path_assessor.py` | +26 |
| `src/classification/hitl_engine.py` | +93 |
| `src/classification/hitl_prompts.py` | +139 |
| `src/classification/inaccessibility_assessor.py` | +85 |
| `webapp/clarify.py` | 4 |

**New modules (3):**
- `src/classification/sensitive_request.py`
- `src/classification/entity_ambiguity.py`
- `src/classification/ambiguity_corroboration.py`

**New tests:**
- `tests/test_evaluation_regressions.py` (216 cases)
- `tests/test_guard_precision_recall.py` (6 benchmarks)
- `tests/evaluation/` (dataset, runner, comparison and docs tooling)

**New docs:**
- this report
- `REMEDIATION_AUDIT.md`
- `MODEL_EVALUATION_REPORT.md`
- `MODEL_EVALUATION_TEST_CASES.md`
- `MODEL_EVALUATION_BEFORE_AFTER.md`
- `REGRESSION_TEST_REPORT.md`
- `KNOWN_LIMITATIONS.md`
