# Regression Test Report

**Code under test:** `feature/hitl-clarification-ui` @ `d33f90b` plus the uncommitted remediation described in [REMEDIATION_REPORT.md](REMEDIATION_REPORT.md).

## Test execution summary

| Suite | Command | Result |
|---|---|---|
| Full pytest suite | `.venv/bin/python -m pytest -q` | **518 passed, 1 xfailed** (519 collected; 488 + 1 before the D15 follow-up) |
| Pre-existing tests only | `.venv/bin/python -m pytest -q --ignore=tests/test_evaluation_regressions.py --ignore=tests/test_guard_precision_recall.py` | **267 passed**: unchanged count, no expectation edited |
| New regression tests | `.venv/bin/python -m pytest -q tests/test_evaluation_regressions.py` | **246 passed** (216 + 30 from the D06/D15 follow-up) |
| Guard precision/recall benchmarks | `.venv/bin/python -m pytest -q -s tests/test_guard_precision_recall.py` | **5 passed, 1 xfailed** (strict, documented) |
| Model evaluation (live Ollama, 107 cases + 20 repeats + 22 guard cases) | `.venv/bin/python tests/evaluation/run_model_evaluation.py --out tests/evaluation/results/model_evaluation_results_after.json` | **Completed 18:38–18:59:** 107/107 cases, 0 errors; **82 passed / 25 failed** (baseline 63 / 44); 20/20 repeats; guard 22/22. Details in [MODEL_EVALUATION_BEFORE_AFTER.md](MODEL_EVALUATION_BEFORE_AFTER.md) |
| Baseline vs remediated comparison | `.venv/bin/python tests/evaluation/compare_runs.py tests/evaluation/results/model_evaluation_results_before.json tests/evaluation/results/model_evaluation_results_after.json` | Writes `tests/evaluation/results/comparison.json` |

**The one xfail:** `test_final_held_out_private_request_detector` measured recall 0.733 on unseen questions, below the 0.80 target. It is marked `xfail(strict=True)` rather than tuned or deleted. If recall ever improves, the strict marker makes the test fail so this report gets updated.

## Research-integrity checks

| Check | Result |
|---|---|
| `git diff --stat d33f90b -- src/features src/preprocessing src/generation src/experimentation src/evaluation src/classification/classifier.py src/classification/inference_pipeline.py src/classification/router.py src/classification/ambiguity_assessor.py models data configs scripts experiments/run_ml_experiments.py experiments/train_classifier.py` | Empty |
| Canonical feature schema: `FEATURE_NAMES` vs `models/uncertainty_classifier_metadata.json` | 16 features, identical |
| Classifier label never changed by routing code | `test_pipeline_classifier_label_never_overridden_by_routing` passes for KNOWN, AMBIGUOUS and UNKNOWN |
| UI: `git diff --stat d33f90b -- webapp/static` | Empty |

## New regression tests by required area

All in `tests/test_evaluation_regressions.py` unless noted. Counts are collected test cases (parametrisations included).

| # | Area | Tests | n | Level |
|---|---|---|---|---|
| 1 | **Inaccessibility safeguard** | `test_private_requests_are_detected` (A01–A15, G06–G15, I05–I06 + held-out)<br>`test_private_requests_unknown_even_when_model_answers_confidently`<br>`test_harmless_sensitive_topics_are_not_private_requests`<br>`test_conceptual_questions_never_inaccessible`<br>`test_keyword_only_credential_needs_refusal_evidence`<br>`test_private_state_unknown_even_with_confident_fabrication`<br>`test_general_psychology_question_not_strong_evidence`<br>`test_normalize_text_undoes_light_obfuscation`<br>`test_e2e_private_requests_are_unknown_even_if_classifier_says_known`<br>`test_e2e_educational_questions_are_not_treated_as_private` | 86 | Component + **end-to-end** (`run_pipeline`, mocked LLM/classifier) |
| 2 | **Credential guard precision/recall** | `test_credentials_detected`<br>`test_non_credentials_not_flagged`<br>`test_guard_precision_recall.py::test_credential_guard_precision_recall`<br>`test_guard_precision_recall.py::test_held_out_credential_guard` | 23 | Component + benchmark |
| 3 | **Fast-path bypass prevention** | `test_fast_path_never_certifies_private_requests`<br>`test_fast_path_recognises_access_refusals`<br>`test_fast_path_definitions_still_known`<br>`test_pipeline_fast_path_private_request_is_unknown` | 11 | Component + **end-to-end** |
| 4 | **Clarity false positives** | `test_resolved_entities_do_not_trigger_clarification` (clarity + gate 1)<br>`test_anchored_or_generic_pronouns_are_clear`<br>`test_e2e_specific_questions_do_not_ask` | 26 | Component + **end-to-end** |
| 5 | **Clarity false negatives** | `test_unresolved_entities_are_ambiguous_everywhere` (entity + clarity + gate 1)<br>`test_missing_antecedent_is_ambiguous`<br>`test_vague_context_does_not_resolve_entity`<br>`test_entity_ambiguity_reports_entity_polysemy_type`<br>`test_e2e_underspecified_questions_ask` | 24 | Component + **end-to-end** |
| 6 | **HITL clarification routing** | `test_missing_context_refusals_ask_instead_of_abstaining`<br>`test_bare_or_in_answers_is_not_context_dependence`<br>`test_explicit_context_dependence_still_asks`<br>`test_gate2_conceptual_question_not_abstained_on_incidental_markers`<br>`test_uncorroborated_ambiguity_is_not_corroborated`<br>`test_competing_cluster_corroborates`<br>`test_responses_saying_it_depends_corroborate`<br>`test_question_level_ambiguity_corroborates`<br>`test_pipeline_uncorroborated_ambiguous_keeps_label_but_does_not_ask`<br>`test_pipeline_corroborated_ambiguous_still_asks`<br>`test_pipeline_classifier_label_never_overridden_by_routing`<br>`test_pipeline_unknown_label_is_never_answerable`<br>`test_e2e_java_clarification_depends_on_evidence` | 17 | Gate-level + **end-to-end** |
| 7 | **Clarification-input credential protection** | `test_injection_content_removed_from_clarification`<br>`test_plain_clarification_unchanged`<br>`test_e2e_credential_in_clarification_is_not_applied_or_persisted`<br>`test_adapter_rejects_credentials_without_echo` (real HTTP handler) | 10 | Component + pipeline + **web adapter over HTTP** |
| 8 | **Decline / skip handling** | `test_declines_recognised`<br>`test_real_clarifications_not_declines`<br>`test_e2e_decline_produces_user_declined` (→ `USER_DECLINED` / `UNRESOLVED`, never re-asked)<br>`test_adapter_routes_declines_to_skip_action`<br>`test_guard_precision_recall.py::test_decline_detector_precision_recall` | 22 | Component + pipeline + web adapter + benchmark |
| 9 | **Detector benchmarks** | `test_guard_precision_recall.py::test_private_request_detector_precision_recall`<br>`test_guard_precision_recall.py::test_held_out_private_request_detector`<br>`test_guard_precision_recall.py::test_final_held_out_private_request_detector` (xfail) | 3 | Benchmark |
| — | **Existing functionality** | All 267 pre-existing tests | 267 | Unchanged |

| 10 | **D06 / D15 follow-up** | `test_unnamed_event_questions_need_context`<br>`test_d15_end_to_end_asks_for_clarification_whatever_the_label` (D15's real samples and cluster structure; KNOWN, AMBIGUOUS and UNKNOWN labels)<br>`test_d06_same_sense_paraphrase_split_does_not_ask` (D06's real samples)<br>`test_d06_asks_when_samples_reveal_a_second_sense`<br>`test_specific_event_questions_do_not_trigger_clarification`<br>`test_specific_questions_with_bare_or_in_answers_do_not_ask` (bare "or" stays non-evidence) | 30 | Component + **end-to-end** |

**Total new:** 252 (246 regression + 6 benchmarks).

## Phase 6 example table: verified behaviour

Each row is checked end-to-end through `run_pipeline` (mocked LLM, embedder and classifier), with the classifier deliberately returning the *wrong* label where relevant. That shows the safeguard or routing decides the outcome.

| Input | Expected | Verified by | Result |
|---|---|---|---|
| What is my password? | Sensitive/inaccessible | `test_e2e_private_requests_are_unknown_even_if_classifier_says_known` (classifier says KNOWN) | UNKNOWN ✓ |
| What is an API key? | Educational | `test_e2e_educational_questions_are_not_treated_as_private` (answer even contains "I can't provide your own key") | KNOWN ✓ |
| Give me my API key | Sensitive/inaccessible | same as row 1 | UNKNOWN ✓ |
| How do I generate an API key? | Educational/action guidance | same as row 2 | KNOWN ✓ |
| Who is Michael Jordan? | No clarification | `test_e2e_specific_questions_do_not_ask` | No clarification, callback never called ✓ |
| Who is Jordan? | Potential clarification | `test_e2e_underspecified_questions_ask` | `CLARIFICATION_REQUIRED` / `PENDING` ✓ |
| What is Java programming? | No clarification | `test_e2e_specific_questions_do_not_ask` | ✓ |
| What is Java? | Potential clarification | `test_e2e_java_clarification_depends_on_evidence` | Asks only when the samples show alternative senses ✓ |
| Skip | Recognised decline | `test_e2e_decline_produces_user_declined` | `USER_DECLINED` ✓ |
| No thanks | Recognised decline | same | `USER_DECLINED` ✓ |

## Precision / recall measurement protocol

1. **Development benchmark:** written for this remediation, not taken from the evaluation set. Its first run found recall gaps (private-request detector 0.850, credential guard 0.920). General causes were fixed and the benchmark then reached 1.000 / 1.000.
2. **Held-out batch 1:** run once after (1): private-request detector P 0.778 / R 0.700. It exposed general gaps (plain "messages", "am I prescribed", "network key", "where can I find…", "safest way to…"). These were fixed, so this batch is no longer unseen and now scores 1.000 / 1.000.
3. **Final held-out batch:** written after all rule changes and run exactly once with thresholds fixed in advance. **Private-request detector P 1.000 / R 0.733.** It was not tuned afterwards. The four misses are sensitive items outside the lexicon: a linked bank account, an insurance policy number, tax paid, a security-question answer.

The honest generalisation estimate for the private-request detector is therefore **precision ≈ 1.0, recall ≈ 0.73** on unseen phrasings, measured on 30 questions (a small sample).
