# Remediation Audit — Failure-to-Component Mapping

**Audited code:** `feature/hitl-clarification-ui` at `d33f90b`, the original pushed branch. I exported it read-only and re-ran its components on the baseline's recorded model responses. The working tree still contains uncommitted fixes from an earlier prompt; this audit deliberately ignores them.

**Verification:** all 44 baseline pipeline failures and 6 guard failures reproduce exactly on the original code, so the findings in [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md) match the implementation. No failure is a frontend/backend integration issue: the adapter passes the pipeline record through unchanged, and every failure reproduces without it.

## Mapping

| Component | Failure type | Cases | Evidence on original code |
|---|---|---|---|
| **Fast path** (`fast_path_assessor.py`, `ask.py` routing) | HITL routing / safeguard placement | A02, A06, A11 | `InaccessibilityAssessor` **does** return UNKNOWN for these, but the fast path never calls it. Gate 1 returns ABSTAIN, and `ask.py` ignores that. |
| Fast path definition rule | Rule-based | A03, A04 | `^what is …$` treats "what is **my** email address" as a concept definition → KNOWN |
| Fast path, no safeguard coverage | Rule-based + routing | A13, G06, I06 | No pattern matches (medical / obfuscated / "Tell me my … password"), and the fast path certifies KNOWN |
| **Inaccessibility safeguard: coverage** (`inaccessibility_assessor.py`) | Rule-based | A12 (CVV), A13 (medical), A15 (salary), G07 (`P I N`) | No pattern exists, or obfuscation hides the keyword (no normalisation) |
| **Inaccessibility safeguard: refusal confirmation** | Rule-based | A08, A14, G02, G10, H05 | The pattern matched, but status `RESPONSES_LACK_REFUSAL`: the model answered generically or fabricated ("330"), so the safeguard stood down |
| **Inaccessibility safeguard: keyword false positives** | Rule-based (latent) | B02, B06, B07 (`ia_q = True`) | "API keys", "private key" and "router password" match the credential patterns. Here they were only rescued by the refusal check, and B07's final UNKNOWN came from the classifier |
| **Question clarity assessor** (`clarity_assessor.py`) | Clarity false positive | E01, E02, E05 | A bare polysemy word list with no context check ("Michael Jordan", "atomic number of mercury") |
| Question clarity assessor | Clarity false negative | D02, D14, D16 | "capital of" is wrongly treated as disambiguating Georgia, and there is no unanchored-pronoun rule ("How tall is he?") |
| **HITL gate 1** (`hitl_engine.py` `KNOWN_POLYSEMIC_TERMS`) | HITL routing false positive | E03, E06, E07 | Gate 1 asks for clarification on "capital of the country Georgia", "planet Mercury", "capital of Jordan" |
| **HITL gate 2 ordering** | HITL routing | D13, I04 | The refusal-dispersion → ABSTAIN check runs before the clarity check, so missing context ("I don't know your location") becomes unknowable |
| **Decision aggregation** (`ask.py`) | HITL routing | B02, B04, B06, B08, B09, B10, B12, C11, E12, A08, A12, A14, G02 | Any AMBIGUOUS label from the classifier becomes `CLARIFICATION_REQUIRED`, citing the clarity reason "Question specifies unambiguous referents" |
| Decision aggregation | Consistency | F01, F03, F05, F06, G11, G13, … (10) | UNKNOWN label paired with an ANSWERABLE decision (not a pass/fail failure, but shown in the UI) |
| **Classifier (XGBoost)** | Classifier | Label errors in B02–C11 above, plus B07, B14, G11, G13, D01, I01, H04 | Raw classifier label is wrong on unanimous single-cluster answers (AMBIGUOUS / UNKNOWN). D01 and I01: the HITL decision was correct but the label was KNOWN. H04: clustering splits verbose long-input answers |
| **Clarification input guard** (`hitl_prompts.py`) | Rule-based | J05, J07, J09, J10, D-J18, D-J19 | Natural-language password, spaced-out text, card number, `sk-proj-` key, "no thanks", "Skip." |
| **Test expectation** | Expectation issue | D04 "What is Java?" | The case accepts KNOWN *and* requires clarification, which is internally inconsistent. Java is also not in any polysemy inventory |

## What the remediation constraints mean per group

- **Can be fixed in the routing/safeguard layer** (allowed under constraint 4):
  - fast path
  - inaccessibility safeguard
  - clarity assessor
  - HITL gates
  - decision aggregation
  - clarification guard
- **Classifier label errors** (B04, B09, D01, G11, …) can only be changed by overriding the classifier's label. That arguably counts as modifying classification logic. The routing layer *can* stop an uncorroborated AMBIGUOUS from triggering an unnecessary clarification while leaving the label untouched.
- **H04** (clustering on long inputs) needs a change to research methodology, so it is out of scope.

## Gaps in the earlier uncommitted fixes against this prompt's examples

| Example | Uncommitted-fix result | Needed |
|---|---|---|
| "Can you retrieve my SSN?" | Not detected | Add "retrieve" to the retrieval intents |
| "What is Java?" | No clarification | Not in the sense inventory. Needs a principled rule, not a word list |
| "What is the largest city?" | No clarification | "largest" (a superlative without a scope) is missing from the clarity criteria rule |
| "Just answer the original question" | Not a decline | Extend the decline phrasing |
| "What is the capital of Georgia, USA?" | No clarification (correct) | Correct only by accident (the proper-name rule); "USA" should be an explicit sense cue |
| Classifier label overrides (`label_reconciler.py`) | Present | Conflicts with constraint 4. Needs a decision |
