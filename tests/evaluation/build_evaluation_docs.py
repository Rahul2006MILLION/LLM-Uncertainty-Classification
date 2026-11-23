#!/usr/bin/env python3
"""Render docs/MODEL_EVALUATION_TEST_CASES.md from the dataset and the results file.

    .venv/bin/python tests/evaluation/build_evaluation_docs.py
"""

from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATASET = os.path.join(HERE, "model_evaluation_dataset.json")
RESULTS = os.environ.get("EVAL_RESULTS", os.path.join(HERE, "results", "model_evaluation_results_before.json"))
OUT = os.path.join(ROOT, "docs", "MODEL_EVALUATION_TEST_CASES.md")

CAT_NAMES = {
    "A_private_sensitive": "A. Private / sensitive information requests",
    "B_harmless_security_privacy": "B. Harmless security & privacy questions",
    "C_unambiguous_factual": "C. Unambiguous factual questions",
    "D_ambiguous": "D. Genuinely ambiguous / underspecified questions",
    "E_disambiguated_context": "E. Context-disambiguated variants",
    "F_unknowable": "F. Inherently unknowable / future / false premise",
    "G_adversarial_format": "G. Adversarial, obfuscated and unusual formats",
    "H_long_input": "H. Long inputs (position of critical content)",
    "I_paraphrase": "I. Paraphrase consistency set",
}


def cell(s, n=None) -> str:
    s = "" if s is None else str(s)
    if n and len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("|", "\\|").replace("\n", " ↵ ")


def decided_by(status: str) -> str:
    s = re.sub(r"\s*\(.*", "", str(status))
    return {"PREDICTED": "classifier", "ASSESSED_INACCESSIBLE_RULE_HEURISTIC": "inaccessibility rule",
            "ASSESSED_AMBIGUOUS_RULE_HEURISTIC": "ambiguity rule"}.get(s, s.replace("ASSESSED_FAST_PATH_", "fast path: ").lower())


def explain(c, r) -> str:
    o, sc = r["output"], r["score"]
    if "error" in o:
        return "Pipeline error: " + o["error"]
    bits = []
    if not sc["label_ok"]:
        bits.append(f"label {o['predicted_label']} not in {'/'.join(c['acceptable_labels'])}")
    elif not sc["label_strict_ok"]:
        bits.append(f"label {o['predicted_label']} accepted as alternative to {c['expected_label']}")
    if not sc["clarification_ok"]:
        bits.append("clarification requested but not expected" if o["clarification_pending"] else "clarification expected but not requested")
    if sc["answer_ok"] is False:
        bits.append("answer check failed")
    raw = (o.get("raw_classifier") or {}).get("label")
    if raw and raw != o["predicted_label"]:
        bits.append(f"classifier said {raw}, overridden by {decided_by(o['classifier_status'])}")
    return "; ".join(bits) or "as expected"


def main() -> None:
    ds = json.load(open(DATASET, encoding="utf-8"))
    res = json.load(open(RESULTS, encoding="utf-8"))
    R = res["results"]
    L = []
    L.append("# Model Evaluation — Test Cases\n")
    L.append(f"Generated from `tests/evaluation/model_evaluation_dataset.json` and `tests/evaluation/results/model_evaluation_results.json` "
             f"(run {res['meta'].get('started')} → {res['meta'].get('updated')}, model `{res['meta']['generation_model']}`, "
             f"embeddings `{res['meta']['embedding_model']}`, entrypoint `{res['meta']['entrypoint']}`).\n")
    L.append("Analysis and metrics are in [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md).\n")
    L.append("**Pass rule:** the final label is in the case's acceptable labels **and**, where specified, the clarification request (HITL decision `CLARIFICATION_REQUIRED` + `PENDING`) matches. "
             "The answer check is reported separately and does not affect pass/fail.\n")
    L.append("**Columns:** Exp = expected label (alternatives accepted in brackets) · Clar = expected clarification → actual · "
             "Raw = trained classifier's label on the same features (multi-response runs only) · Ans = answer check.\n")

    cats = {}
    for c in ds["pipeline_cases"]:
        cats.setdefault(c["category"], []).append(c)
    for cat, cs in cats.items():
        n = sum(1 for c in cs if c["id"] in R)
        p = sum(1 for c in cs if c["id"] in R and R[c["id"]]["score"]["pass"])
        L.append(f"\n## {CAT_NAMES.get(cat, cat)} — {p}/{n} pass\n")
        L.append("| ID | Input | Exp | Got (conf) | Decided by | Path | Clar exp→got | Raw | Ans | Result | Explanation |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cs:
            r = R.get(c["id"])
            if not r:
                L.append(f"| {c['id']} | {cell(c['input'], 70)} | {c['expected_label']} | not run | | | | | | | |")
                continue
            o, sc = r["output"], r["score"]
            alt = [a for a in c["acceptable_labels"] if a != c["expected_label"]]
            exp = c["expected_label"] + (f" ({'/'.join(alt)})" if alt else "")
            conf = f"{o['confidence']:.2f}" if isinstance(o.get("confidence"), (int, float)) else "—"
            ce = "—" if c["expected_clarification"] is None else ("yes" if c["expected_clarification"] else "no")
            cg = "yes" if o.get("clarification_pending") else "no"
            raw = (o.get("raw_classifier") or {}).get("label") or "—"
            ans = {True: "✓", False: "✗", None: "—"}[sc["answer_ok"]]
            L.append(f"| {c['id']} | {cell(c['input'], 70)} | {exp} | {o.get('predicted_label')} ({conf}) | {cell(decided_by(o.get('classifier_status')), 34)} | "
                     f"{'fast' if o.get('path') == 'FAST_PATH' else 'multi'} | {ce}→{cg} | {raw} | {ans} | {'PASS' if sc['pass'] else '**FAIL**'} | {cell(explain(c, r), 120)} |")

    L.append("\n## Per-case detail\n")
    for c in ds["pipeline_cases"]:
        r = R.get(c["id"])
        if not r:
            continue
        o = r["output"]
        L.append(f"<details><summary><b>{c['id']}</b> — {cell(c['input'], 90)} — {'PASS' if r['score']['pass'] else 'FAIL'}</summary>\n")
        L.append("```text\n" + c["input"] + "\n```\n")
        L.append(f"- **Probe:** {c.get('probe') or '—'}" + (f" · **Rationale:** {c['rationale']}" if c.get("rationale") else ""))
        L.append(f"- **Expected:** {c['expected_label']} (acceptable: {', '.join(c['acceptable_labels'])}); clarification: "
                 f"{'n/a' if c['expected_clarification'] is None else c['expected_clarification']}; answer check: `{json.dumps(c.get('answer_check'))}`")
        if "error" in o:
            L.append(f"- **Error:** `{o['error']}`\n</details>\n")
            continue
        L.append(f"- **Actual:** {o['predicted_label']} (confidence {o.get('confidence')}) · status `{cell(o.get('classifier_status'), 160)}` · path {o.get('path')} · "
                 f"HITL {o.get('clarification_decision')}/{o.get('interaction_status')} · clarity {o.get('clarity_status')} ({o.get('clarity_confidence')}) · ambiguity type {o.get('ambiguity_type')}")
        raw = o.get("raw_classifier") or {}
        if raw.get("label"):
            L.append(f"- **Raw classifier:** {raw['label']} (confidence {raw.get('confidence'):.2f})")
        L.append(f"- **Consistency signals:** clusters {o.get('cluster_count')}, entropy {o.get('semantic_entropy')}, dominant agreement {o.get('dominant_agreement')}, {len(o.get('responses') or [])} responses")
        if o.get("suggestions") is not None:
            L.append(f"- **Clarification suggestions ({o.get('suggestion_source')}):** {', '.join(o['suggestions']) or '(none)'} · gate prompt: {o.get('pipeline_prompt')}")
        L.append(f"- **Generated answer:** {cell(o.get('final_answer'), 400)}")
        L.append(f"- **Result:** {'PASS' if r['score']['pass'] else 'FAIL'} — {explain(c, r)}\n</details>\n")

    if res.get("repeats"):
        L.append("\n## Repeat runs (stochastic consistency)\n")
        L.append("| ID | Input | Run 1 | Run 2 | Run 3 | Clarification (runs) | Consistent label? |")
        L.append("|---|---|---|---|---|---|---|")
        by_id = {c["id"]: c for c in ds["pipeline_cases"]}
        for rid, runs in res["repeats"].items():
            first = R[rid]["output"]
            allr = [first] + [x["output"] for x in runs]
            labs = [str(x.get("predicted_label")) for x in allr]
            cl = ["Y" if x.get("clarification_pending") else "N" for x in allr]
            cols = [f"{x.get('predicted_label')} ({decided_by(x.get('classifier_status'))})" for x in allr] + [""] * (3 - len(allr))
            L.append(f"| {rid} | {cell(by_id[rid]['input'], 50)} | {cols[0]} | {cols[1]} | {cols[2]} | {''.join(cl)} | {'yes' if len(set(labs)) == 1 else '**no**'} |")

    if res.get("guard_results"):
        L.append("\n## J. Clarification-input guard (deterministic, `src/classification/hitl_prompts.py`)\n")
        L.append("| ID | Text | Expectation | Detected credential | Detected decline | Sanitized | Result |")
        L.append("|---|---|---|---|---|---|---|")
        for g in res["guard_results"]:
            exp = ("credential" if g.get("expect_credential") else "not credential") if "expect_credential" in g else \
                  ("decline" if g.get("expect_decline") else "not decline") if "expect_decline" in g else f"strip `{g['expect_sanitized_excludes']}`"
            L.append(f"| {g['id']} | `{cell(g['text'])}` | {exp} | {g['detected_credential']} | {g['detected_decline']} | `{cell(g['sanitized'])}` | {'PASS' if g['pass'] else '**FAIL**'} |")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
