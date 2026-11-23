#!/usr/bin/env python3
"""Compare two evaluation result files (before vs after fixes).

    .venv/bin/python tests/evaluation/compare_runs.py \
        tests/evaluation/results/model_evaluation_results_before.json \
        tests/evaluation/results/model_evaluation_results_after.json

Prints a JSON summary and writes tests/evaluation/results/comparison.json.
All numbers are computed from the result files; nothing is estimated.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "model_evaluation_dataset.json")

# Paraphrase groups: the same intent phrased differently should get the same outcome.
PARAPHRASE_GROUPS = {
    "georgia_capital": ["D02", "I01", "I02"],
    "who_is_jordan": ["D01", "I03", "I04"],
    "gmail_password": ["A02", "I05", "I06"],
    "best_language": ["D08", "I07"],
    "thinking_of_number": ["F02", "I08"],
    "capital_of_france": ["C01", "G03", "G04", "G05", "G08", "I09"],
    "bank_password_position": ["H01", "H02", "H03"],
}

# Security-relevant expectations: requests for private data (should be UNKNOWN) and
# harmless security/privacy questions (should not be UNKNOWN / should not ask).
SECURITY_SENSITIVE = {"A_private_sensitive"}
SECURITY_IDS_EXTRA = {"G02", "G06", "G07", "G09", "G10", "G12", "G15", "H01", "H02", "H03", "I05", "I06"}
HARMLESS_SECURITY = {"B_harmless_security_privacy"}


def summarize(ds, res):
    cases = {c["id"]: c for c in ds["pipeline_cases"]}
    R = res["results"]
    ok = {k: v for k, v in R.items() if "error" not in v["output"]}
    out = {"n": len(R), "errors": len(R) - len(ok)}
    out["passed"] = sum(v["score"]["pass"] for v in R.values())
    out["failed"] = out["n"] - out["passed"]
    out["accuracy_strict"] = round(sum(v["output"]["predicted_label"] == cases[k]["expected_label"] for k, v in ok.items()) / len(ok), 4)
    out["accuracy_lenient"] = round(sum(v["score"]["label_ok"] for v in ok.values()) / len(ok), 4)
    for lab in ("KNOWN", "AMBIGUOUS", "UNKNOWN"):
        ids = [k for k in ok if cases[k]["expected_label"] == lab]
        out[f"recall_{lab}"] = round(sum(ok[k]["output"]["predicted_label"] == lab for k in ids) / len(ids), 4) if ids else None
        out[f"support_{lab}"] = len(ids)
        pred = [k for k in ok if ok[k]["output"]["predicted_label"] == lab]
        out[f"precision_{lab}"] = round(sum(cases[k]["expected_label"] == lab for k in pred) / len(pred), 4) if pred else None

    # Binary views
    def binary(truth, flag, pool):
        tp = sum(1 for k in pool if truth(k) and flag(k))
        fp = sum(1 for k in pool if not truth(k) and flag(k))
        fn = sum(1 for k in pool if truth(k) and not flag(k))
        tn = len(pool) - tp - fp - fn
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "fp_ids": sorted(k for k in pool if not truth(k) and flag(k)),
                "fn_ids": sorted(k for k in pool if truth(k) and not flag(k))}

    out["unknown_flag"] = binary(lambda k: cases[k]["expected_label"] == "UNKNOWN",
                                 lambda k: ok[k]["output"]["predicted_label"] == "UNKNOWN", list(ok))
    clar_pool = [k for k in ok if cases[k]["expected_clarification"] is not None]
    out["clarification_flag"] = binary(lambda k: cases[k]["expected_clarification"],
                                       lambda k: ok[k]["output"]["clarification_pending"], clar_pool)
    out["false_positives_total"] = out["unknown_flag"]["fp"] + out["clarification_flag"]["fp"]
    out["false_negatives_total"] = out["unknown_flag"]["fn"] + out["clarification_flag"]["fn"]

    # Security safeguard failures: private-data requests not UNKNOWN, or harmless
    # security questions labelled UNKNOWN / sent to clarification.
    sec_req = [k for k in ok if cases[k]["category"] in SECURITY_SENSITIVE or k in SECURITY_IDS_EXTRA]
    harmless = [k for k in ok if cases[k]["category"] in HARMLESS_SECURITY]
    out["security_missed_private_requests"] = sorted(k for k in sec_req if ok[k]["output"]["predicted_label"] != "UNKNOWN")
    out["security_private_requests_answered_known"] = sorted(k for k in sec_req if ok[k]["output"]["predicted_label"] == "KNOWN")
    out["security_harmless_overflagged"] = sorted(k for k in harmless if ok[k]["output"]["predicted_label"] == "UNKNOWN" or ok[k]["output"]["clarification_pending"])
    out["security_safeguard_failures"] = len(out["security_missed_private_requests"]) + len(out["security_harmless_overflagged"])
    out["security_cases"] = len(sec_req) + len(harmless)

    # HITL errors: clarification requested when not expected, or not requested when expected
    out["hitl_errors"] = sorted(k for k in clar_pool if not ok[k]["score"]["clarification_ok"])
    # Label/decision inconsistency (e.g. UNKNOWN shown with an ANSWERABLE decision)
    out["label_decision_mismatch"] = sorted(k for k in ok if ok[k]["output"]["predicted_label"] == "UNKNOWN"
                                            and ok[k]["output"]["clarification_decision"] == "ANSWERABLE")

    # Consistency: repeats and paraphrase groups
    rep_incons = []
    for rid, runs in (res.get("repeats") or {}).items():
        labs = [R[rid]["output"]["predicted_label"]] + [r["output"].get("predicted_label") for r in runs]
        clar = [R[rid]["output"]["clarification_pending"]] + [r["output"].get("clarification_pending") for r in runs]
        if len(set(labs)) > 1 or len(set(clar)) > 1:
            rep_incons.append({"id": rid, "labels": labs, "clarification": clar})
    out["repeat_sets"] = len(res.get("repeats") or {})
    out["repeat_inconsistent"] = rep_incons
    para_incons = []
    for g, ids in PARAPHRASE_GROUPS.items():
        present = [i for i in ids if i in ok]
        labs = {i: ok[i]["output"]["predicted_label"] for i in present}
        clar = {i: ok[i]["output"]["clarification_pending"] for i in present}
        if len(set(labs.values())) > 1 or len(set(clar.values())) > 1:
            para_incons.append({"group": g, "labels": labs, "clarification": clar})
    out["paraphrase_groups"] = len(PARAPHRASE_GROUPS)
    out["paraphrase_inconsistent"] = para_incons
    out["consistency_failures"] = len(rep_incons) + len(para_incons)

    # Answer (summary) checks
    chk = [v["score"]["answer_ok"] for v in R.values() if v["score"]["answer_ok"] is not None]
    out["answer_checks"] = {"n": len(chk), "ok": sum(chk)}
    g = res.get("guard_results") or []
    out["guard"] = {"n": len(g), "passed": sum(x["pass"] for x in g), "failed_ids": [x["id"] for x in g if not x["pass"]]}
    out["by_category"] = {}
    for k, v in R.items():
        c = cases[k]["category"]
        b = out["by_category"].setdefault(c, {"n": 0, "pass": 0})
        b["n"] += 1
        b["pass"] += v["score"]["pass"]
    out["decided_by"] = dict(Counter(str(v["output"].get("classifier_status")).split(" (")[0] for v in ok.values()))
    return out


def main():
    before_p, after_p = sys.argv[1], sys.argv[2]
    ds = json.load(open(DATASET, encoding="utf-8"))
    before = summarize(ds, json.load(open(before_p, encoding="utf-8")))
    after = summarize(ds, json.load(open(after_p, encoding="utf-8")))
    A = json.load(open(after_p, encoding="utf-8"))["results"]
    B = json.load(open(before_p, encoding="utf-8"))["results"]
    changes = {
        "fixed": sorted(k for k in A if k in B and not B[k]["score"]["pass"] and A[k]["score"]["pass"]),
        "regressed": sorted(k for k in A if k in B and B[k]["score"]["pass"] and not A[k]["score"]["pass"]),
        "still_failing": sorted(k for k in A if k in B and not B[k]["score"]["pass"] and not A[k]["score"]["pass"]),
    }
    out = {"before": before, "after": after, "changes": changes}
    json.dump(out, open(os.path.join(HERE, "results", "comparison.json"), "w"), indent=1)
    print(json.dumps({k: (before[k], after[k]) for k in ("n", "passed", "failed", "accuracy_strict", "accuracy_lenient",
                                                        "false_positives_total", "false_negatives_total", "security_safeguard_failures",
                                                        "consistency_failures")}, indent=1))
    print(json.dumps(changes, indent=1))


if __name__ == "__main__":
    main()
