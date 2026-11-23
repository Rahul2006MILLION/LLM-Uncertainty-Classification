#!/usr/bin/env python3
"""Compare the live run with the deployed model (A) against the live run with the candidate (B).

    .venv/bin/python experiments/production_retraining/compare_live_runs.py

Reads:  tests/evaluation/results/model_evaluation_results_after.json  (A, remediated system)
        experiments/production_retraining/results/live_eval_candidate.json (B)
Writes: experiments/production_retraining/results/live_comparison.json  (nothing else)
"""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
A_PATH = os.path.join(ROOT, "tests", "evaluation", "results", "model_evaluation_results_after.json")
B_PATH = os.path.join(HERE, "results", "live_eval_candidate.json")
DS_PATH = os.path.join(ROOT, "tests", "evaluation", "model_evaluation_dataset.json")
RET_PATH = os.path.join(HERE, "results", "retraining_results.json")
OUT = os.path.join(HERE, "results", "live_comparison.json")

spec = importlib.util.spec_from_file_location("compare_runs", os.path.join(ROOT, "tests", "evaluation", "compare_runs.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)  # only summarize() is used; its main() (which writes a file) is not called


def subset(res, ids):
    out = dict(res)
    out["results"] = {k: v for k, v in res["results"].items() if k in ids}
    out["repeats"] = {k: v for k, v in (res.get("repeats") or {}).items() if k in ids}
    return out


def main():
    ds = json.load(open(DS_PATH, encoding="utf-8"))
    A = json.load(open(A_PATH, encoding="utf-8"))
    B = json.load(open(B_PATH, encoding="utf-8"))
    ret = json.load(open(RET_PATH, encoding="utf-8"))
    overlap = set(ret["benchmark_107_not_independent"]["overlapping_cases_excluded_in_clean_subset"])
    all_ids = {c["id"] for c in ds["pipeline_cases"]}
    clean = all_ids - overlap

    out = {"A": cr.summarize(ds, A), "B": cr.summarize(ds, B)}
    ds_clean = dict(ds)
    ds_clean["pipeline_cases"] = [c for c in ds["pipeline_cases"] if c["id"] in clean]
    out["A_clean"] = cr.summarize(ds_clean, subset(A, clean))
    out["B_clean"] = cr.summarize(ds_clean, subset(B, clean))
    RA, RB = A["results"], B["results"]
    out["changes"] = {
        "fixed": sorted(k for k in RB if not RA[k]["score"]["pass"] and RB[k]["score"]["pass"]),
        "regressed": sorted(k for k in RB if RA[k]["score"]["pass"] and not RB[k]["score"]["pass"]),
        "still_failing": sorted(k for k in RB if not RA[k]["score"]["pass"] and not RB[k]["score"]["pass"]),
        "clarification_decision_changed": sorted(
            (k, RA[k]["output"]["clarification_decision"], RB[k]["output"]["clarification_decision"])
            for k in RB if RA[k]["output"]["clarification_pending"] != RB[k]["output"]["clarification_pending"]),
        "label_changed": sorted((k, RA[k]["output"]["predicted_label"], RB[k]["output"]["predicted_label"])
                                for k in RB if RA[k]["output"]["predicted_label"] != RB[k]["output"]["predicted_label"]),
    }
    out["overlap_ids"] = sorted(overlap)
    out["B_meta"] = B.get("meta")
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    keys = ("n", "passed", "failed", "accuracy_strict", "accuracy_lenient", "false_positives_total",
            "false_negatives_total", "security_safeguard_failures", "consistency_failures")
    for k in keys:
        print(f"{k:32} A={out['A'][k]!s:8} B={out['B'][k]!s:8} | clean A={out['A_clean'][k]!s:8} B={out['B_clean'][k]!s}")
    print(json.dumps(out["changes"], indent=1))


if __name__ == "__main__":
    main()
