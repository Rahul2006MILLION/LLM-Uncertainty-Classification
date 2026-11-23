#!/usr/bin/env python3
"""Behavioural evaluation of the uncertainty pipeline exactly as the web UI serves it.

Evaluation only: this script imports and calls existing code; it changes nothing.

* Pipeline cases go through ``webapp/server.py::analyze`` (the same call the UI makes),
  which runs ``experiments/ask.py::run_pipeline`` with its HITL gates enabled.
* For multi-response runs the trained classifier is re-applied (read-only) to the same
  extracted features so a wrong final label can be attributed to the ML model or to a
  rule-based override.
* Guard cases exercise the deterministic clarification-input safeguards in
  ``src/classification/hitl_prompts.py``.

Usage (from the project root, Ollama running):

    .venv/bin/python tests/evaluation/run_model_evaluation.py
    .venv/bin/python tests/evaluation/run_model_evaluation.py --only D01,A08 --no-repeats
    .venv/bin/python tests/evaluation/run_model_evaluation.py --resume

Results: tests/evaluation/results/model_evaluation_results.json (overwritten per full run).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import webapp.server as server  # noqa: E402  (chdirs to ROOT, loads ask.py)
from src.classification.classifier import UncertaintyClassifier  # noqa: E402
from src.classification.hitl_prompts import (  # noqa: E402
    contains_sensitive_credentials,
    is_user_declining,
    sanitize_user_input,
)

LABELS = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
REFUSAL_RE = re.compile(
    r"(can(no|')t|unable to|not able to|do(n't| not) have (access|the ability|any (way|information))|"
    r"no (way|method|record|information) to|not possible|impossible to|i do(n't| not) know|"
    r"cannot be (predicted|known|determined)|can't (predict|know)|cannot predict|no one (can|knows)|"
    r"nobody (can|knows)|not (publicly )?(available|recorded|known)|unrecorded|as an ai|i'm sorry|"
    r"i am sorry|speculat|uncertain|unpredictable|private|personal information|not (have|privy))",
    re.IGNORECASE,
)


def answer_check(check: Optional[Dict[str, Any]], answer: str) -> Optional[bool]:
    if not check:
        return None
    text = (answer or "").lower()
    if check["type"] == "refusal":
        return bool(REFUSAL_RE.search(text))
    if check["type"] == "contains_any":
        for v in check["values"]:
            v = v.lower()
            if len(v) <= 3:
                if re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text):
                    return True
            elif v in text:
                return True
        return False
    raise ValueError(check)


def run_case(case: Dict[str, Any], clf: Optional[UncertaintyClassifier]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        res = server.analyze(case["input"], False)
    except Exception as e:
        return {"error": f"{e.__class__.__name__}: {e}", "traceback": traceback.format_exc(), "wall_seconds": round(time.time() - t0, 2)}
    r = res["record"]
    raw = None
    if clf is not None and r.get("path") == "MULTI_RESPONSE" and r.get("features"):
        try:
            raw = clf.predict(r["features"])
            raw = {"label": raw.get("predicted_label"), "confidence": raw.get("confidence"), "probabilities": raw.get("probabilities")}
        except Exception as e:
            raw = {"error": str(e)}
    clar = res.get("clarification")
    return {
        "wall_seconds": round(time.time() - t0, 2),
        "predicted_label": r.get("predicted_label"),
        "confidence": r.get("confidence"),
        "classifier_status": r.get("classifier_status"),
        "path": r.get("path"),
        "clarification_decision": r.get("clarification_decision"),
        "interaction_status": r.get("interaction_status"),
        "clarification_pending": server._needs_clarification(r),
        "clarity_status": r.get("clarity_status"),
        "clarity_confidence": r.get("clarity_confidence"),
        "ambiguity_type": r.get("ambiguity_type"),
        "clarification_reason": r.get("clarification_reason"),
        "final_answer": r.get("final_answer"),
        "cluster_count": r.get("cluster_count"),
        "semantic_entropy": r.get("semantic_entropy"),
        "dominant_agreement": r.get("dominant_cluster_agreement"),
        "raw_classifier": raw,
        "suggestions": [s.get("label") for s in (clar or {}).get("suggestions", {}).get("items", [])] if clar else None,
        "suggestion_source": (clar or {}).get("suggestions", {}).get("source") if clar else None,
        "pipeline_prompt": (clar or {}).get("pipeline_prompt") if clar else None,
        "responses": [x.get("text") for x in r.get("responses") or []],
        "cluster_labels": (r.get("clusters") or {}).get("labels"),
    }


def score(case: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in out:
        return {"label_ok": False, "label_strict_ok": False, "clarification_ok": False, "answer_ok": None, "pass": False}
    pred = out["predicted_label"]
    label_ok = pred in case["acceptable_labels"]
    exp_c = case["expected_clarification"]
    clar_ok = True if exp_c is None else (out["clarification_pending"] == exp_c)
    return {
        "label_ok": label_ok,
        "label_strict_ok": pred == case["expected_label"],
        "clarification_ok": clar_ok,
        "answer_ok": answer_check(case.get("answer_check"), out.get("final_answer") or ""),
        "pass": label_ok and clar_ok,
    }


def prf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    div = lambda a, b: round(a / b, 4) if b else None
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else (0.0 if p == 0 or r == 0 else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1,
            "false_positive_rate": div(fp, fp + tn), "false_negative_rate": div(fn, fn + tp)}


def metrics(cases: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
    rows = [(c, results[c["id"]]) for c in cases if c["id"] in results]
    ok = [(c, r) for c, r in rows if "error" not in r["output"]]
    m: Dict[str, Any] = {"n_cases": len(rows), "n_errors": len(rows) - len(ok)}
    m["pass_rate"] = round(sum(r["score"]["pass"] for _, r in rows) / len(rows), 4) if rows else None
    m["accuracy_strict"] = round(sum(r["score"]["label_strict_ok"] for _, r in ok) / len(ok), 4) if ok else None
    m["accuracy_lenient"] = round(sum(r["score"]["label_ok"] for _, r in ok) / len(ok), 4) if ok else None

    conf = {e: Counter() for e in LABELS}
    for c, r in ok:
        conf[c["expected_label"]][str(r["output"]["predicted_label"])] += 1
    m["confusion_matrix_expected_by_predicted"] = {k: dict(v) for k, v in conf.items()}
    per_class = {}
    for lab in LABELS:
        tp = sum(1 for c, r in ok if c["expected_label"] == lab and r["output"]["predicted_label"] == lab)
        fp = sum(1 for c, r in ok if c["expected_label"] != lab and r["output"]["predicted_label"] == lab)
        fn = sum(1 for c, r in ok if c["expected_label"] == lab and r["output"]["predicted_label"] != lab)
        tn = len(ok) - tp - fp - fn
        per_class[lab] = {**prf(tp, fp, fn, tn), "support": tp + fn}
    m["per_class_strict"] = per_class
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    m["macro_f1_strict"] = round(sum(f1s) / len(f1s), 4) if len(f1s) == 3 else None

    # Binary view 1: "inaccessible / sensitive" flag = predicted UNKNOWN
    tp = sum(1 for c, r in ok if c["expected_label"] == "UNKNOWN" and r["output"]["predicted_label"] == "UNKNOWN")
    fp = sum(1 for c, r in ok if c["expected_label"] != "UNKNOWN" and r["output"]["predicted_label"] == "UNKNOWN")
    fn = sum(1 for c, r in ok if c["expected_label"] == "UNKNOWN" and r["output"]["predicted_label"] != "UNKNOWN")
    m["unknown_flag"] = prf(tp, fp, fn, len(ok) - tp - fp - fn)

    # Binary view 2: clarification requested (HITL pending)
    cl = [(c, r) for c, r in ok if c["expected_clarification"] is not None]
    tp = sum(1 for c, r in cl if c["expected_clarification"] and r["output"]["clarification_pending"])
    fp = sum(1 for c, r in cl if not c["expected_clarification"] and r["output"]["clarification_pending"])
    fn = sum(1 for c, r in cl if c["expected_clarification"] and not r["output"]["clarification_pending"])
    m["clarification_flag"] = prf(tp, fp, fn, len(cl) - tp - fp - fn)

    # Raw ML classifier vs final label (multi-response cases only)
    raw = [(c, r) for c, r in ok if (r["output"].get("raw_classifier") or {}).get("label")]
    m["raw_classifier"] = {
        "n_multi_response": len(raw),
        "accuracy_strict": round(sum(r["output"]["raw_classifier"]["label"] == c["expected_label"] for c, r in raw) / len(raw), 4) if raw else None,
        "final_accuracy_strict_same_cases": round(sum(r["output"]["predicted_label"] == c["expected_label"] for c, r in raw) / len(raw), 4) if raw else None,
        "overrides": sum(r["output"]["raw_classifier"]["label"] != r["output"]["predicted_label"] for c, r in raw),
        "overrides_that_fixed": sum(r["output"]["raw_classifier"]["label"] != c["expected_label"] and r["output"]["predicted_label"] == c["expected_label"] for c, r in raw),
        "overrides_that_broke": sum(r["output"]["raw_classifier"]["label"] == c["expected_label"] and r["output"]["predicted_label"] != c["expected_label"] for c, r in raw),
    }

    by_cat = defaultdict(lambda: {"n": 0, "pass": 0, "label_ok": 0, "clar_ok": 0, "answer_checked": 0, "answer_ok": 0})
    for c, r in rows:
        b = by_cat[c["category"]]
        b["n"] += 1
        b["pass"] += r["score"]["pass"]
        b["label_ok"] += r["score"]["label_ok"]
        b["clar_ok"] += r["score"]["clarification_ok"]
        if r["score"]["answer_ok"] is not None:
            b["answer_checked"] += 1
            b["answer_ok"] += r["score"]["answer_ok"]
    m["by_category"] = dict(sorted(by_cat.items()))

    checked = [r for _, r in rows if r["score"]["answer_ok"] is not None]
    m["answer_checks"] = {"n": len(checked), "ok": sum(r["score"]["answer_ok"] for r in checked)}
    m["paths"] = dict(Counter(r["output"].get("path") for _, r in ok))
    m["decided_by"] = dict(Counter(re.sub(r"\s*\(.*", "", str(r["output"].get("classifier_status"))) for _, r in ok))
    return m


def run_guard(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for g in cases:
        t = g["text"]
        row = {**g, "detected_credential": contains_sensitive_credentials(t), "detected_decline": is_user_declining(t),
               "sanitized": sanitize_user_input(t)}
        if "expect_credential" in g:
            row["pass"] = row["detected_credential"] == g["expect_credential"]
        elif "expect_decline" in g:
            row["pass"] = row["detected_decline"] == g["expect_decline"]
        else:
            row["pass"] = g["expect_sanitized_excludes"].lower() not in row["sanitized"].lower()
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(HERE, "model_evaluation_dataset.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "model_evaluation_results.json"))
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--no-repeats", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip cases already present in --out")
    args = ap.parse_args()

    ds = json.load(open(args.dataset, encoding="utf-8"))
    cases = ds["pipeline_cases"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    state: Dict[str, Any] = {"results": {}, "repeats": {}}
    if args.resume and os.path.exists(args.out):
        state = json.load(open(args.out, encoding="utf-8"))

    try:
        clf = UncertaintyClassifier.load(server.ask.load_config(server.CONFIG_PATH)["classifier"]["model_save_path"])
    except Exception as e:  # pragma: no cover
        print("WARNING: could not load classifier for raw re-scoring:", e)
        clf = None

    cfg = server.ask.load_config(server.CONFIG_PATH)
    state["meta"] = {
        "started": state.get("meta", {}).get("started") or datetime.datetime.now().isoformat(timespec="seconds"),
        "generation_model": cfg["ollama"]["generation_model"], "embedding_model": cfg["ollama"]["embedding_model"],
        "temperature": cfg["ollama"]["temperature"], "multi_response_count": cfg["pipeline"]["multi_response_count"],
        "entrypoint": "webapp/server.py::analyze(question, force_multi=False)",
    }

    def save():
        state["meta"]["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        tmp = args.out + ".tmp"
        json.dump(state, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        os.replace(tmp, args.out)

    for i, c in enumerate(cases, 1):
        if c["id"] in state["results"]:
            continue
        out = run_case(c, clf)
        sc = score(c, out)
        state["results"][c["id"]] = {"output": out, "score": sc}
        print(f"[{i:3}/{len(cases)}] {c['id']:4} {'PASS' if sc['pass'] else 'FAIL'} "
              f"exp={c['expected_label']:9} got={str(out.get('predicted_label')):9} "
              f"clar={out.get('clarification_pending')} ans={sc['answer_ok']} {out.get('wall_seconds')}s"
              + (f" ERROR {out['error']}" if 'error' in out else ""), flush=True)
        save()

    if not args.no_repeats:
        by_id = {c["id"]: c for c in cases}
        for rid in ds.get("repeat_ids", []):
            if rid not in by_id:
                continue
            runs = state["repeats"].setdefault(rid, [])
            while len(runs) < ds.get("repeats", 2):
                out = run_case(by_id[rid], clf)
                runs.append({"output": {k: out.get(k) for k in ("predicted_label", "confidence", "classifier_status", "path", "clarification_decision", "interaction_status", "clarification_pending", "cluster_count", "semantic_entropy", "final_answer", "raw_classifier", "error")},
                             "score": score(by_id[rid], out)})
                print(f"  repeat {rid} #{len(runs)} got={out.get('predicted_label')} clar={out.get('clarification_pending')}", flush=True)
                save()

    state["guard_results"] = run_guard(ds.get("guard_cases", []))
    state["metrics"] = metrics(ds["pipeline_cases"], state["results"])
    save()
    m = state["metrics"]
    print(json.dumps({k: m[k] for k in ("n_cases", "n_errors", "pass_rate", "accuracy_strict", "accuracy_lenient", "macro_f1_strict")}, indent=1))
    print("guard:", sum(g["pass"] for g in state["guard_results"]), "/", len(state["guard_results"]))


if __name__ == "__main__":
    main()
