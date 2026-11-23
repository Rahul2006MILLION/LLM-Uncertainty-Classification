#!/usr/bin/env python3
"""Controlled retraining experiment: deployed (legacy-trained) classifier vs. a candidate
trained on the project's intended 300-record production dataset.

Isolated experiment — it does NOT modify the production model, the feature pipeline, the
safeguard/HITL layer, the datasets, labels, or the 107-case evaluation benchmark.

* Candidate model  -> models/candidates/uncertainty_classifier_f300_candidate.joblib
* Results          -> experiments/production_retraining/results/

Run from the project root (Ollama needed only for the embedding-based overlap check and for
recomputing benchmark features from the recorded responses):

    .venv/bin/python experiments/production_retraining/run_retraining_experiment.py
"""

from __future__ import annotations

import collections
import importlib.util
import json
import math
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold  # noqa: E402

from src.classification.classifier import ID_TO_LABEL, LABEL_TO_ID, UncertaintyClassifier  # noqa: E402
from src.features.feature_extractor import FEATURE_NAMES as FN  # noqa: E402

DEV_PATH = "data/features/features_300_v1.1.jsonl"
LEGACY_PATH = "data/features.jsonl"
PROD_MODEL = "models/uncertainty_classifier.joblib"
BENCH_DATASET = "tests/evaluation/model_evaluation_dataset.json"
BENCH_RESULTS = "tests/evaluation/results/model_evaluation_results_after.json"  # remediated system run
CANDIDATE_PATH = "models/candidates/uncertainty_classifier_f300_candidate.joblib"
OUT_DIR = os.path.join(HERE, "results")
SEED = 42                       # project default (scripts/generate_gold_splits.py)
REPEAT_SEEDS = [42, 43, 44, 45, 46]
EMB_NEAR_DUP = 0.90             # cosine threshold for semantic near-duplicate review
L = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
CONSISTENCY = ["semantic_entropy", "normalized_entropy", "dominant_agreement", "second_agreement", "agreement_margin",
               "cluster_count", "cluster_ratio", "mean_pairwise_similarity", "min_pairwise_similarity"]
LINGUISTIC = [f for f in FN if f not in CONSISTENCY]

# The project's own fold generator, imported unchanged
_spec = importlib.util.spec_from_file_location("gold_splits", os.path.join(ROOT, "scripts", "generate_gold_splits.py"))
gold_splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gold_splits)
normq = gold_splits.normalize_question_text


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def label_of(r):
    return r.get("gold_label") or r.get("label")


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "per_class": {L[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
        "unknown_recall": float(r[2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def mean_std(dicts, keys=("accuracy", "macro_f1", "weighted_f1", "unknown_recall")):
    out = {k: {"mean": float(np.mean([d[k] for d in dicts])), "std": float(np.std([d[k] for d in dicts]))} for k in keys}
    for c in L:
        out[f"recall_{c}"] = {"mean": float(np.mean([d["per_class"][c]["recall"] for d in dicts])),
                              "std": float(np.std([d["per_class"][c]["recall"] for d in dicts]))}
    return out


def train_candidate(X, y):
    """Existing classifier implementation with its default (production) hyperparameters."""
    clf = UncertaintyClassifier(model_type="xgboost")
    clf.train(X, y, feature_names=FN)
    return clf


# ------------------------------------------------------------------------------------------
# PHASE 1 — data validation and leakage exclusion
# ------------------------------------------------------------------------------------------
def validate(dev, bench_cases):
    v = {"path": DEV_PATH, "records": len(dev)}
    ids = [r.get("id") for r in dev]
    v["duplicate_ids"] = [i for i, c in collections.Counter(ids).items() if c > 1]
    qn = collections.Counter(normq(r["question"]) for r in dev)
    v["duplicate_questions"] = [q for q, c in qn.items() if c > 1]
    v["label_distribution"] = dict(collections.Counter(label_of(r) for r in dev))
    v["label_field_mismatch"] = [r["id"] for r in dev if r.get("label") and r.get("gold_label") and r["label"] != r["gold_label"]]
    v["invalid_labels"] = [r["id"] for r in dev if label_of(r) not in LABEL_TO_ID]
    bad = []
    for r in dev:
        f = r.get("features") or {}
        if set(f) != set(FN):
            bad.append({"id": r["id"], "issue": "schema", "missing": sorted(set(FN) - set(f)), "extra": sorted(set(f) - set(FN))})
        for n in FN:
            x = f.get(n)
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                bad.append({"id": r["id"], "issue": f"invalid value {n}={x!r}"})
            elif n in ("normalized_entropy", "dominant_agreement", "second_agreement", "agreement_margin", "cluster_ratio",
                       "distinct_1gram_ratio", "distinct_2gram_ratio", "question_overlap_mean") and not (-1e-9 <= x <= 1 + 1e-9):
                bad.append({"id": r["id"], "issue": f"out of range {n}={x}"})
    v["feature_issues"] = bad
    v["schema_ok"] = all(set((r.get("features") or {})) == set(FN) for r in dev) and len(FN) == 16
    v["groups"] = len({r["question_group_id"] for r in dev})
    v["categories"] = dict(collections.Counter(r["category"] for r in dev))
    v["similarity_thresholds"] = dict(collections.Counter(r.get("similarity_threshold") for r in dev))

    # Overlap with the 107-case benchmark: exact (normalised) and semantic near-duplicates
    exact = []
    for c in bench_cases:
        for r in dev:
            if normq(c["input"]) == normq(r["question"]):
                exact.append({"bench_id": c["id"], "dev_id": r["id"], "group": r["question_group_id"], "question": r["question"]})
    near = []
    try:
        from src.generation.ollama_client import OllamaClient
        from src.preprocessing.embeddings import EmbeddingEngine
        emb = EmbeddingEngine(client=OllamaClient(), embedding_model="all-minilm:latest")
        Eb = emb.embed_batch([c["input"] for c in bench_cases])
        Ed = emb.embed_batch([r["question"] for r in dev])
        S = Eb @ Ed.T
        for i, c in enumerate(bench_cases):
            for j in np.where(S[i] >= EMB_NEAR_DUP)[0]:
                if normq(c["input"]) != normq(dev[j]["question"]):
                    near.append({"bench_id": c["id"], "bench_q": c["input"][:80], "dev_id": dev[j]["id"], "dev_q": dev[j]["question"],
                                 "group": dev[j]["question_group_id"], "cosine": round(float(S[i, j]), 3)})
        v["near_duplicate_check"] = f"all-minilm cosine >= {EMB_NEAR_DUP}"
    except Exception as e:  # pragma: no cover
        v["near_duplicate_check"] = f"skipped: {e}"
    v["overlap_exact"] = exact
    v["overlap_near"] = near
    # Exclusion rule: remove every record in a question group that contains an exact or
    # semantic near-duplicate of a benchmark question (groups may hold paraphrase variants).
    excl_groups = sorted({o["group"] for o in exact} | {o["group"] for o in near})
    v["excluded_groups"] = excl_groups
    v["excluded_records"] = [{"id": r["id"], "question": r["question"], "label": label_of(r), "group": r["question_group_id"],
                              "reason": "question group overlaps the 107-case benchmark (exact or semantic near-duplicate)"}
                             for r in dev if r["question_group_id"] in excl_groups]
    # Benchmark cases whose question (or paraphrase) is in the excluded groups: contaminated for the
    # deployed model's comparison only if the deployed model saw them; recorded for transparency.
    v["benchmark_cases_overlapping_dev"] = sorted({o["bench_id"] for o in exact} | {o["bench_id"] for o in near})
    return v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev_all = rows(DEV_PATH)
    legacy = rows(LEGACY_PATH)
    bench = json.load(open(BENCH_DATASET, encoding="utf-8"))["pipeline_cases"]
    results = {"protocol": {
        "dev_data": DEV_PATH, "cv": "scripts/generate_gold_splits.py::FiveFoldCrossValidator (5-fold StratifiedGroupKFold, group=question_group_id, seed 42)",
        "repeat_seeds": REPEAT_SEEDS, "classifier": "src.classification.classifier.UncertaintyClassifier(model_type='xgboost') defaults",
        "hyperparameters": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "random_state": 42, "eval_metric": "mlogloss"},
        "tuning": "none"}}

    # ---- Phase 1
    val = validate(dev_all, bench)
    excl = set(val["excluded_groups"])
    dev = [r for r in dev_all if r["question_group_id"] not in excl]
    val["records_after_exclusion"] = len(dev)
    val["label_distribution_after_exclusion"] = dict(collections.Counter(label_of(r) for r in dev))
    results["data_validation"] = val
    print(f"[phase1] {len(dev_all)} records -> {len(dev)} after excluding {len(excl)} overlapping groups", flush=True)

    X = np.array([[r["features"][n] for n in FN] for r in dev], dtype=np.float32)
    y = np.array([LABEL_TO_ID[label_of(r)] for r in dev])
    G = np.array([r["question_group_id"] for r in dev])
    CAT = np.array([r["category"] for r in dev])
    ids = [r["id"] for r in dev]

    prod = UncertaintyClassifier.load(PROD_MODEL)
    legacy_q = {normq(r["question"]) for r in legacy}
    seen_by_prod = np.array([normq(r["question"]) in legacy_q for r in dev])   # deployed model may have trained on these
    results["deployed_model"] = {"path": PROD_MODEL, "training_source": LEGACY_PATH,
                                 "metadata_n_samples": prod.metadata.get("n_samples"),
                                 "dev_records_also_in_legacy": int(seen_by_prod.sum())}

    # ---- Phase 2/3: grouped CV with the project's fold generator (seed 42) + repeated seeds
    def cv_run(seed):
        recs = [dict(id=ids[i], question_group_id=G[i], final_gold_label=label_of(dev[i])) for i in range(len(dev))]
        folds = gold_splits.FiveFoldCrossValidator(n_splits=5, seed=seed).split(recs)["folds"]
        idx = {i: k for k, i in enumerate(ids)}
        oof_b = np.zeros((len(dev), 3)); per_fold = []
        for f in folds:
            tr = np.array([idx[i] for i in f["train_ids"]]); te = np.array([idx[i] for i in f["val_ids"]])
            cand = train_candidate(X[tr], y[tr])
            pb = cand.model.predict_proba(X[te]); oof_b[te] = pb
            pa = prod.model.predict_proba(X[te])
            fair = ~seen_by_prod[te]          # identical records for A and B, excluding anything A may have seen
            per_fold.append({"fold": f["fold_index"], "n_test": int(len(te)), "n_test_fair": int(fair.sum()),
                             "B_candidate": metrics(y[te], pb.argmax(1)),
                             "A_deployed_fair": metrics(y[te][fair], pa[fair].argmax(1)),
                             "B_candidate_fair": metrics(y[te][fair], pb[fair].argmax(1))})
        return oof_b, per_fold

    oof42, folds42 = cv_run(SEED)
    pa_all = prod.model.predict_proba(X)
    fair_all = ~seen_by_prod
    results["cv_seed42"] = {
        "per_fold": folds42,
        "fold_variance": {"B_candidate": mean_std([f["B_candidate"] for f in folds42]),
                          "A_deployed_fair": mean_std([f["A_deployed_fair"] for f in folds42]),
                          "B_candidate_fair": mean_std([f["B_candidate_fair"] for f in folds42])},
        "pooled_oof": {"B_candidate": metrics(y, oof42.argmax(1)),
                       "A_deployed_fair": metrics(y[fair_all], pa_all[fair_all].argmax(1)),
                       "B_candidate_fair": metrics(y[fair_all], oof42[fair_all].argmax(1))},
    }
    # Paired significance on identical records (seed-42 OOF): exact McNemar + bootstrap CI of the
    # macro-F1 difference (group-level resampling would be stricter; records are resampled here).
    from scipy.stats import binomtest
    ya, yb = pa_all[fair_all].argmax(1), oof42[fair_all].argmax(1); yt = y[fair_all]
    b_only = int(np.sum((yb == yt) & (ya != yt))); a_only = int(np.sum((ya == yt) & (yb != yt)))
    rng = np.random.RandomState(SEED); diffs = []
    for _ in range(2000):
        ix = rng.randint(0, len(yt), len(yt))
        diffs.append(f1_score(yt[ix], yb[ix], average="macro") - f1_score(yt[ix], ya[ix], average="macro"))
    results["cv_seed42"]["significance_B_vs_A_fair"] = {
        "n": int(len(yt)), "B_correct_A_wrong": b_only, "A_correct_B_wrong": a_only,
        "mcnemar_exact_p": float(binomtest(b_only, b_only + a_only, 0.5).pvalue) if b_only + a_only else 1.0,
        "folds_B_better": int(sum(f["B_candidate_fair"]["macro_f1"] > f["A_deployed_fair"]["macro_f1"] for f in folds42)),
        "macro_f1_diff_bootstrap_95ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    rep = []
    for s in REPEAT_SEEDS:
        oof, _ = cv_run(s); rep.append(metrics(y, oof.argmax(1)))
    results["cv_repeated_seeds"] = {"seeds": REPEAT_SEEDS, "per_seed": rep, "summary": mean_std(rep)}
    print("[phase3] cross-validation done", flush=True)

    # ---- Phase 4: category-held-out, per-category, small categories
    loco_b = np.zeros((len(dev), 3))
    for tr, te in LeaveOneGroupOut().split(X, y, CAT):
        loco_b[te] = train_candidate(X[tr], y[tr]).model.predict_proba(X[te])
    per_cat = {}
    for c in sorted(set(CAT)):
        m = CAT == c; mf = m & fair_all
        per_cat[c] = {"n": int(m.sum()), "label": collections.Counter(label_of(dev[i]) for i in np.where(m)[0]).most_common(1)[0][0],
                      "cv_accuracy_B": float(accuracy_score(y[m], oof42[m].argmax(1))),
                      "loco_accuracy_B": float(accuracy_score(y[m], loco_b[m].argmax(1))),
                      "deployed_accuracy_A_fair": float(accuracy_score(y[mf], pa_all[mf].argmax(1))) if mf.sum() else None,
                      "n_fair": int(mf.sum())}
    results["category_held_out"] = {"B_candidate": metrics(y, loco_b.argmax(1)),
                                    "A_deployed_fair_same_records": metrics(y[fair_all], pa_all[fair_all].argmax(1))}
    results["per_category"] = per_cat
    print("[phase4] category analysis done", flush=True)

    # ---- Final candidate on all retained development records; save separately
    cand = train_candidate(X, y)
    cand.metadata.update({"experiment": "production_retraining", "training_data": DEV_PATH,
                          "excluded_groups": sorted(excl), "promoted": False,
                          "note": "Candidate only. NOT the production model."})
    cand.save(CANDIDATE_PATH)
    results["candidate_model"] = {"path": CANDIDATE_PATH, "n_train": int(len(y)),
                                  "train_accuracy": float(accuracy_score(y, cand.model.predict(X)))}

    # ---- Phase 5: feature analysis
    def gain(model):
        g = model.get_booster().get_score(importance_type="gain"); t = sum(g.values()) or 1
        out = {n: 0.0 for n in FN}
        for k, val_ in g.items():
            n = FN[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
            out[n] = val_ / t
        return out
    import xgboost as xgb
    def mean_abs_contrib(model, Xs):
        c = model.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)   # (n, 3, 17)
        a = np.abs(c[:, :, :16]).mean(axis=(0, 1)); return {n: float(a[i] / a.sum()) for i, n in enumerate(FN)}
    gA, gB = gain(prod.model), gain(cand.model)
    shA, shB = mean_abs_contrib(prod.model, X[fair_all]), mean_abs_contrib(cand.model, X)
    perm = permutation_importance(cand.model, X, y, scoring="f1_macro", n_repeats=20, random_state=SEED)
    # Grouped-CV permutation importance (held-out folds) for the candidate
    pi_cv = np.zeros(16); rng = np.random.RandomState(SEED)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, G):
        m = train_candidate(X[tr], y[tr]).model; base = f1_score(y[te], m.predict(X[te]), average="macro")
        for j in range(16):
            drops = []
            for _ in range(10):
                Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j]); drops.append(base - f1_score(y[te], m.predict(Xp), average="macro"))
            pi_cv[j] += np.mean(drops) / 5
    def subset_cv(cols):
        oof = np.zeros(len(y), dtype=int)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, G):
            sub = UncertaintyClassifier(model_type="xgboost"); sub.train(X[tr][:, cols], y[tr], feature_names=[FN[c] for c in cols])
            oof[te] = sub.model.predict(X[te][:, cols])
        return metrics(y, oof)
    ci = [FN.index(f) for f in CONSISTENCY]; li = [FN.index(f) for f in LINGUISTIC]
    results["feature_analysis"] = {
        "gain_A_deployed": gA, "gain_B_candidate": gB,
        "mean_abs_contribution_A": shA, "mean_abs_contribution_B": shB,
        "group_share": {k: {"consistency": sum(d[f] for f in CONSISTENCY), "linguistic": sum(d[f] for f in LINGUISTIC)}
                        for k, d in [("gain_A", gA), ("gain_B", gB), ("contrib_A", shA), ("contrib_B", shB)]},
        "permutation_importance_B_cv_macro_f1_drop": {n: float(pi_cv[i]) for i, n in enumerate(FN)},
        "permutation_importance_B_train_macro_f1_drop": {n: float(perm.importances_mean[i]) for i, n in enumerate(FN)},
        "ablation_cv_consistency_only": subset_cv(ci), "ablation_cv_linguistic_only": subset_cv(li),
    }
    print("[phase5] feature analysis done", flush=True)

    # ---- 107-case benchmark (NOT an independent test set): raw classifier and label-level replay
    bench_out = {"note": "The 107-case benchmark was used to develop the safeguard/HITL layer and overlaps the "
                         "development data; it is NOT an independent test set."}
    try:
        from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
        from src.classification.inaccessibility_assessor import InaccessibilityAssessor
        from src.features.clustering import SemanticClusterer
        from src.features.feature_extractor import FeatureExtractor
        from src.generation.ollama_client import OllamaClient
        from src.preprocessing.embeddings import EmbeddingEngine
        emb = EmbeddingEngine(client=OllamaClient(), embedding_model="all-minilm:latest")
        cl = SemanticClusterer(similarity_threshold=0.80, linkage="average"); fx = FeatureExtractor(clusterer=cl)
        R = json.load(open(BENCH_RESULTS, encoding="utf-8"))["results"]
        bcases = {c["id"]: c for c in bench}
        overl = set(val["benchmark_cases_overlapping_dev"]) | {c["id"] for c in bench if normq(c["input"]) in legacy_q}
        rowsb, reproduced = [], 0
        for k, v_ in R.items():
            o = v_["output"]; exp = LABEL_TO_ID[bcases[k]["expected_label"]]
            if o.get("path") != "MULTI_RESPONSE" or not o.get("responses"):
                rowsb.append({"id": k, "path": "FAST_PATH", "exp": exp, "sysA": o["predicted_label"], "sysB": o["predicted_label"]}); continue
            resp = o["responses"]; Eb = emb.embed_batch(resp); cr = cl.cluster_responses(resp, Eb)
            f = fx.extract_features(question=bcases[k]["input"], responses=resp, embeddings=Eb, cluster_result=cr)
            xa = np.array([[f[n] for n in FN]], dtype=np.float32)
            la, lb = ID_TO_LABEL[int(prod.model.predict(xa)[0])], ID_TO_LABEL[int(cand.model.predict(xa)[0])]
            reproduced += int(la == (o.get("raw_classifier") or {}).get("label"))
            def replay(lbl):   # same override order as experiments/ask.py
                if CriterionAmbiguityAssessor.assess(question=bcases[k]["input"], responses=resp, features=f).is_ambiguous and lbl != "AMBIGUOUS":
                    lbl = "AMBIGUOUS"
                if InaccessibilityAssessor.assess(question=bcases[k]["input"], responses=resp, features=f).is_inaccessible:
                    lbl = "UNKNOWN"
                return lbl
            rowsb.append({"id": k, "path": "MULTI_RESPONSE", "exp": exp, "rawA": la, "rawB": lb, "sysA": replay(la), "sysB": replay(lb),
                          "recorded_final": o["predicted_label"], "overlap": k in overl})
        multi = [r for r in rowsb if r["path"] == "MULTI_RESPONSE"]
        clean = [r for r in multi if not r["overlap"]]
        acc_lab = lambda rs, key: metrics([r["exp"] for r in rs], [LABEL_TO_ID[r[key]] for r in rs])
        bench_out.update({
            "replay_reproduces_recorded_raw_labels": f"{reproduced}/{len(multi)}",
            "replay_reproduces_recorded_final_labels_A": f"{sum(r['sysA'] == r['recorded_final'] for r in multi)}/{len(multi)}",
            "overlapping_cases_excluded_in_clean_subset": sorted(overl),
            "raw_classifier_multi_response": {"A": acc_lab(multi, "rawA"), "B": acc_lab(multi, "rawB")},
            "raw_classifier_multi_response_clean": {"A": acc_lab(clean, "rawA"), "B": acc_lab(clean, "rawB")},
            "system_label_replay_all107": {"A": acc_lab(rowsb, "sysA"), "B": acc_lab(rowsb, "sysB")},
            "system_label_replay_clean": {"A": acc_lab([r for r in rowsb if not r.get("overlap")], "sysA"),
                                          "B": acc_lab([r for r in rowsb if not r.get("overlap")], "sysB")},
            "per_case": rowsb,
        })
    except Exception as e:  # pragma: no cover
        bench_out["error"] = f"benchmark replay skipped: {e}"
    results["benchmark_107_not_independent"] = bench_out

    json.dump(results, open(os.path.join(OUT_DIR, "retraining_results.json"), "w"), indent=1, default=float)
    print("[done] results ->", os.path.join(OUT_DIR, "retraining_results.json"))


if __name__ == "__main__":
    main()
