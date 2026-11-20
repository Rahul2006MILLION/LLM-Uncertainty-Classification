"""Comprehensive End-to-End Synthetic Integration Tests for Research Infrastructure.

Milestone 7.5.6: Final Research Infrastructure Integration & Scientific Audit
Validates the entire scientific research pipeline end-to-end on synthetic fixtures:
1. Synthetic Candidate Benchmark
2. Independent Blinded Human Annotation Simulation
3. Annotation Validation (scripts/validate_annotations.py)
4. Inter-Annotator Agreement Analysis (scripts/analyze_annotation_agreement.py)
5. Adjudication Docket Generation & Adjudication Logging
6. Gold Benchmark Assembly & Validation (scripts/finalize_gold_benchmark.py)
7. Group-Stratified Splitting & 5-Fold CV (scripts/generate_gold_splits.py)
8. Independent Leakage Validation (LeakageValidator)
9. Experiment Runner on Baselines & Proposed Model (scripts/run_experiment.py)
10. Standardized Results & Markdown Audit Reports Generation
11. Comprehensive Safe-Failure Gate Verification
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from scripts.analyze_annotation_agreement import (
    align_annotations_via_mapping,
    check_completeness_gate,
    compute_cohens_kappa,
    compute_confusion_matrix,
    compute_krippendorffs_alpha_nominal,
    compute_raw_agreement,
    extract_disagreements_and_docket,
)
from scripts.generate_gold_splits import (
    CANONICAL_CLASSES,
    FiveFoldCrossValidator,
    GoldDatasetValidator,
    GroupStratifiedSplitter,
    LeakageValidator,
)
from src.evaluation.evaluation_pipeline import (
    CANONICAL_16_FEATURES,
    EvaluationDatasetLoader,
    EvaluationMetricsCalculator,
    ExperimentConfig,
    ExperimentRunner,
)


@pytest.fixture
def synthetic_full_pipeline_env(tmp_path):
    """Sets up a complete, temporary synthetic fixture environment for end-to-end testing."""
    env_dir = tmp_path / "pipeline_run"
    env_dir.mkdir()

    # 1. Generate Synthetic Candidate Benchmark (30 questions, 15 groups, 10 per class)
    candidate_records = []
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    q_idx = 1
    for g in range(15):
        grp_id = f"grp_synth_{g:03d}"
        c_label = classes[g % 3]
        for v in range(2):
            candidate_records.append({
                "id": f"synth_cand_{q_idx:03d}",
                "question_group_id": grp_id,
                "question": f"Synthetic question prompt {q_idx} for group {grp_id}?",
                "category": "determinate_factual" if c_label == "KNOWN" else ("polysemic_ambiguity" if c_label == "AMBIGUOUS" else "inaccessible_future"),
                "intended_interpretation": f"Synthetic interpretation {q_idx}",
                "gold_label": c_label,  # candidate label
                "rationale": f"Synthetic rationale {q_idx}",
                "source": "synthetic_pipeline_test",
                "is_adversarial_variant": (v == 1),
                "reviewer_agreement": {
                    "annotation_status": "needs_human_review",
                    "annotator_count": 0,
                    "individual_labels": [],
                    "inter_annotator_agreement": "pending_human_review",
                    "adjudicator_id": None,
                    "adjudication_rationale": None,
                },
            })
            q_idx += 1

    candidate_file = env_dir / "synthetic_candidate_benchmark.jsonl"
    with open(candidate_file, "w", encoding="utf-8") as f:
        for r in candidate_records:
            f.write(json.dumps(r) + "\n")

    # 2. Generate Independent Blinding Keys and Randomized Blinded Files
    rng_a = np.random.default_rng(101)
    rng_b = np.random.default_rng(202)

    shuffled_a = candidate_records.copy()
    rng_a.shuffle(shuffled_a)
    shuffled_b = candidate_records.copy()
    rng_b.shuffle(shuffled_b)

    mapping_a = {}
    records_a = []
    for idx, r in enumerate(shuffled_a, start=1):
        bid = f"ANN_A_{idx:03d}"
        mapping_a[bid] = {"original_id": r["id"], "question": r["question"]}
        # In 28/30 questions, agree with candidate; in 2 questions, disagree to exercise docket
        assigned = r["gold_label"]
        if idx == 5:
            assigned = "AMBIGUOUS" if r["gold_label"] == "KNOWN" else "KNOWN"
        records_a.append({
            "blinded_id": bid,
            "question": r["question"],
            "assigned_label": assigned,
            "confidence": "HIGH" if idx != 7 else "LOW",
            "rationale": f"Annotator A reasoning for {bid}",
            "requires_adjudication": False,
            "notes": "",
        })

    mapping_b = {}
    records_b = []
    for idx, r in enumerate(shuffled_b, start=1):
        bid = f"ANN_B_{idx:03d}"
        mapping_b[bid] = {"original_id": r["id"], "question": r["question"]}
        assigned = r["gold_label"]
        if idx == 12:
            assigned = "UNKNOWN" if r["gold_label"] == "AMBIGUOUS" else "AMBIGUOUS"
        records_b.append({
            "blinded_id": bid,
            "question": r["question"],
            "assigned_label": assigned,
            "confidence": "HIGH" if idx != 7 else "LOW",
            "rationale": f"Annotator B reasoning for {bid}",
            "requires_adjudication": False,
            "notes": "",
        })

    blinding_key = {
        "mapping_annotator_a": mapping_a,
        "mapping_annotator_b": mapping_b,
        "unblinding_annotator_a": {v["original_id"]: k for k, v in mapping_a.items()},
        "unblinding_annotator_b": {v["original_id"]: k for k, v in mapping_b.items()},
    }

    key_file = env_dir / "synthetic_blinding_key.json"
    with open(key_file, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f)

    file_a = env_dir / "blinded_annotation_annotator_a.jsonl"
    with open(file_a, "w", encoding="utf-8") as f:
        for r in records_a:
            f.write(json.dumps(r) + "\n")

    file_b = env_dir / "blinded_annotation_annotator_b.jsonl"
    with open(file_b, "w", encoding="utf-8") as f:
        for r in records_b:
            f.write(json.dumps(r) + "\n")

    # 3. Generate Synthetic 16-Feature Dataset
    features_file = env_dir / "synthetic_features.jsonl"
    with open(features_file, "w", encoding="utf-8") as f:
        for idx, r in enumerate(candidate_records):
            c = r["gold_label"]
            f_vals = {
                "semantic_entropy": 0.1 if c == "KNOWN" else (0.6 if c == "AMBIGUOUS" else 1.2),
                "normalized_entropy": 0.05 if c == "KNOWN" else (0.5 if c == "AMBIGUOUS" else 0.95),
                "dominant_agreement": 0.95 if c == "KNOWN" else (0.5 if c == "AMBIGUOUS" else 0.2),
                "second_agreement": 0.05 if c == "KNOWN" else (0.4 if c == "AMBIGUOUS" else 0.2),
                "agreement_margin": 0.90 if c == "KNOWN" else (0.1 if c == "AMBIGUOUS" else 0.0),
                "cluster_count": 1.0 if c == "KNOWN" else (2.0 if c == "AMBIGUOUS" else 4.0),
                "cluster_ratio": 0.1 if c == "KNOWN" else (0.2 if c == "AMBIGUOUS" else 0.4),
                "mean_pairwise_similarity": 0.95 if c == "KNOWN" else (0.65 if c == "AMBIGUOUS" else 0.35),
                "min_pairwise_similarity": 0.90 if c == "KNOWN" else (0.45 if c == "AMBIGUOUS" else 0.15),
                "response_len_mean": 20.0 + idx,
                "response_len_std": 2.0,
                "distinct_1gram_ratio": 0.4,
                "distinct_2gram_ratio": 0.6,
                "question_overlap_mean": 0.3,
                "hedging_score": 0.0 if c != "UNKNOWN" else 0.7,
                "negation_count": 0.0 if c != "UNKNOWN" else 2.0,
            }
            f.write(json.dumps({"id": r["id"], "features": f_vals}) + "\n")

    return {
        "env_dir": env_dir,
        "candidate_file": str(candidate_file),
        "candidate_records": candidate_records,
        "key_file": str(key_file),
        "blinding_key": blinding_key,
        "file_a": str(file_a),
        "file_b": str(file_b),
        "records_a": records_a,
        "records_b": records_b,
        "features_file": str(features_file),
    }


# =====================================================================
# Full End-to-End Synthetic Pipeline Integration Test
# =====================================================================

def test_full_pipeline_end_to_end_synthetic(synthetic_full_pipeline_env):
    """Executes the complete research pipeline from annotation to model evaluation."""
    env = synthetic_full_pipeline_env
    env_dir = env["env_dir"]

    # -------------------------------------------------------------
    # Step 1: Annotation Validation
    # -------------------------------------------------------------
    from scripts.validate_annotations import AnnotationValidator
    val_a = AnnotationValidator(expected_count=30)
    rep_a = val_a.validate_file(env["file_a"], expected_annotator="A")
    assert rep_a.is_valid is True, f"Annotator A validation failed: {rep_a.errors}"
    assert rep_a.is_complete is True

    val_b = AnnotationValidator(expected_count=30)
    rep_b = val_b.validate_file(env["file_b"], expected_annotator="B")
    assert rep_b.is_valid is True, f"Annotator B validation failed: {rep_b.errors}"
    assert rep_b.is_complete is True

    # -------------------------------------------------------------
    # Step 2: Alignment & Inter-Annotator Agreement Analysis
    # -------------------------------------------------------------
    gate_passed, gate_reasons = check_completeness_gate(env["records_a"], env["records_b"], expected_count=30)
    assert gate_passed is True

    is_aligned, aligned, align_errs = align_annotations_via_mapping(env["records_a"], env["records_b"], env["blinding_key"])
    assert is_aligned is True
    assert len(align_errs) == 0
    assert len(aligned) == 30

    y_a = [item["label_a"] for item in aligned]
    y_b = [item["label_b"] for item in aligned]

    raw_agr = compute_raw_agreement(y_a, y_b)
    assert 0.0 <= raw_agr["raw_agreement_po"] <= 1.0

    kappa = compute_cohens_kappa(y_a, y_b)
    assert -1.0 <= kappa["cohens_kappa"] <= 1.0

    alpha = compute_krippendorffs_alpha_nominal(y_a, y_b)
    assert -1.0 <= alpha <= 1.0

    cm = compute_confusion_matrix(y_a, y_b)
    assert "matrix" in cm and "class_wise_agreement" in cm

    disagreements, docket = extract_disagreements_and_docket(aligned)

    # -------------------------------------------------------------
    # Step 3: Adjudication Logging Simulation
    # -------------------------------------------------------------
    adjudication_file = env_dir / "synthetic_adjudication_log.jsonl"
    adjudication_records = []
    with open(adjudication_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(docket, start=1):
            # Resolve disagreement using tie-breaking protocol (senior consensus)
            res_label = item["label_a"]  # synthetic resolution
            rec = {
                "docket_id": item["docket_id"],
                "underlying_question_id": item["underlying_question_id"],
                "trigger_reason": item["trigger_reasons"][0] if item.get("trigger_reasons") else "disagreement",
                "adjudicated_label": res_label,
                "adjudication_rationale": "Resolved following consensus adjudication rubric v1.1",
                "adjudicator": "senior_adjudicator_consensus",
                "timestamp": "2026-11-20T12:00:00Z",
            }
            f.write(json.dumps(rec) + "\n")
            adjudication_records.append(rec)

    # -------------------------------------------------------------
    # Step 4: Gold Benchmark Assembly & Schema Validation
    # -------------------------------------------------------------
    gold_benchmark_file = env_dir / "synthetic_gold_benchmark.jsonl"
    manifest_file = env_dir / "synthetic_gold_manifest.json"

    cmd_finalize = [
        sys.executable,
        "scripts/finalize_gold_benchmark.py",
        "--candidate", env["candidate_file"],
        "--annotator-a", env["file_a"],
        "--annotator-b", env["file_b"],
        "--blinding-key", env["key_file"],
        "--adjudication-log", str(adjudication_file),
        "--output-gold", str(gold_benchmark_file),
        "--output-manifest", str(manifest_file),
    ]

    res_fin = subprocess.run(cmd_finalize, capture_output=True, text=True)
    assert res_fin.returncode == 0, f"Finalization failed: {res_fin.stderr}\n{res_fin.stdout}"
    assert gold_benchmark_file.exists()
    assert manifest_file.exists()

    # -------------------------------------------------------------
    # Step 5: Group-Stratified Splitting & Cross-Validation
    # -------------------------------------------------------------
    exp_dir = env_dir / "experiments"
    rep_dir = env_dir / "reports"

    cmd_split = [
        sys.executable,
        "scripts/generate_gold_splits.py",
        "--dataset", str(gold_benchmark_file),
        "--manifest", str(manifest_file),
        "--seed", "42",
        "--output-dir", str(exp_dir),
        "--reports-dir", str(rep_dir),
    ]

    res_split = subprocess.run(cmd_split, capture_output=True, text=True)
    assert res_split.returncode == 0, f"Splitting failed: {res_split.stderr}\n{res_split.stdout}"

    split_file = exp_dir / "gold_split_v1.json"
    cv_file = exp_dir / "gold_cv5_v1.json"
    audit_json = rep_dir / "dataset_split_audit_v1.json"
    audit_md = rep_dir / "DATASET_SPLIT_AUDIT_REPORT.md"

    assert split_file.exists()
    assert cv_file.exists()
    assert audit_json.exists()
    assert audit_md.exists()

    # -------------------------------------------------------------
    # Step 6: Model Training & Evaluation (Baselines & XGBoost)
    # -------------------------------------------------------------
    for model_name in ["majority_baseline", "simple_feature_baseline", "logistic_regression", "random_forest", "xgboost"]:
        config_data = {
            "experiment_id": f"full_test_{model_name}",
            "description": f"End-to-end synthetic test for {model_name}",
            "dataset_path": str(gold_benchmark_file),
            "features_path": env["features_file"],
            "split_path": str(split_file),
            "cv_path": str(cv_file),
            "manifest_path": str(manifest_file),
            "output_dir": str(exp_dir / model_name),
            "model_name": model_name,
            "feature_set": "canonical_16",
            "random_seed": 42,
            "cv_folds": 5,
            "dry_run": False,
            "hyperparameters": {"n_estimators": 10, "max_depth": 3} if model_name in {"random_forest", "xgboost"} else {},
        }
        cfg_file = env_dir / f"config_{model_name}.json"
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        cmd_run = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config", str(cfg_file),
        ]
        res_run = subprocess.run(cmd_run, capture_output=True, text=True)
        assert res_run.returncode == 0, f"Model {model_name} failed: {res_run.stderr}\n{res_run.stdout}"

        results_out = exp_dir / model_name / f"full_test_{model_name}_results.json"
        audit_out = exp_dir / model_name / f"full_test_{model_name}_audit.md"

        assert results_out.exists()
        assert audit_out.exists()

        with open(results_out, "r", encoding="utf-8") as f:
            m_res = json.load(f)

        assert m_res["model_name"] == model_name
        assert "test_metrics" in m_res
        assert "accuracy" in m_res["test_metrics"]
        assert "confusion_matrix" in m_res["test_metrics"]
        assert "cross_validation" in m_res


# =====================================================================
# Safe-Failure Gate Verification Tests
# =====================================================================

def test_safe_failure_candidate_benchmark_rejection():
    """Verifies that the evaluation runner strictly rejects candidate benchmarks."""
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config", "configs/experiments/proposed_xgboost.yaml",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "EXPERIMENTAL EVALUATION PIPELINE" in res.stdout
    assert "BLOCKED" in res.stdout


def test_safe_failure_split_generation_on_candidate():
    """Verifies that split generator halts with code 2 on candidate benchmark."""
    cmd = [
        sys.executable,
        "scripts/generate_gold_splits.py",
        "--dataset", "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "GOLD BENCHMARK SPLIT GENERATION" in res.stdout
    assert "BLOCKED" in res.stdout


def test_safe_failure_agreement_analysis_on_incomplete():
    """Verifies that agreement analysis halts with code 2 when human annotations are incomplete."""
    cmd = [
        sys.executable,
        "scripts/analyze_annotation_agreement.py",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "ANNOTATION AGREEMENT ANALYSIS" in res.stdout
    assert "BLOCKED" in res.stdout
    assert "Human annotation is incomplete" in res.stdout


def test_safe_failure_gold_finalization_on_incomplete():
    """Verifies that gold finalization halts with code 2 when human annotations are incomplete."""
    cmd = [
        sys.executable,
        "scripts/finalize_gold_benchmark.py",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "CRITICAL SCIENTIFIC SAFEGUARD: HALTING FINALIZATION" in res.stdout
    assert "Human annotation is NOT completed" in res.stdout
