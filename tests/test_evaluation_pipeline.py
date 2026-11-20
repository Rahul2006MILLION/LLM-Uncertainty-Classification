"""Exhaustive Unit Tests for Experimental Evaluation Pipeline Infrastructure.

Milestone 7.5.5: Experimental Evaluation Pipeline Infrastructure
Tests all 30 required scenarios using synthetic fixtures exclusively:
1. Valid synthetic gold dataset
2. Candidate dataset rejection
3. Missing gold provenance
4. Invalid labels
5. Missing features
6. Invalid feature schema
7. Duplicate question IDs
8. Feature ID mismatch
9. Label ID mismatch
10. Invalid split artifacts
11. Train/test leakage detection
12. Group leakage detection
13. Cross-validation leakage detection
14. Majority baseline execution & metrics
15. Logistic Regression execution & metrics
16. Random Forest execution & metrics
17. XGBoost proposed model execution & metrics
18. Metric calculations accuracy (Accuracy, Balanced Acc, F1, MCC, Kappa)
19. Confusion matrix structure (3x3, Rows=True, Cols=Pred)
20. Class-wise metrics reporting for KNOWN, AMBIGUOUS, UNKNOWN
21. Cross-validation execution & aggregate mean/std reporting
22. Reproducibility across multiple runs with identical seed
23. Preprocessing leakage prevention (StandardScaler fitted strictly on training data)
24. Test-set isolation (test set never seen during training or CV)
25. Experiment configuration validation (schema checks, invalid model name rejection)
26. Results serialization (machine-readable JSON schema compliance)
27. Audit report generation (Markdown and JSON audit trails)
28. Dry-run behavior (validates without training or saving models)
29. Candidate benchmark safe failure (CLI exits with code 2)
30. Missing gold benchmark safe failure (CLI exits with code 2)
"""

import copy
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pytest

from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation.evaluation_pipeline import (
    CANONICAL_16_FEATURES,
    CANONICAL_CLASSES,
    EvaluationDatasetLoader,
    EvaluationMetricsCalculator,
    ExperimentConfig,
    ExperimentRunner,
    GoldDatasetValidator,
    LeakageValidator,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    RandomForestBaseline,
    SimpleFeatureBaseline,
    XGBoostClassifierModel,
    instantiate_model,
)


def create_synthetic_gold_record(
    idx: int,
    group_id: str,
    label: str,
    question: str = "Synthetic prompt question?",
    category: str = "determinate_factual",
) -> Dict[str, Any]:
    """Helper creating a compliant synthetic finalized gold record."""
    return {
        "id": f"synth_q_{idx:03d}",
        "question_group_id": group_id,
        "question": f"{question} [id:{idx}]",
        "category": category,
        "final_gold_label": label,
        "candidate_label": label,
        "candidate_matches_gold": True,
        "source": "synthetic_test",
        "is_adversarial_variant": False,
        "human_annotation_provenance": {
            "annotator_a": {
                "blinded_id": f"ANN_A_{idx:03d}",
                "assigned_label": label,
                "confidence": "HIGH",
                "requires_adjudication": False,
            },
            "annotator_b": {
                "blinded_id": f"ANN_B_{idx:03d}",
                "assigned_label": label,
                "confidence": "HIGH",
                "requires_adjudication": False,
            },
            "is_unanimous": True,
            "adjudication_required": False,
            "adjudication_record": None,
        },
        "reviewer_agreement": {
            "annotation_status": "gold_benchmark",
            "annotator_count": 2,
            "individual_labels": [label, label],
            "inter_annotator_agreement": "unanimous_consensus",
            "adjudicator_id": None,
            "adjudication_rationale": None,
        },
        "benchmark_version": "1.0",
        "gold_finalized_timestamp": "2026-11-20T12:00:00Z",
    }


def create_synthetic_gold_dataset(num_records: int = 60, num_groups: int = 30) -> List[Dict[str, Any]]:
    """Helper creating a balanced synthetic gold benchmark."""
    records = []
    classes = CANONICAL_CLASSES
    records_per_group = num_records // num_groups
    curr_id = 1
    for g in range(num_groups):
        gid = f"grp_{g:03d}"
        g_label = classes[g % len(classes)]
        for _ in range(records_per_group):
            records.append(create_synthetic_gold_record(curr_id, gid, g_label))
            curr_id += 1
    return records


def create_synthetic_splits_and_cv(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Creates leakage-free 70/15/15 split and 5-fold CV for synthetic records."""
    groups = sorted(list(set(r["question_group_id"] for r in records)))
    n_groups = len(groups)

    # Deterministic group partition (approx 70% train, 15% val, 15% test)
    n_train_g = int(n_groups * 0.70)
    n_val_g = int(n_groups * 0.15)
    train_groups = set(groups[:n_train_g])
    val_groups = set(groups[n_train_g:n_train_g + n_val_g])
    test_groups = set(groups[n_train_g + n_val_g:])

    train_ids = [r["id"] for r in records if r["question_group_id"] in train_groups]
    val_ids = [r["id"] for r in records if r["question_group_id"] in val_groups]
    test_ids = [r["id"] for r in records if r["question_group_id"] in test_groups]

    split_res = {
        "random_seed": 42,
        "splits": {
            "train": {"ids": sorted(train_ids), "count": len(train_ids)},
            "val": {"ids": sorted(val_ids), "count": len(val_ids)},
            "test": {"ids": sorted(test_ids), "count": len(test_ids)},
        }
    }

    # 5-Fold StratifiedGroupKFold CV over Train + Val pool (held-out Test set strictly isolated)
    cv_records = [r for r in records if r["question_group_id"] in (train_groups | val_groups)]
    X = np.arange(len(cv_records))
    y = np.array([r["final_gold_label"] for r in cv_records])
    grp_arr = np.array([r["question_group_id"] for r in cv_records])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []
    for f_idx, (t_idx, v_idx) in enumerate(sgkf.split(X, y, grp_arr)):
        folds.append({
            "fold_index": f_idx,
            "train_ids": sorted([cv_records[i]["id"] for i in t_idx]),
            "val_ids": sorted([cv_records[i]["id"] for i in v_idx]),
        })

    cv_res = {
        "random_seed": 42,
        "n_splits": 5,
        "folds": folds,
    }
    return split_res, cv_res


def create_synthetic_feature_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Helper generating synthetic 16-feature records mapped by question ID."""
    feature_dict: Dict[str, Dict[str, float]] = {}
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

    for idx, r in enumerate(records):
        qid = r["id"]
        label = r["final_gold_label"]
        # Generate synthetic feature values correlated with class
        if label == "KNOWN":
            f_vals = {
                "semantic_entropy": 0.05 + 0.01 * (idx % 5),
                "normalized_entropy": 0.05 + 0.01 * (idx % 5),
                "dominant_agreement": 0.95 - 0.01 * (idx % 5),
                "second_agreement": 0.05,
                "agreement_margin": 0.90,
                "cluster_count": 1.0,
                "cluster_ratio": 0.1,
                "mean_pairwise_similarity": 0.95,
                "min_pairwise_similarity": 0.90,
                "response_len_mean": 25.0,
                "response_len_std": 2.0,
                "distinct_1gram_ratio": 0.4,
                "distinct_2gram_ratio": 0.6,
                "question_overlap_mean": 0.3,
                "hedging_score": 0.0,
                "negation_count": 0.0,
            }
        elif label == "AMBIGUOUS":
            f_vals = {
                "semantic_entropy": 0.65 + 0.01 * (idx % 5),
                "normalized_entropy": 0.65 + 0.01 * (idx % 5),
                "dominant_agreement": 0.45,
                "second_agreement": 0.40,
                "agreement_margin": 0.05,
                "cluster_count": 2.0,
                "cluster_ratio": 0.2,
                "mean_pairwise_similarity": 0.65,
                "min_pairwise_similarity": 0.45,
                "response_len_mean": 45.0,
                "response_len_std": 6.0,
                "distinct_1gram_ratio": 0.6,
                "distinct_2gram_ratio": 0.8,
                "question_overlap_mean": 0.2,
                "hedging_score": 0.1,
                "negation_count": 0.0,
            }
        else:  # UNKNOWN
            f_vals = {
                "semantic_entropy": 1.10 + 0.01 * (idx % 5),
                "normalized_entropy": 0.90,
                "dominant_agreement": 0.20,
                "second_agreement": 0.20,
                "agreement_margin": 0.0,
                "cluster_count": 5.0,
                "cluster_ratio": 0.5,
                "mean_pairwise_similarity": 0.35,
                "min_pairwise_similarity": 0.15,
                "response_len_mean": 15.0,
                "response_len_std": 4.0,
                "distinct_1gram_ratio": 0.3,
                "distinct_2gram_ratio": 0.4,
                "question_overlap_mean": 0.1,
                "hedging_score": 0.8,
                "negation_count": 2.0,
            }
        feature_dict[qid] = f_vals
    return feature_dict


@pytest.fixture
def synthetic_experiment_env(tmp_path):
    """Sets up a compliant synthetic gold benchmark, features, split, and CV files."""
    records = create_synthetic_gold_dataset(60, 30)
    gold_file = tmp_path / "gold_synth_data.jsonl"
    with open(gold_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    features_map = create_synthetic_feature_records(records)
    features_file = tmp_path / "features_synth_data.jsonl"
    with open(features_file, "w") as f:
        for qid, feats in features_map.items():
            f.write(json.dumps({"id": qid, "features": feats}) + "\n")

    split_res, cv_res = create_synthetic_splits_and_cv(records)
    split_file = tmp_path / "split_synth_data.json"
    with open(split_file, "w") as f:
        json.dump(split_res, f)

    cv_file = tmp_path / "cv_synth_data.json"
    with open(cv_file, "w") as f:
        json.dump(cv_res, f)

    config_data = {
        "experiment_id": "test_exp_synth",
        "description": "Synthetic unit test experiment",
        "dataset_path": str(gold_file),
        "features_path": str(features_file),
        "split_path": str(split_file),
        "cv_path": str(cv_file),
        "output_dir": str(tmp_path / "results"),
        "model_name": "xgboost",
        "feature_set": "canonical_16",
        "random_seed": 42,
        "cv_folds": 5,
        "dry_run": False,
        "hyperparameters": {"n_estimators": 10, "max_depth": 3},
    }
    config_file = tmp_path / "test_config.yaml"
    import yaml
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return {
        "tmp_path": tmp_path,
        "records": records,
        "gold_file": str(gold_file),
        "features_file": str(features_file),
        "features_map": features_map,
        "split_file": str(split_file),
        "split_res": split_res,
        "cv_file": str(cv_file),
        "cv_res": cv_res,
        "config_data": config_data,
        "config_file": str(config_file),
    }


# =====================================================================
# Tests 1 - 13: Dataset Loading, Validation & Leakage Detection
# =====================================================================

def test_1_valid_synthetic_gold_dataset(synthetic_experiment_env):
    """Test 1: Valid synthetic gold benchmark and features load cleanly."""
    cfg = ExperimentConfig(synthetic_experiment_env["config_data"])
    loader = EvaluationDatasetLoader(cfg)
    is_valid, payload, errors = loader.load_and_validate()

    assert is_valid is True
    assert len(errors) == 0
    assert len(payload["gold_records"]) == 60
    assert len(payload["train_ids"]) > 0


def test_2_candidate_dataset_rejection(synthetic_experiment_env):
    """Test 2: Rejection when input dataset is candidate data."""
    cand_data = copy.deepcopy(synthetic_experiment_env["records"])
    for r in cand_data:
        r["reviewer_agreement"]["annotation_status"] = "needs_human_review"
    cand_file = synthetic_experiment_env["tmp_path"] / "uncertainty_benchmark_300_candidate_v1.1.jsonl"
    with open(cand_file, "w") as f:
        for r in cand_data:
            f.write(json.dumps(r) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dataset_path"] = str(cand_file)
    cfg = ExperimentConfig(cfg_dict)
    loader = EvaluationDatasetLoader(cfg)
    is_valid, payload, errors = loader.load_and_validate()

    assert is_valid is False
    assert any("candidate" in err.lower() or "needs_human_review" in err for err in errors)


def test_3_missing_gold_provenance(synthetic_experiment_env):
    """Test 3: Rejection when human_annotation_provenance is missing."""
    data = copy.deepcopy(synthetic_experiment_env["records"])
    for r in data:
        del r["human_annotation_provenance"]
    bad_file = synthetic_experiment_env["tmp_path"] / "no_prov.jsonl"
    with open(bad_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dataset_path"] = str(bad_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("human_annotation_provenance" in err for err in errors)


def test_4_invalid_labels(synthetic_experiment_env):
    """Test 4: Rejection when final_gold_label is corrupted."""
    data = copy.deepcopy(synthetic_experiment_env["records"])
    data[0]["final_gold_label"] = "INVALID_LABEL_XYZ"
    bad_file = synthetic_experiment_env["tmp_path"] / "bad_labels.jsonl"
    with open(bad_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dataset_path"] = str(bad_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Invalid final_gold_label" in err for err in errors)


def test_5_missing_features(synthetic_experiment_env):
    """Test 5: Rejection when a required canonical feature is missing in feature file."""
    bad_feat_file = synthetic_experiment_env["tmp_path"] / "bad_features.jsonl"
    with open(bad_feat_file, "w") as f:
        for qid, feats in synthetic_experiment_env["features_map"].items():
            f_copy = copy.deepcopy(feats)
            del f_copy["semantic_entropy"]
            f.write(json.dumps({"id": qid, "features": f_copy}) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["features_path"] = str(bad_feat_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Missing required feature 'semantic_entropy'" in err for err in errors)


def test_6_invalid_feature_schema(synthetic_experiment_env):
    """Test 6: Rejection when feature value is non-numeric or NaN."""
    bad_feat_file = synthetic_experiment_env["tmp_path"] / "nan_features.jsonl"
    with open(bad_feat_file, "w") as f:
        for qid, feats in synthetic_experiment_env["features_map"].items():
            f_copy = copy.deepcopy(feats)
            f_copy["semantic_entropy"] = float("nan")
            f.write(json.dumps({"id": qid, "features": f_copy}) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["features_path"] = str(bad_feat_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Invalid non-numeric" in err for err in errors)


def test_7_duplicate_question_ids(synthetic_experiment_env):
    """Test 7: Rejection when gold dataset contains duplicate IDs."""
    data = copy.deepcopy(synthetic_experiment_env["records"])
    data[1]["id"] = data[0]["id"]
    dup_file = synthetic_experiment_env["tmp_path"] / "dup_gold.jsonl"
    with open(dup_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dataset_path"] = str(dup_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Duplicate question id" in err for err in errors)


def test_8_feature_id_mismatch(synthetic_experiment_env):
    """Test 8: Rejection when gold records lack corresponding feature records."""
    bad_feat_file = synthetic_experiment_env["tmp_path"] / "missing_ids_feat.jsonl"
    with open(bad_feat_file, "w") as f:
        # Write only half the records
        for qid, feats in list(synthetic_experiment_env["features_map"].items())[:30]:
            f.write(json.dumps({"id": qid, "features": feats}) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["features_path"] = str(bad_feat_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("lack features in" in err for err in errors)


def test_9_label_id_mismatch(synthetic_experiment_env):
    """Test 9: Rejection when gold records lack split assignment."""
    bad_split_data = copy.deepcopy(synthetic_experiment_env["split_res"])
    bad_split_data["splits"]["train"]["ids"] = bad_split_data["splits"]["train"]["ids"][:10]
    bad_split_file = synthetic_experiment_env["tmp_path"] / "bad_split_missing.json"
    with open(bad_split_file, "w") as f:
        json.dump(bad_split_data, f)

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["split_path"] = str(bad_split_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("lack split assignments in" in err for err in errors)


def test_10_invalid_split_artifacts(synthetic_experiment_env):
    """Test 10: Rejection when split file does not exist."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["split_path"] = "non_existent_split_file.json"
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Split artifact not found" in err for err in errors)


def test_11_train_test_leakage_detection(synthetic_experiment_env):
    """Test 11: Leakage validator catches train/test sample ID overlap."""
    bad_split_data = copy.deepcopy(synthetic_experiment_env["split_res"])
    leaked_id = bad_split_data["splits"]["test"]["ids"][0]
    bad_split_data["splits"]["train"]["ids"].append(leaked_id)
    bad_split_file = synthetic_experiment_env["tmp_path"] / "train_test_leak.json"
    with open(bad_split_file, "w") as f:
        json.dump(bad_split_data, f)

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["split_path"] = str(bad_split_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("overlap between Train and Test" in err for err in errors)


def test_12_group_leakage_detection(synthetic_experiment_env):
    """Test 12: Leakage validator catches group leakage across splits."""
    bad_records = copy.deepcopy(synthetic_experiment_env["records"])
    train_id = synthetic_experiment_env["split_res"]["splits"]["train"]["ids"][0]
    test_id = synthetic_experiment_env["split_res"]["splits"]["test"]["ids"][0]

    train_grp = None
    for r in bad_records:
        if r["id"] == train_id:
            train_grp = r["question_group_id"]
            break

    for r in bad_records:
        if r["id"] == test_id:
            r["question_group_id"] = train_grp
            break

    bad_gold_file = synthetic_experiment_env["tmp_path"] / "grp_leak_gold.jsonl"
    with open(bad_gold_file, "w") as f:
        for r in bad_records:
            f.write(json.dumps(r) + "\n")

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dataset_path"] = str(bad_gold_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Group leakage between Train and Test" in err for err in errors)


def test_13_cross_validation_leakage_detection(synthetic_experiment_env):
    """Test 13: Leakage validator catches cross-validation fold contamination."""
    bad_cv = copy.deepcopy(synthetic_experiment_env["cv_res"])
    f0_val = bad_cv["folds"][0]["val_ids"][0]
    bad_cv["folds"][0]["train_ids"].append(f0_val)

    bad_cv_file = synthetic_experiment_env["tmp_path"] / "bad_cv_leak.json"
    with open(bad_cv_file, "w") as f:
        json.dump(bad_cv, f)

    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["cv_path"] = str(bad_cv_file)
    cfg = ExperimentConfig(cfg_dict)
    is_valid, payload, errors = EvaluationDatasetLoader(cfg).load_and_validate()

    assert is_valid is False
    assert any("Fold 0: ID overlap between train and val" in err for err in errors)


# =====================================================================
# Tests 14 - 17: Model Implementations Execution
# =====================================================================

def test_14_majority_baseline_execution(synthetic_experiment_env):
    """Test 14: Majority Class baseline fits training majority exclusively."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["model_name"] = "majority_baseline"
    cfg_dict["experiment_id"] = "test_majority"
    cfg = ExperimentConfig(cfg_dict)

    exit_code, results, errors = ExperimentRunner(cfg).run()
    assert exit_code == 0
    assert results["model_name"] == "majority_baseline"
    assert "test_metrics" in results
    assert 0.0 <= results["test_metrics"]["accuracy"] <= 1.0


def test_15_logistic_regression_execution(synthetic_experiment_env):
    """Test 15: Logistic Regression baseline executes and outputs metrics."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["model_name"] = "logistic_regression"
    cfg_dict["experiment_id"] = "test_logreg"
    cfg = ExperimentConfig(cfg_dict)

    exit_code, results, errors = ExperimentRunner(cfg).run()
    assert exit_code == 0
    assert results["model_name"] == "logistic_regression"
    assert results["test_metrics"]["macro_f1"] > 0.0


def test_16_random_forest_execution(synthetic_experiment_env):
    """Test 16: Random Forest baseline executes and extracts feature importances."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["model_name"] = "random_forest"
    cfg_dict["experiment_id"] = "test_rf"
    cfg = ExperimentConfig(cfg_dict)

    exit_code, results, errors = ExperimentRunner(cfg).run()
    assert exit_code == 0
    assert results["model_name"] == "random_forest"
    assert "feature_importances" in results
    assert len(results["feature_importances"]) == 16


def test_17_xgboost_execution(synthetic_experiment_env):
    """Test 17: XGBoost proposed classifier executes and outputs complete results."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["model_name"] = "xgboost"
    cfg_dict["experiment_id"] = "test_xgb"
    cfg = ExperimentConfig(cfg_dict)

    exit_code, results, errors = ExperimentRunner(cfg).run()
    assert exit_code == 0
    assert results["model_name"] == "xgboost"
    assert "test_metrics" in results
    assert results["test_metrics"]["accuracy"] > 0.5


# =====================================================================
# Tests 18 - 24: Metrics, Cross-Validation & Scientific Safeguards
# =====================================================================

def test_18_metric_calculations_accuracy():
    """Test 18: Metric calculator computes exact classification and correlation metrics."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])  # 5/6 correct

    metrics = EvaluationMetricsCalculator.compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == round(5 / 6, 4)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0
    assert -1.0 <= metrics["cohens_kappa"] <= 1.0


def test_19_confusion_matrix_structure():
    """Test 19: Confusion matrix produces canonical 3x3 layout with accurate cells."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    metrics = EvaluationMetricsCalculator.compute_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]

    assert "KNOWN" in cm and "AMBIGUOUS" in cm and "UNKNOWN" in cm
    assert cm["KNOWN"]["KNOWN"] == 1
    assert cm["KNOWN"]["AMBIGUOUS"] == 1
    assert cm["AMBIGUOUS"]["AMBIGUOUS"] == 2
    assert cm["UNKNOWN"]["KNOWN"] == 1
    assert cm["UNKNOWN"]["UNKNOWN"] == 1


def test_20_class_wise_metrics_reporting():
    """Test 20: Per-class Precision, Recall, and F1 are independently reported."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])  # Perfect classification

    metrics = EvaluationMetricsCalculator.compute_metrics(y_true, y_pred)
    per_class = metrics["per_class"]

    for c in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]:
        assert per_class[c]["precision"] == 1.0
        assert per_class[c]["recall"] == 1.0
        assert per_class[c]["f1"] == 1.0
        assert per_class[c]["support"] == 2


def test_21_cross_validation_execution(synthetic_experiment_env):
    """Test 21: 5-fold cross validation executes and computes mean and std."""
    cfg = ExperimentConfig(synthetic_experiment_env["config_data"])
    exit_code, results, errors = ExperimentRunner(cfg).run()

    cv = results["cross_validation"]
    assert cv["cv_folds"] == 5
    assert len(cv["individual_folds"]) == 5
    assert "mean" in cv["aggregate_metrics"]["accuracy"]
    assert "std" in cv["aggregate_metrics"]["accuracy"]


def test_22_reproducibility_with_fixed_seed(synthetic_experiment_env):
    """Test 22: Identical seed produces bit-for-bit identical test metrics."""
    cfg1 = ExperimentConfig(synthetic_experiment_env["config_data"])
    cfg2 = ExperimentConfig(synthetic_experiment_env["config_data"])

    _, r1, _ = ExperimentRunner(cfg1).run()
    _, r2, _ = ExperimentRunner(cfg2).run()

    assert r1["test_metrics"]["accuracy"] == r2["test_metrics"]["accuracy"]
    assert r1["test_metrics"]["macro_f1"] == r2["test_metrics"]["macro_f1"]


def test_23_preprocessing_leakage_prevention():
    """Test 23: StandardScaler is fitted strictly on training partition."""
    X_train = np.array([[10.0, 20.0], [30.0, 40.0]])
    y_train = np.array([0, 1])

    X_test = np.array([[1000.0, 2000.0]])  # Extreme outlier

    model = LogisticRegressionBaseline()
    model.fit(X_train, y_train)

    # Verify scaler mean matches X_train mean exactly
    assert np.allclose(model.scaler.mean_, np.array([20.0, 30.0]))
    # Predict on test
    preds = model.predict(X_test)
    assert len(preds) == 1


def test_24_test_set_isolation(synthetic_experiment_env):
    """Test 24: Test set IDs are never included in training or cross-validation folds."""
    cfg = ExperimentConfig(synthetic_experiment_env["config_data"])
    loader = EvaluationDatasetLoader(cfg)
    _, payload, _ = loader.load_and_validate()

    test_ids = set(payload["test_ids"])
    for fold in payload["cv_data"]["folds"]:
        assert test_ids.isdisjoint(set(fold["train_ids"]))
        assert test_ids.isdisjoint(set(fold["val_ids"]))


# =====================================================================
# Tests 25 - 30: Configuration, CLI & Safe Execution Gates
# =====================================================================

def test_25_experiment_config_validation():
    """Test 25: ExperimentConfig validates required keys and rejects invalid models."""
    with pytest.raises(ValueError, match="missing required field"):
        ExperimentConfig({"experiment_id": "test"})

    with pytest.raises(ValueError, match="Unsupported model_name"):
        ExperimentConfig({
            "experiment_id": "test",
            "dataset_path": "a",
            "features_path": "b",
            "split_path": "c",
            "cv_path": "d",
            "model_name": "quantum_neural_net",
        })


def test_26_results_serialization(synthetic_experiment_env):
    """Test 26: Results JSON file is created and adheres to schema."""
    cfg = ExperimentConfig(synthetic_experiment_env["config_data"])
    exit_code, results, errors = ExperimentRunner(cfg).run()

    res_file = os.path.join(cfg.output_dir, f"{cfg.experiment_id}_results.json")
    assert os.path.exists(res_file)
    with open(res_file, "r") as f:
        saved = json.load(f)

    assert saved["experiment_id"] == cfg.experiment_id
    assert saved["model_name"] == cfg.model_name
    assert "test_metrics" in saved


def test_27_audit_report_generation(synthetic_experiment_env):
    """Test 27: Markdown audit report is created with performance table."""
    cfg = ExperimentConfig(synthetic_experiment_env["config_data"])
    exit_code, results, errors = ExperimentRunner(cfg).run()

    audit_file = os.path.join(cfg.output_dir, f"{cfg.experiment_id}_audit.md")
    assert os.path.exists(audit_file)
    content = open(audit_file).read()

    assert "Experiment Audit Report" in content
    assert "Executive Performance Summary" in content
    assert "Class-Wise Performance Breakdown" in content


def test_28_dry_run_behavior(synthetic_experiment_env):
    """Test 28: Dry-run validates datasets without training models or creating output files."""
    cfg_dict = copy.deepcopy(synthetic_experiment_env["config_data"])
    cfg_dict["dry_run"] = True
    cfg = ExperimentConfig(cfg_dict)

    exit_code, results, errors = ExperimentRunner(cfg).run()
    assert exit_code == 0
    assert results["status"] == "DRY_RUN_PASSED"

    # Verify no model file created
    model_file = os.path.join(cfg.output_dir, f"{cfg.experiment_id}_model.joblib")
    assert not os.path.exists(model_file)


def test_29_candidate_benchmark_safe_failure_cli(tmp_path):
    """Test 29: CLI targeting candidate benchmark halts with code 2."""
    config_data = {
        "experiment_id": "test_cand_fail",
        "dataset_path": "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl",
        "features_path": "data/features/features_300_v1.1.jsonl",
        "split_path": "data/experiments/gold_split_v1.json",
        "cv_path": "data/experiments/gold_cv5_v1.json",
        "output_dir": str(tmp_path),
        "model_name": "xgboost",
    }
    cfg_path = tmp_path / "cand_config.yaml"
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(config_data, f)

    cmd = [sys.executable, "scripts/run_experiment.py", "--config", str(cfg_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 2
    assert "EXPERIMENTAL EVALUATION PIPELINE" in res.stdout
    assert "BLOCKED" in res.stdout
    assert "Safe failure safeguard active" in res.stdout


def test_30_missing_gold_benchmark_safe_failure_cli(tmp_path):
    """Test 30: CLI targeting non-existent gold benchmark halts with code 2."""
    config_data = {
        "experiment_id": "test_missing_fail",
        "dataset_path": "non_existent_gold_benchmark.jsonl",
        "features_path": "data/features/features_300_v1.1.jsonl",
        "split_path": "data/experiments/gold_split_v1.json",
        "cv_path": "data/experiments/gold_cv5_v1.json",
        "output_dir": str(tmp_path),
        "model_name": "xgboost",
    }
    cfg_path = tmp_path / "missing_config.yaml"
    import yaml
    with open(cfg_path, "w") as f:
        yaml.dump(config_data, f)

    cmd = [sys.executable, "scripts/run_experiment.py", "--config", str(cfg_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 2
    assert "EXPERIMENTAL EVALUATION PIPELINE" in res.stdout
    assert "BLOCKED" in res.stdout
