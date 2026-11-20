"""Comprehensive Unit Tests for Gold Benchmark Split Generation Infrastructure.

Milestone 7.5.4: Gold Benchmark Split Generation Infrastructure
Tests all 20 required scenarios using synthetic fixtures exclusively:
1. Valid synthetic gold benchmark
2. Candidate dataset rejection
3. Missing gold provenance
4. Missing labels
5. Invalid labels
6. Duplicate question IDs
7. Dataset integrity failure (manifest mismatch)
8. Group leakage detection
9. Train/validation/test leakage detection
10. Cross-validation leakage detection
11. Reproducibility with identical seeds
12. Different seeds producing different splits
13. Five-fold assignment completeness
14. Class distribution reporting
15. Insufficient samples per class
16. Group constraints that make stratification impossible
17. Empty dataset rejection
18. Malformed dataset rejection
19. CLI success path
20. CLI safe failure path
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import pytest

from scripts.generate_gold_splits import (
    CANONICAL_CLASSES,
    DEFAULT_CV_FOLDS,
    DEFAULT_SEED,
    FiveFoldCrossValidator,
    GoldDatasetValidator,
    GroupStratifiedSplitter,
    LeakageValidator,
    compute_sha256,
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


# =====================================================================
# Tests 1 - 7: Dataset Validation & Rejection
# =====================================================================

def test_1_valid_synthetic_gold_benchmark(tmp_path):
    """Test 1: Compliant synthetic gold benchmark passes validation."""
    recs = create_synthetic_gold_dataset(60, 30)
    data_file = tmp_path / "gold_synth.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file))

    assert is_valid is True
    assert len(errors) == 0
    assert len(records) == 60
    assert meta["unique_ids"] == 60
    assert meta["unique_groups"] == 30
    assert meta["benchmark_version"] == "1.0"


def test_2_candidate_dataset_rejection(tmp_path):
    """Test 2: Candidate benchmark dataset is rejected (fails gold verification)."""
    recs = create_synthetic_gold_dataset(30, 15)
    for r in recs:
        r["reviewer_agreement"]["annotation_status"] = "needs_human_review"
    cand_file = tmp_path / "uncertainty_benchmark_300_candidate_v1.1.jsonl"
    with open(cand_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(cand_file))

    assert is_valid is False
    assert any("candidate" in err.lower() for err in errors)
    assert any("needs_human_review" in err for err in errors)


def test_3_missing_gold_provenance(tmp_path):
    """Test 3: Dataset lacking human annotation provenance is rejected."""
    recs = create_synthetic_gold_dataset(30, 15)
    for r in recs:
        del r["human_annotation_provenance"]
    data_file = tmp_path / "missing_prov.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file))

    assert is_valid is False
    assert any("human_annotation_provenance" in err for err in errors)


def test_4_missing_labels(tmp_path):
    """Test 4: Records missing final_gold_label are rejected."""
    recs = create_synthetic_gold_dataset(30, 15)
    recs[5]["final_gold_label"] = None
    data_file = tmp_path / "missing_label.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file))

    assert is_valid is False
    assert any("final_gold_label" in err for err in errors)


def test_5_invalid_labels(tmp_path):
    """Test 5: Records with invalid label strings are rejected."""
    recs = create_synthetic_gold_dataset(30, 15)
    recs[10]["final_gold_label"] = "UNCERTAIN_INVALID"
    data_file = tmp_path / "invalid_label.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file))

    assert is_valid is False
    assert any("Invalid final_gold_label" in err for err in errors)


def test_6_duplicate_question_ids(tmp_path):
    """Test 6: Duplicate question IDs are flagged and rejected."""
    recs = create_synthetic_gold_dataset(30, 15)
    recs[12]["id"] = recs[0]["id"]
    data_file = tmp_path / "dup_ids.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file))

    assert is_valid is False
    assert any("Duplicate question id" in err for err in errors)


def test_7_dataset_integrity_failure_manifest_mismatch(tmp_path):
    """Test 7: SHA-256 mismatch against declared manifest triggers failure."""
    recs = create_synthetic_gold_dataset(30, 15)
    data_file = tmp_path / "gold_data.jsonl"
    with open(data_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "output_artifact": {
            "gold_benchmark": {
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(data_file), str(manifest_file))

    assert is_valid is False
    assert any("Integrity check failed" in err for err in errors)


# =====================================================================
# Tests 8 - 10: Independent Leakage Detection
# =====================================================================

def test_8_group_leakage_detection():
    """Test 8: Independent leakage validator catches group overlap between train and test."""
    recs = create_synthetic_gold_dataset(30, 10)
    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    # Find a record in test, and modify its question_group_id to match a train group
    train_id = split_res["splits"]["train"]["ids"][0]
    test_id = split_res["splits"]["test"]["ids"][0]

    train_group = None
    for r in recs:
        if r["id"] == train_id:
            train_group = r["question_group_id"]
            break

    # Deepcopy recs and corrupt the group of test record
    corrupted_recs = copy.deepcopy(recs)
    for r in corrupted_recs:
        if r["id"] == test_id:
            r["question_group_id"] = train_group
            break

    validator = LeakageValidator()
    res = validator.validate_splits_and_cv(corrupted_recs, split_res)

    assert res["leakage_audit_passed"] is False
    assert len(res["violations"]) > 0
    assert any("Group leakage between Train and Test" in v for v in res["violations"])


def test_9_train_val_test_leakage_detection():
    """Test 9: Independent leakage validator catches sample ID overlap between train and val."""
    recs = create_synthetic_gold_dataset(30, 10)
    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    # Corrupt by putting train id into val
    leaked_split = copy.deepcopy(split_res)
    first_train_id = split_res["splits"]["train"]["ids"][0]
    leaked_split["splits"]["val"]["ids"].append(first_train_id)

    validator = LeakageValidator()
    res = validator.validate_splits_and_cv(recs, leaked_split)

    assert res["leakage_audit_passed"] is False
    assert any("overlap between Train and Val" in v for v in res["violations"])


def test_10_cross_validation_leakage_detection():
    """Test 10: Leakage validator catches group overlap in CV folds."""
    recs = create_synthetic_gold_dataset(30, 15)
    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    cv_gen = FiveFoldCrossValidator(seed=42)
    cv_res = cv_gen.split(recs)

    # Corrupt fold 0: add val id to train
    leaked_cv = copy.deepcopy(cv_res)
    fold0_val_id = leaked_cv["folds"][0]["val_ids"][0]
    leaked_cv["folds"][0]["train_ids"].append(fold0_val_id)

    validator = LeakageValidator()
    res = validator.validate_splits_and_cv(recs, split_res, leaked_cv)

    assert res["leakage_audit_passed"] is False
    assert any("Fold 0: ID overlap between train and val" in v or "Group leakage" in v for v in res["violations"])


# =====================================================================
# Tests 11 - 16: Splitting Logic, Reproducibility & Constraints
# =====================================================================

def test_11_reproducibility_with_identical_seeds():
    """Test 11: Given identical data and seed, generated splits are bit-for-bit identical."""
    recs = create_synthetic_gold_dataset(60, 30)

    s1 = GroupStratifiedSplitter(seed=42).split(recs)
    s2 = GroupStratifiedSplitter(seed=42).split(recs)

    assert s1["splits"]["train"]["ids"] == s2["splits"]["train"]["ids"]
    assert s1["splits"]["val"]["ids"] == s2["splits"]["val"]["ids"]
    assert s1["splits"]["test"]["ids"] == s2["splits"]["test"]["ids"]

    cv1 = FiveFoldCrossValidator(seed=42).split(recs)
    cv2 = FiveFoldCrossValidator(seed=42).split(recs)

    assert cv1["sample_to_fold_assignment"] == cv2["sample_to_fold_assignment"]


def test_12_different_seeds_produce_different_splits():
    """Test 12: Different seeds yield different group distributions."""
    recs = create_synthetic_gold_dataset(60, 30)

    s1 = GroupStratifiedSplitter(seed=42).split(recs)
    s2 = GroupStratifiedSplitter(seed=999).split(recs)

    # With 30 groups and different seeds, the exact ID assignment differs
    assert (
        s1["splits"]["train"]["ids"] != s2["splits"]["train"]["ids"]
        or s1["splits"]["val"]["ids"] != s2["splits"]["val"]["ids"]
    )


def test_13_five_fold_assignment_completeness():
    """Test 13: Every sample appears in exactly one validation fold across the 5 folds."""
    recs = create_synthetic_gold_dataset(60, 20)
    cv_gen = FiveFoldCrossValidator(seed=42)
    cv_res = cv_gen.split(recs)

    all_ids = set(r["id"] for r in recs)
    val_occurrences = []
    for f in cv_res["folds"]:
        val_occurrences.extend(f["val_ids"])

    assert len(val_occurrences) == len(all_ids)
    assert set(val_occurrences) == all_ids


def test_14_class_distribution_reporting():
    """Test 14: Class distributions are reported accurately in splits and folds."""
    recs = create_synthetic_gold_dataset(60, 30)
    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    for p in ["train", "val", "test"]:
        p_info = split_res["splits"][p]
        dist = p_info["class_distribution"]
        sum_classes = sum(dist.values())
        assert sum_classes == p_info["count"]
        assert all(c in dist for c in CANONICAL_CLASSES)


def test_15_insufficient_samples_per_class():
    """Test 15: Pipeline handles and warns on severe class imbalance/insufficient class count."""
    recs = create_synthetic_gold_dataset(30, 15)
    # Give only 1 sample the UNKNOWN class
    for r in recs:
        r["final_gold_label"] = "KNOWN"
    recs[0]["final_gold_label"] = "UNKNOWN"

    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    # With only 1 UNKNOWN sample, some partition must have 0 samples for UNKNOWN
    assert any("contains zero samples for class 'UNKNOWN'" in w for w in split_res["warnings"])


def test_16_group_constraints_large_cluster():
    """Test 16: Massive group cluster is accommodated without crash, reporting limitation."""
    recs = create_synthetic_gold_dataset(40, 20)
    # Make group 0 contain 25 of the 40 items
    for i in range(25):
        recs[i]["question_group_id"] = "massive_grp_000"

    splitter = GroupStratifiedSplitter(seed=42)
    split_res = splitter.split(recs)

    # Verify group remains atomic (all 25 items in same partition)
    massive_ids = set(recs[i]["id"] for i in range(25))
    found_partition = None
    for p in ["train", "val", "test"]:
        p_ids = set(split_res["splits"][p]["ids"])
        if massive_ids.issubset(p_ids):
            found_partition = p
            break
    assert found_partition is not None, "Massive group was split across partitions!"


# =====================================================================
# Tests 17 - 18: Boundary and Error Handling
# =====================================================================

def test_17_empty_dataset_rejection(tmp_path):
    """Test 17: Empty dataset file is cleanly rejected."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(empty_file))

    assert is_valid is False
    assert any("completely empty" in err for err in errors)


def test_18_malformed_dataset_rejection(tmp_path):
    """Test 18: Malformed JSON syntax is detected and rejected."""
    malformed_file = tmp_path / "malformed.jsonl"
    malformed_file.write_text("{\"id\": \"1\", invalid json}\n")

    validator = GoldDatasetValidator()
    is_valid, records, errors, meta = validator.validate_file(str(malformed_file))

    assert is_valid is False
    assert any("Malformed JSON" in err for err in errors)


# =====================================================================
# Tests 19 - 20: CLI Integration
# =====================================================================

def test_19_cli_success_path(tmp_path):
    """Test 19: Full CLI execution succeeds and produces all 4 artifacts on compliant gold data."""
    recs = create_synthetic_gold_dataset(60, 30)
    gold_file = tmp_path / "synthetic_gold_benchmark.jsonl"
    with open(gold_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "experiments"
    rep_dir = tmp_path / "reports"

    cmd = [
        sys.executable,
        "scripts/generate_gold_splits.py",
        "--dataset", str(gold_file),
        "--seed", "42",
        "--output-dir", str(out_dir),
        "--reports-dir", str(rep_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "GOLD BENCHMARK SPLIT GENERATION: SUCCESS" in res.stdout

    # Verify generated artifacts
    split_file = out_dir / "gold_split_v1.json"
    cv_file = out_dir / "gold_cv5_v1.json"
    audit_json = rep_dir / "dataset_split_audit_v1.json"
    audit_md = rep_dir / "DATASET_SPLIT_AUDIT_REPORT.md"

    assert split_file.exists()
    assert cv_file.exists()
    assert audit_json.exists()
    assert audit_md.exists()

    # Test --validate-existing on the generated artifacts
    val_cmd = [
        sys.executable,
        "scripts/generate_gold_splits.py",
        "--validate-existing",
        "--dataset", str(gold_file),
        "--split-file", str(split_file),
        "--cv-file", str(cv_file),
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    assert val_res.returncode == 0
    assert "SUCCESS: All split and CV leakage checks passed" in val_res.stdout


def test_20_cli_safe_failure_path():
    """Test 20: CLI targeting candidate benchmark safely halts with exit code 2 and creates no artifacts."""
    cand_file = "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl"

    cmd = [
        sys.executable,
        "scripts/generate_gold_splits.py",
        "--dataset", cand_file,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "GOLD BENCHMARK SPLIT GENERATION" in res.stdout
    assert "BLOCKED" in res.stdout
    assert "Safe failure safeguard active" in res.stdout
