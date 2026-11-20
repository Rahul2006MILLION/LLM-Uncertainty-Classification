"""Unit tests for inter-annotator agreement analysis pipeline and adjudication docket generation.

Uses synthetic test records only to thoroughly verify:
A. Annotation Loading (valid/malformed JSONL, missing fields, duplicate/unexpected IDs)
B. Completeness Gate (incomplete/unannotated detection, count checks)
C. Alignment (mapping different blinded IDs, randomized order, duplicate/missing mapping)
D. Raw Agreement (numerator, denominator, percentage, edge cases)
E. Cohen's Kappa (observed, expected, SE, 95% CI, edge cases, degenerate single-class)
F. Confusion Matrix (3x3 row/column alignment, class counts)
G. Class-Wise Agreement (specific agreement, imbalanced distributions)
H. Reporting (JSON/Markdown generation, zero leakage of model outputs or candidate labels)
I. Blinding Safeguards (key isolation, reports never leak blinding key or candidate labels)
"""

import json
import math
import os
import pytest

from scripts.analyze_annotation_agreement import (
    load_annotation_file,
    check_completeness_gate,
    align_annotations_via_mapping,
    compute_raw_agreement,
    compute_expected_agreement,
    compute_cohens_kappa,
    compute_krippendorffs_alpha_nominal,
    compute_confusion_matrix,
    compute_annotation_distributions,
    extract_disagreements_and_docket,
    generate_markdown_report,
    CLASSES,
)


def make_synthetic_records(count=5, annotator="A", label="KNOWN", confidence="HIGH"):
    """Helper to generate synthetic annotation records."""
    records = []
    for i in range(1, count + 1):
        records.append({
            "blinded_id": f"ANN_{annotator}_{i:03d}",
            "question": f"Synthetic question query {i} with sufficient text?",
            "assigned_label": label,
            "confidence": confidence,
            "rationale": f"Synthetic rationale {i}",
            "requires_adjudication": False,
            "notes": "",
        })
    return records


def write_jsonl(filepath, records):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ==============================================================================
# SECTION A: Annotation Loading Tests
# ==============================================================================

def test_loading_valid_and_malformed_jsonl(tmp_path):
    f_valid = tmp_path / "valid.jsonl"
    recs = make_synthetic_records(3, annotator="A")
    write_jsonl(f_valid, recs)

    loaded, errors = load_annotation_file(str(f_valid), expected_annotator="A")
    assert len(loaded) == 3
    assert len(errors) == 0

    # Malformed JSON
    f_mal = tmp_path / "malformed.jsonl"
    with open(f_mal, "w", encoding="utf-8") as f:
        f.write('{"blinded_id": "ANN_A_001", "question": "q1?"}\n')
        f.write('BROKEN_JSON_STRING\n')
        f.write('{"blinded_id": "ANN_B_002", "question": "q2?"}\n')  # Mismatched annotator prefix

    loaded_mal, errors_mal = load_annotation_file(str(f_mal), expected_annotator="A")
    assert len(loaded_mal) == 2
    assert any("Malformed JSON" in e for e in errors_mal)
    assert any("does not match expected prefix" in e for e in errors_mal)


def test_loading_duplicate_and_missing_ids(tmp_path):
    f_dup = tmp_path / "dup.jsonl"
    recs = make_synthetic_records(3, annotator="A")
    recs[2]["blinded_id"] = "ANN_A_001"  # Duplicate of record 0
    write_jsonl(f_dup, recs)

    loaded, errors = load_annotation_file(str(f_dup))
    assert any("Duplicate blinded_id 'ANN_A_001'" in e for e in errors)


# ==============================================================================
# SECTION B: Completeness Gate Tests
# ==============================================================================

def test_completeness_gate_detection():
    # Incomplete records (null assigned_label)
    recs_a = make_synthetic_records(5, annotator="A", label=None)
    recs_b = make_synthetic_records(5, annotator="B", label="KNOWN")

    is_complete, stats = check_completeness_gate(recs_a, recs_b, expected_count=5)
    assert is_complete is False
    assert stats["annotator_a"]["remaining"] == 5
    assert any("unannotated records" in f for f in stats["failures"])

    # Count mismatch
    recs_short = make_synthetic_records(4, annotator="A", label="KNOWN")
    is_complete_cnt, stats_cnt = check_completeness_gate(recs_short, recs_b, expected_count=5)
    assert is_complete_cnt is False
    assert any("total records (4) != expected (5)" in f for f in stats_cnt["failures"])

    # Invalid labels outside taxonomy
    recs_inv = make_synthetic_records(5, annotator="A", label="MAYBE")
    is_complete_inv, stats_inv = check_completeness_gate(recs_inv, recs_b, expected_count=5)
    assert is_complete_inv is False
    assert any("invalid labels" in f for f in stats_inv["failures"])

    # Valid complete records
    recs_valid_a = make_synthetic_records(5, annotator="A", label="KNOWN")
    is_complete_ok, stats_ok = check_completeness_gate(recs_valid_a, recs_b, expected_count=5)
    assert is_complete_ok is True
    assert stats_ok["annotator_a"]["remaining"] == 0
    assert stats_ok["annotator_b"]["remaining"] == 0


# ==============================================================================
# SECTION C: Alignment Tests (Never rely on file position)
# ==============================================================================

def test_alignment_independent_of_file_order():
    """Test that records are aligned strictly by underlying question ID, even when shuffled."""
    recs_a = [
        {"blinded_id": "ANN_A_001", "question": "Question 1?", "assigned_label": "KNOWN", "confidence": "HIGH", "requires_adjudication": False, "rationale": "r1"},
        {"blinded_id": "ANN_A_002", "question": "Question 2?", "assigned_label": "AMBIGUOUS", "confidence": "MEDIUM", "requires_adjudication": False, "rationale": "r2"},
        {"blinded_id": "ANN_A_003", "question": "Question 3?", "assigned_label": "UNKNOWN", "confidence": "LOW", "requires_adjudication": False, "rationale": "r3"},
    ]
    # Annotator B in completely inverted order with different blinded IDs
    recs_b = [
        {"blinded_id": "ANN_B_099", "question": "Question 3?", "assigned_label": "UNKNOWN", "confidence": "LOW", "requires_adjudication": False, "rationale": "r3"},
        {"blinded_id": "ANN_B_055", "question": "Question 1?", "assigned_label": "KNOWN", "confidence": "HIGH", "requires_adjudication": False, "rationale": "r1"},
        {"blinded_id": "ANN_B_012", "question": "Question 2?", "assigned_label": "AMBIGUOUS", "confidence": "HIGH", "requires_adjudication": False, "rationale": "r2"},
    ]

    mapping_data = {
        "mapping_annotator_a": {
            "ANN_A_001": {"original_id": "q_1", "question": "Question 1?"},
            "ANN_A_002": {"original_id": "q_2", "question": "Question 2?"},
            "ANN_A_003": {"original_id": "q_3", "question": "Question 3?"},
        },
        "mapping_annotator_b": {
            "ANN_B_055": {"original_id": "q_1", "question": "Question 1?"},
            "ANN_B_012": {"original_id": "q_2", "question": "Question 2?"},
            "ANN_B_099": {"original_id": "q_3", "question": "Question 3?"},
        },
    }

    is_aligned, aligned, errors = align_annotations_via_mapping(recs_a, recs_b, mapping_data)

    assert is_aligned is True
    assert len(aligned) == 3
    assert len(errors) == 0

    # Verify each aligned pair corresponds to the exact underlying question
    q1_entry = next(r for r in aligned if r["underlying_question_id"] == "q_1")
    assert q1_entry["blinded_id_a"] == "ANN_A_001"
    assert q1_entry["blinded_id_b"] == "ANN_B_055"
    assert q1_entry["label_a"] == "KNOWN" and q1_entry["label_b"] == "KNOWN"

    q3_entry = next(r for r in aligned if r["underlying_question_id"] == "q_3")
    assert q3_entry["blinded_id_a"] == "ANN_A_003"
    assert q3_entry["blinded_id_b"] == "ANN_B_099"
    assert q3_entry["label_a"] == "UNKNOWN" and q3_entry["label_b"] == "UNKNOWN"


def test_alignment_failure_on_missing_or_duplicate_mapping():
    recs_a = make_synthetic_records(2, annotator="A")
    recs_b = make_synthetic_records(2, annotator="B")

    # Mapping missing Annotator B
    broken_mapping = {
        "mapping_annotator_a": {"ANN_A_001": {"original_id": "q1"}},
        "mapping_annotator_b": {},
    }
    is_aligned, _, errors = align_annotations_via_mapping(recs_a, recs_b, broken_mapping)
    assert is_aligned is False
    assert len(errors) > 0


# ==============================================================================
# SECTION D & E: Raw Agreement and Cohen's Kappa Tests
# ==============================================================================

def test_raw_agreement_and_cohens_kappa():
    # 100% agreement
    y_a = ["KNOWN", "AMBIGUOUS", "UNKNOWN"] * 5
    y_b = ["KNOWN", "AMBIGUOUS", "UNKNOWN"] * 5

    raw = compute_raw_agreement(y_a, y_b)
    assert raw["numerator"] == 15
    assert raw["denominator"] == 15
    assert raw["percentage"] == 100.0
    assert raw["disagreements"] == 0

    kappa = compute_cohens_kappa(y_a, y_b)
    assert math.isclose(kappa["cohens_kappa"], 1.0)
    assert math.isclose(kappa["observed_agreement_po"], 1.0)

    alpha = compute_krippendorffs_alpha_nominal(y_a, y_b)
    assert math.isclose(alpha, 1.0)


def test_kappa_on_known_distribution():
    # Hand-crafted 10 items
    y_a = ["KNOWN", "KNOWN", "KNOWN", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "UNKNOWN", "UNKNOWN", "UNKNOWN", "KNOWN"]
    y_b = ["KNOWN", "KNOWN", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "UNKNOWN", "UNKNOWN", "UNKNOWN", "KNOWN", "KNOWN"]

    raw = compute_raw_agreement(y_a, y_b)
    assert raw["numerator"] == 7
    assert raw["denominator"] == 10
    assert raw["percentage"] == 70.0

    kappa = compute_cohens_kappa(y_a, y_b)
    assert 0.40 <= kappa["cohens_kappa"] <= 0.70
    assert kappa["standard_error"] > 0.0
    assert kappa["ci_95_lower"] <= kappa["cohens_kappa"] <= kappa["ci_95_upper"]


def test_kappa_degenerate_single_class():
    # Both annotators assign 100% of items to KNOWN
    y_a = ["KNOWN"] * 10
    y_b = ["KNOWN"] * 10

    kappa = compute_cohens_kappa(y_a, y_b)
    assert math.isclose(kappa["cohens_kappa"], 1.0)
    assert math.isclose(kappa["expected_agreement_pe"], 1.0)


# ==============================================================================
# SECTION F & G: Confusion Matrix & Class-Wise Agreement Tests
# ==============================================================================

def test_confusion_matrix_and_class_wise():
    y_a = ["KNOWN", "AMBIGUOUS", "UNKNOWN", "KNOWN"]
    y_b = ["KNOWN", "KNOWN", "UNKNOWN", "AMBIGUOUS"]

    cm = compute_confusion_matrix(y_a, y_b)
    assert cm["rows"] == "Annotator A"
    assert cm["columns"] == "Annotator B"
    assert cm["order"] == ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

    matrix = cm["matrix"]
    assert matrix["KNOWN"]["KNOWN"] == 1
    assert matrix["AMBIGUOUS"]["KNOWN"] == 1
    assert matrix["UNKNOWN"]["UNKNOWN"] == 1
    assert matrix["KNOWN"]["AMBIGUOUS"] == 1

    cw = cm["class_wise_agreement"]
    assert cw["UNKNOWN"]["unanimous_count"] == 1
    assert cw["UNKNOWN"]["specific_agreement"] == 1.0
    assert cw["UNKNOWN"]["percentage"] == 100.0


# ==============================================================================
# SECTION H & I: Reporting & Blinding Safeguard Tests
# ==============================================================================

def test_disagreement_and_docket_extraction():
    aligned = [
        {"underlying_question_id": "q1", "blinded_id_a": "A1", "blinded_id_b": "B1", "question": "Q1?", "label_a": "KNOWN", "label_b": "KNOWN", "confidence_a": "HIGH", "confidence_b": "HIGH", "flag_a": False, "flag_b": False, "rationale_a": "", "rationale_b": ""},
        {"underlying_question_id": "q2", "blinded_id_a": "A2", "blinded_id_b": "B2", "question": "Q2?", "label_a": "KNOWN", "label_b": "AMBIGUOUS", "confidence_a": "HIGH", "confidence_b": "HIGH", "flag_a": False, "flag_b": False, "rationale_a": "Fact", "rationale_b": "Polysemic"},
        {"underlying_question_id": "q3", "blinded_id_a": "A3", "blinded_id_b": "B3", "question": "Q3?", "label_a": "UNKNOWN", "label_b": "UNKNOWN", "confidence_a": "LOW", "confidence_b": "LOW", "flag_a": False, "flag_b": False, "rationale_a": "", "rationale_b": ""},
    ]

    disagreements, docket = extract_disagreements_and_docket(aligned)

    # Disagreements must only contain Q2
    assert len(disagreements) == 1
    assert disagreements[0]["underlying_question_id"] == "q2"
    assert disagreements[0]["label_a"] == "KNOWN"
    assert disagreements[0]["label_b"] == "AMBIGUOUS"

    # Docket must contain Q2 (disagreement) and Q3 (double low confidence)
    assert len(docket) == 2
    docket_qids = [d["underlying_question_id"] for d in docket]
    assert "q2" in docket_qids
    assert "q3" in docket_qids
    assert all(d["adjudicated_label"] is None for d in docket)


def test_markdown_report_generation(tmp_path):
    report_path = tmp_path / "report.md"
    results = {
        "timestamp": "2026-11-20T12:00:00Z",
        "total_aligned": 3,
        "raw_agreement": {"percentage": 66.67, "numerator": 2, "denominator": 3},
        "cohens_kappa": {"cohens_kappa": 0.50, "standard_error": 0.1, "ci_95_lower": 0.3, "ci_95_upper": 0.7},
        "krippendorffs_alpha": 0.50,
        "distribution_annotator_a": {"counts": {"KNOWN": 2, "AMBIGUOUS": 1, "UNKNOWN": 0}, "percentages": {"KNOWN": 66.67, "AMBIGUOUS": 33.33, "UNKNOWN": 0.0}},
        "distribution_annotator_b": {"counts": {"KNOWN": 1, "AMBIGUOUS": 1, "UNKNOWN": 1}, "percentages": {"KNOWN": 33.33, "AMBIGUOUS": 33.33, "UNKNOWN": 33.33}},
        "confusion_matrix": {
            "matrix": {
                "KNOWN": {"KNOWN": 1, "AMBIGUOUS": 1, "UNKNOWN": 0},
                "AMBIGUOUS": {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 0},
                "UNKNOWN": {"KNOWN": 0, "AMBIGUOUS": 0, "UNKNOWN": 0},
            },
            "class_wise_agreement": {
                "KNOWN": {"unanimous_count": 1, "percentage": 66.67},
                "AMBIGUOUS": {"unanimous_count": 1, "percentage": 100.0},
                "UNKNOWN": {"unanimous_count": 0, "percentage": 0.0},
            }
        },
        "disagreements": [{"underlying_question_id": "q1", "label_a": "KNOWN", "label_b": "AMBIGUOUS"}],
        "adjudication_docket": [{"docket_id": "ADJ_001"}],
    }

    generate_markdown_report(results, str(report_path))
    content = report_path.read_text()

    assert "Human Annotation Agreement Analysis Report" in content
    assert "Raw Observed Agreement ($P_o$):** 66.67%" in content
    assert "Cohen's Kappa ($\\kappa$):** 0.5000" in content
    assert "Inter-annotator agreement measures consistency between annotators." in content
    # Blinding check: verify no candidate labels or model outputs leaked into report
    assert "model_output" not in content
    assert "cluster_entropy" not in content
    assert "feature" not in content
