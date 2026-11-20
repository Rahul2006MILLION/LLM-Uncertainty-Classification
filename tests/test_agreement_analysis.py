"""Unit tests for inter-annotator agreement analysis and adjudication docket tooling.

Uses synthetic test records only to verify Cohen's Kappa, Krippendorff's Alpha,
confusion matrix calculations, disagreement extraction, docket formatting,
and blinding-key release gate conditions.
"""

import json
import math
import pytest

from scripts.analyze_annotation_agreement import (
    compute_raw_agreement,
    compute_expected_agreement,
    compute_cohens_kappa,
    compute_krippendorffs_alpha_nominal,
    compute_confusion_matrix,
    extract_disagreements_and_docket,
    check_blinding_release_gate,
    CLASSES,
)


def test_perfect_agreement():
    """Test metrics when annotators agree on 100% of items."""
    y_a = ["KNOWN", "AMBIGUOUS", "UNKNOWN", "KNOWN", "AMBIGUOUS", "UNKNOWN"]
    y_b = ["KNOWN", "AMBIGUOUS", "UNKNOWN", "KNOWN", "AMBIGUOUS", "UNKNOWN"]

    po = compute_raw_agreement(y_a, y_b)
    assert po == 1.0

    kappa_res = compute_cohens_kappa(y_a, y_b)
    assert math.isclose(kappa_res["cohens_kappa"], 1.0)
    assert kappa_res["raw_agreement_po"] == 1.0

    alpha = compute_krippendorffs_alpha_nominal(y_a, y_b)
    assert math.isclose(alpha, 1.0)


def test_chance_agreement_and_zero_kappa():
    """Test scenario with zero or near-zero agreement beyond chance."""
    # Annotator A assigns everything KNOWN, Annotator B assigns everything AMBIGUOUS
    y_a = ["KNOWN"] * 10
    y_b = ["AMBIGUOUS"] * 10

    po = compute_raw_agreement(y_a, y_b)
    assert po == 0.0

    kappa_res = compute_cohens_kappa(y_a, y_b)
    assert kappa_res["cohens_kappa"] <= 0.0


def test_known_hand_calculated_kappa():
    """Test standard 3x3 scenario against manual calculation."""
    # 10 items
    y_a = ["KNOWN", "KNOWN", "KNOWN", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "UNKNOWN", "UNKNOWN", "UNKNOWN", "KNOWN"]
    y_b = ["KNOWN", "KNOWN", "AMBIGUOUS", "AMBIGUOUS", "AMBIGUOUS", "UNKNOWN", "UNKNOWN", "UNKNOWN", "KNOWN", "KNOWN"]
    # Agreements:
    # 1: K == K (yes)
    # 2: K == K (yes)
    # 3: K != A (no)
    # 4: A == A (yes)
    # 5: A == A (yes)
    # 6: A != U (no)
    # 7: U == U (yes)
    # 8: U == U (yes)
    # 9: U != K (no)
    # 10: K == K (yes)
    # 7 / 10 = 0.70 Po
    po = compute_raw_agreement(y_a, y_b)
    assert math.isclose(po, 0.70)

    kappa_res = compute_cohens_kappa(y_a, y_b)
    assert 0.40 <= kappa_res["cohens_kappa"] <= 0.70
    assert kappa_res["standard_error"] > 0.0


def test_confusion_matrix_structure():
    """Test that 3x3 confusion matrix and per-class metrics are computed correctly."""
    y_a = ["KNOWN", "AMBIGUOUS", "UNKNOWN", "KNOWN"]
    y_b = ["KNOWN", "KNOWN", "UNKNOWN", "AMBIGUOUS"]

    res = compute_confusion_matrix(y_a, y_b)
    matrix = res["matrix"]

    assert matrix["KNOWN"]["KNOWN"] == 1
    assert matrix["AMBIGUOUS"]["KNOWN"] == 1
    assert matrix["UNKNOWN"]["UNKNOWN"] == 1
    assert matrix["KNOWN"]["AMBIGUOUS"] == 1

    per_class = res["per_class"]
    assert "KNOWN" in per_class
    assert "AMBIGUOUS" in per_class
    assert "UNKNOWN" in per_class
    assert per_class["UNKNOWN"]["unanimous_count"] == 1
    assert per_class["UNKNOWN"]["f1_score"] == 1.0


def test_disagreement_and_docket_extraction():
    """Test extraction of disagreements and construction of adjudication docket."""
    rec_a = [
        {"blinded_id": "ANN_A_001", "question": "Query 1?", "assigned_label": "KNOWN", "confidence": "HIGH", "requires_adjudication": False, "rationale": "Clear fact"},
        {"blinded_id": "ANN_A_002", "question": "Query 2?", "assigned_label": "AMBIGUOUS", "confidence": "MEDIUM", "requires_adjudication": True, "rationale": "Polysemic"},
        {"blinded_id": "ANN_A_003", "question": "Query 3?", "assigned_label": "UNKNOWN", "confidence": "LOW", "requires_adjudication": False, "rationale": "Unknowable"},
    ]
    rec_b = [
        {"blinded_id": "ANN_B_001", "question": "Query 1?", "assigned_label": "KNOWN", "confidence": "HIGH", "requires_adjudication": False, "rationale": "Clear fact"},
        {"blinded_id": "ANN_B_002", "question": "Query 2?", "assigned_label": "KNOWN", "confidence": "HIGH", "requires_adjudication": False, "rationale": "Default sense"},
        {"blinded_id": "ANN_B_003", "question": "Query 3?", "assigned_label": "UNKNOWN", "confidence": "LOW", "requires_adjudication": False, "rationale": "Unknowable"},
    ]

    disagreements, docket = extract_disagreements_and_docket(rec_a, rec_b)

    # Disagreement should find Query 2
    assert len(disagreements) == 1
    assert disagreements[0]["question"] == "Query 2?"
    assert disagreements[0]["label_a"] == "AMBIGUOUS"
    assert disagreements[0]["label_b"] == "KNOWN"

    # Docket should include Query 2 (disagreement + flagged) and Query 3 (double low confidence)
    assert len(docket) == 2
    docket_questions = [d["question"] for d in docket]
    assert "Query 2?" in docket_questions
    assert "Query 3?" in docket_questions
    assert all(d["docket_id"].startswith("ADJ_") for d in docket)
    assert all(d["adjudicated_label"] is None for d in docket)


def test_blinding_release_gate_conditions(tmp_path):
    """Test that release gate correctly passes or rejects under varying conditions."""
    f_a = tmp_path / "a.jsonl"
    f_b = tmp_path / "b.jsonl"
    f_a.write_text("dummy")
    f_b.write_text("dummy")

    # Incomplete records (contains None)
    incomplete_a = [{"assigned_label": None}] * 5
    incomplete_b = [{"assigned_label": "KNOWN"}] * 5
    can_release, failures = check_blinding_release_gate(str(f_a), str(f_b), incomplete_a, incomplete_b, expected_count=5)
    assert can_release is False
    assert any("unannotated records" in f for f in failures)

    # Count mismatch
    short_a = [{"assigned_label": "KNOWN"}] * 4
    can_release, failures = check_blinding_release_gate(str(f_a), str(f_b), short_a, incomplete_b, expected_count=5)
    assert can_release is False
    assert any("count mismatch" in f for f in failures)

    # Invalid label
    invalid_a = [{"assigned_label": "NOT_A_LABEL"}] * 5
    can_release, failures = check_blinding_release_gate(str(f_a), str(f_b), invalid_a, incomplete_b, expected_count=5)
    assert can_release is False
    assert any("invalid label" in f for f in failures)

    # All conditions satisfied
    valid_a = [{"assigned_label": "KNOWN"}] * 5
    valid_b = [{"assigned_label": "UNKNOWN"}] * 5
    can_release, failures = check_blinding_release_gate(str(f_a), str(f_b), valid_a, valid_b, expected_count=5)
    assert can_release is True
    assert len(failures) == 0
