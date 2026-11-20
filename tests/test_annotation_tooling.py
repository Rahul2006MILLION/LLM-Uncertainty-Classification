"""Unit tests for human annotation validation tooling and annotation interface.

Uses synthetic test records only to verify validation rules, edge cases,
error reporting, and persistence integrity without touching production benchmark data.
"""

import json
import os
import tempfile
import pytest

from scripts.validate_annotations import (
    AnnotationValidator,
    VALID_LABELS,
    VALID_CONFIDENCES,
)
from scripts.annotate_tool import _atomic_save


def create_synthetic_records(count=5, annotator="A", label="KNOWN", confidence="HIGH"):
    """Helper to generate valid synthetic annotation records."""
    records = []
    for i in range(1, count + 1):
        records.append({
            "blinded_id": f"ANN_{annotator}_{i:03d}",
            "question": f"Synthetic test query {i} with sufficient character length?",
            "assigned_label": label,
            "confidence": confidence,
            "rationale": f"Synthetic test rationale for query {i}",
            "requires_adjudication": False,
            "notes": "",
        })
    return records


def write_jsonl(filepath, records):
    """Helper to write records to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_valid_annotation_records(tmp_path):
    """Test that perfectly valid records pass validation with is_valid=True."""
    fpath = tmp_path / "valid_annotations.jsonl"
    records = create_synthetic_records(count=10, annotator="A")
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=10)
    report = validator.validate_file(str(fpath), expected_annotator="A")

    assert report.is_valid is True
    assert report.total_records == 10
    assert report.valid_records == 10
    assert report.unannotated_count == 0
    assert report.annotator_id == "A"
    assert report.label_distribution["KNOWN"] == 10
    assert len(report.errors) == 0


def test_missing_labels(tmp_path):
    """Test detection of null/missing assigned_label."""
    fpath = tmp_path / "missing_labels.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[2]["assigned_label"] = None  # Unannotated
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5, allow_unannotated=False)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert report.unannotated_count == 1
    assert any("null (unannotated)" in e for e in report.errors)


def test_invalid_labels(tmp_path):
    """Test detection of invalid label values outside KNOWN/AMBIGUOUS/UNKNOWN."""
    fpath = tmp_path / "invalid_labels.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[1]["assigned_label"] = "UNCERTAIN"  # Invalid class
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Invalid 'assigned_label' 'UNCERTAIN'" in e for e in report.errors)


def test_invalid_confidence_values(tmp_path):
    """Test detection of invalid confidence outside HIGH/MEDIUM/LOW."""
    fpath = tmp_path / "invalid_confidence.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[3]["confidence"] = "VERY_HIGH"  # Invalid confidence
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Invalid 'confidence' 'VERY_HIGH'" in e for e in report.errors)


def test_duplicate_blinded_ids(tmp_path):
    """Test detection of duplicate blinded IDs."""
    fpath = tmp_path / "duplicate_ids.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[4]["blinded_id"] = "ANN_A_001"  # Duplicate of record 1
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Duplicate blinded_id 'ANN_A_001'" in e for e in report.errors)


def test_missing_blinded_ids_in_sequence(tmp_path):
    """Test detection of missing IDs in sequence."""
    fpath = tmp_path / "missing_ids.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[2]["blinded_id"] = "ANN_A_010"  # Skips 003, introduces 010
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Missing expected IDs" in e for e in report.errors)


def test_incorrect_record_count(tmp_path):
    """Test error when record count does not match expected count."""
    fpath = tmp_path / "wrong_count.jsonl"
    records = create_synthetic_records(count=4, annotator="A")
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Incorrect record count: found 4, expected 5" in e for e in report.errors)


def test_malformed_jsonl(tmp_path):
    """Test detection of broken JSON lines."""
    fpath = tmp_path / "malformed.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write('{"blinded_id": "ANN_A_001", "question": "valid?"}\n')
        f.write('{"blinded_id": "ANN_A_002", BROKEN_JSON\n')

    validator = AnnotationValidator(expected_count=2)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("Malformed JSON" in e for e in report.errors)


def test_annotator_identity_mismatch(tmp_path):
    """Test detection when file expected for Annotator A contains Annotator B IDs."""
    fpath = tmp_path / "mismatch.jsonl"
    records = create_synthetic_records(count=5, annotator="B")
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath), expected_annotator="A")

    assert report.is_valid is False
    assert any("does not match expected annotator 'A'" in e for e in report.errors)


def test_unexpected_fields_and_leakage(tmp_path):
    """Test rejection of records with extra fields or prohibited metadata leakage."""
    fpath = tmp_path / "leakage.jsonl"
    records = create_synthetic_records(count=3, annotator="A")
    records[0]["candidate_label"] = "KNOWN"  # Prohibited leakage key
    records[1]["unauthorized_field"] = "foo"  # Schema violation
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=3)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert any("CRITICAL LEAKAGE" in e for e in report.errors)
    assert any("Disallowed extra fields" in e for e in report.errors)


def test_atomic_persistence_and_resume(tmp_path):
    """Test atomic file saving and resumption functionality."""
    fpath = tmp_path / "session_save.jsonl"
    records = create_synthetic_records(count=5, annotator="A", label=None, confidence=None)
    write_jsonl(fpath, records)

    # Initial state has 5 unannotated records
    val_init = AnnotationValidator(expected_count=5, allow_unannotated=True)
    rep_init = val_init.validate_file(str(fpath))
    assert rep_init.unannotated_count == 5

    # Simulate annotator annotating item 1 & 2
    records[0]["assigned_label"] = "KNOWN"
    records[0]["confidence"] = "HIGH"
    records[0]["rationale"] = "Objective consensus fact"
    _atomic_save(str(fpath), records)

    # Verify reloaded state matches updated values
    with open(fpath, "r", encoding="utf-8") as f:
        reloaded = [json.loads(line) for line in f]

    assert reloaded[0]["assigned_label"] == "KNOWN"
    assert reloaded[0]["confidence"] == "HIGH"
    assert reloaded[0]["rationale"] == "Objective consensus fact"
    assert reloaded[1]["assigned_label"] is None

    rep_updated = val_init.validate_file(str(fpath))
    assert rep_updated.unannotated_count == 4
    assert rep_updated.label_distribution["KNOWN"] == 1
