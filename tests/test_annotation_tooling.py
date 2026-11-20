"""Unit tests for human annotation validation tooling and annotation interface.

Uses synthetic test records only to verify validation rules, edge cases,
error reporting, isolation, navigation, editing, skipping, and persistence integrity
without touching production benchmark data.
"""

import json
import os
import tempfile
import pytest

from scripts.validate_annotations import (
    AnnotationValidator,
    VALID_LABELS,
    VALID_CONFIDENCES,
    DEFAULT_PATHS,
)
from scripts.annotate_tool import (
    _atomic_save,
    DEFAULT_ANNOTATOR_PATHS,
)


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
    """Test that perfectly valid records pass validation with is_valid=True and is_complete=True."""
    fpath = tmp_path / "valid_annotations.jsonl"
    records = create_synthetic_records(count=10, annotator="A")
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=10)
    report = validator.validate_file(str(fpath), expected_annotator="A")

    assert report.is_valid is True
    assert report.is_complete is True
    assert report.total_records == 10
    assert report.completed_annotations == 10
    assert report.remaining_annotations == 0
    assert report.annotator_id == "A"
    assert report.label_distribution["KNOWN"] == 10
    assert len(report.errors) == 0


def test_annotator_isolation_and_paths():
    """Test that Annotator A and B default paths are strictly isolated and disjoint."""
    path_a = DEFAULT_ANNOTATOR_PATHS.get("A")
    path_b = DEFAULT_ANNOTATOR_PATHS.get("B")

    assert path_a is not None and path_b is not None
    assert path_a != path_b
    assert "annotator_a" in path_a
    assert "annotator_b" in path_b
    assert DEFAULT_PATHS["A"] == path_a
    assert DEFAULT_PATHS["B"] == path_b


def test_missing_and_null_labels(tmp_path):
    """Test detection of null/missing assigned_label and completion status."""
    fpath = tmp_path / "missing_labels.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[2]["assigned_label"] = None  # Unannotated

    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5, allow_unannotated=False)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert report.is_complete is False
    assert report.completed_annotations == 4
    assert report.remaining_annotations == 1
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
    assert report.invalid_label_count == 1
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
    assert report.duplicate_id_count == 1
    assert any("Duplicate blinded_id 'ANN_A_001'" in e for e in report.errors)


def test_missing_and_unexpected_blinded_ids(tmp_path):
    """Test detection of missing and unexpected IDs."""
    fpath = tmp_path / "unexpected_ids.jsonl"
    records = create_synthetic_records(count=5, annotator="A")
    records[2]["blinded_id"] = "ANN_A_999"  # Unexpected ID
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath), expected_annotator="A")

    assert report.is_valid is False
    assert report.unexpected_id_count >= 1
    assert report.missing_id_count >= 1
    assert any("not in expected ID range" in e for e in report.errors)


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
    """Test detection of broken JSON lines and non-dict records."""
    fpath = tmp_path / "malformed.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write('{"blinded_id": "ANN_A_001", "question": "valid?"}\n')
        f.write('{"blinded_id": "ANN_A_002", BROKEN_JSON\n')
        f.write('"just a string not an object"\n')

    validator = AnnotationValidator(expected_count=3)
    report = validator.validate_file(str(fpath))

    assert report.is_valid is False
    assert report.malformed_record_count >= 2
    assert any("Malformed JSON" in e for e in report.errors)
    assert any("not a JSON object" in e for e in report.errors)


def test_annotator_identity_mismatch(tmp_path):
    """Test detection when file expected for Annotator A contains Annotator B IDs."""
    fpath = tmp_path / "mismatch.jsonl"
    records = create_synthetic_records(count=5, annotator="B")
    write_jsonl(fpath, records)

    validator = AnnotationValidator(expected_count=5)
    report = validator.validate_file(str(fpath), expected_annotator="A")

    assert report.is_valid is False
    assert report.unexpected_id_count == 5
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
    assert report.leakage_detected is True
    assert any("CRITICAL LEAKAGE" in e for e in report.errors)
    assert any("Disallowed extra fields" in e for e in report.errors)


def test_atomic_persistence_editing_and_resume(tmp_path):
    """Test incremental atomic saving, editing, skipping, and session resumption."""
    fpath = tmp_path / "session_save.jsonl"
    records = create_synthetic_records(count=5, annotator="A", label=None, confidence=None)
    write_jsonl(fpath, records)

    # 1. Initial state: 0 completed, 5 remaining
    val_init = AnnotationValidator(expected_count=5, allow_unannotated=True)
    rep_init = val_init.validate_file(str(fpath))
    assert rep_init.completed_annotations == 0
    assert rep_init.remaining_annotations == 5

    # 2. Annotate question 1
    records[0]["assigned_label"] = "KNOWN"
    records[0]["confidence"] = "HIGH"
    records[0]["rationale"] = "Factual truth"
    _atomic_save(str(fpath), records)

    # 3. Skip question 2 (leave assigned_label=None), annotate question 3
    records[2]["assigned_label"] = "UNKNOWN"
    records[2]["confidence"] = "LOW"
    records[2]["rationale"] = "Presupposition failure"
    _atomic_save(str(fpath), records)

    # 4. Resume check: reload from disk
    with open(fpath, "r", encoding="utf-8") as f:
        reloaded = [json.loads(line) for line in f]

    assert reloaded[0]["assigned_label"] == "KNOWN"
    assert reloaded[1]["assigned_label"] is None  # Skipped
    assert reloaded[2]["assigned_label"] == "UNKNOWN"

    rep_step2 = val_init.validate_file(str(fpath))
    assert rep_step2.completed_annotations == 2
    assert rep_step2.remaining_annotations == 3

    # 5. Edit question 1 (change from KNOWN to AMBIGUOUS)
    reloaded[0]["assigned_label"] = "AMBIGUOUS"
    reloaded[0]["confidence"] = "MEDIUM"
    reloaded[0]["rationale"] = "Revised to polysemic term"
    _atomic_save(str(fpath), reloaded)

    # 6. Verify edit was saved
    with open(fpath, "r", encoding="utf-8") as f:
        reloaded_after_edit = [json.loads(line) for line in f]
    assert reloaded_after_edit[0]["assigned_label"] == "AMBIGUOUS"
    assert reloaded_after_edit[0]["confidence"] == "MEDIUM"


def test_blinding_safeguard_key_quarantine():
    """Verify that the annotation tool and validator code do not import or open the blinding key."""
    import inspect
    import scripts.annotate_tool as at
    import scripts.validate_annotations as va

    at_src = inspect.getsource(at)
    va_src = inspect.getsource(va)

    assert "annotation_blinding_key.json" not in at_src
    assert "annotation_blinding_key.json" not in va_src
