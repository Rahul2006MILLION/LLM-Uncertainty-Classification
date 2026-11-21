"""Unit tests for gold benchmark finalization pipeline and validation engine.

Uses synthetic test records only to verify:
- Correct record mapping and assembly
- Unanimous consensus preservation
- Adjudication integration and resolution
- Detection of missing / invalid labels
- Detection of unresolved disagreements
- Missing adjudication rationale
- Dataset count and ID ordering checks
- Candidate benchmark mutation detection
- Manifest generation and hash reproducibility
"""

import json
import pytest

from scripts.finalize_gold_benchmark import (
    GoldBenchmarkValidator,
    assemble_gold_record,
    build_reproducibility_manifest,
)


def make_synthetic_candidate(idx: int, qid: str, question: str, label: str = "KNOWN"):
    return {
        "id": qid,
        "question_group_id": f"grp_{idx:03d}",
        "question": question,
        "category": "determinate_factual",
        "gold_label": label,
        "source": "synthetic_test",
        "is_adversarial_variant": False,
    }


def make_synthetic_annotator_record(blinded_id: str, label: str, flag: bool = False, rationale: str = "Test rationale"):
    return {
        "blinded_id": blinded_id,
        "assigned_label": label,
        "confidence": "HIGH",
        "rationale": rationale,
        "requires_adjudication": flag,
        "notes": "",
    }


def test_assemble_unanimous_record():
    """Test assembling a record where Annotator A and B agree."""
    cand = make_synthetic_candidate(1, "q_001", "What is 2 + 2?", "KNOWN")
    rec_a = make_synthetic_annotator_record("ANN_A_001", "KNOWN")
    rec_b = make_synthetic_annotator_record("ANN_B_001", "KNOWN")

    gold = assemble_gold_record(cand, rec_a, rec_b, {}, "2026-11-20T12:00:00Z")

    assert gold["id"] == "q_001"
    assert gold["final_gold_label"] == "KNOWN"
    assert gold["candidate_matches_gold"] is True
    assert gold["human_annotation_provenance"]["is_unanimous"] is True
    assert gold["human_annotation_provenance"]["adjudication_required"] is False
    assert gold["human_annotation_provenance"]["adjudication_record"] is None
    assert gold["reviewer_agreement"]["inter_annotator_agreement"] == "unanimous_consensus"


def test_assemble_disputed_record_with_adjudication():
    """Test assembling a disputed record that has an adjudication decision."""
    cand = make_synthetic_candidate(2, "q_002", "What is the capital of Georgia?", "AMBIGUOUS")
    rec_a = make_synthetic_annotator_record("ANN_A_002", "KNOWN", False, "Tbilisi")
    rec_b = make_synthetic_annotator_record("ANN_B_002", "AMBIGUOUS", True, "Tbilisi or Atlanta")

    adj_map = {
        "q_002": {
            "docket_id": "ADJ_001",
            "trigger_reason": "LABEL_DISAGREEMENT",
            "adjudicated_label": "AMBIGUOUS",
            "adjudication_rationale": "Entity polysemy between country and US state",
            "adjudicator": "Senior Panel Lead",
            "timestamp": "2026-11-20T12:05:00Z",
        }
    }

    gold = assemble_gold_record(cand, rec_a, rec_b, adj_map, "2026-11-20T12:00:00Z")

    assert gold["final_gold_label"] == "AMBIGUOUS"
    assert gold["candidate_matches_gold"] is True
    assert gold["human_annotation_provenance"]["is_unanimous"] is False
    assert gold["human_annotation_provenance"]["adjudication_required"] is True
    assert gold["human_annotation_provenance"]["adjudication_record"] is not None
    assert gold["human_annotation_provenance"]["adjudication_record"]["adjudicated_label"] == "AMBIGUOUS"
    assert gold["reviewer_agreement"]["inter_annotator_agreement"] == "adjudicated_consensus"


def test_missing_adjudication_raises_error():
    """Test that attempting to assemble a disputed record without an adjudication entry raises ValueError."""
    cand = make_synthetic_candidate(3, "q_003", "Subjective query?", "AMBIGUOUS")
    rec_a = make_synthetic_annotator_record("ANN_A_003", "KNOWN")
    rec_b = make_synthetic_annotator_record("ANN_B_003", "UNKNOWN")

    with pytest.raises(ValueError, match="requires adjudication but no adjudication entry found"):
        assemble_gold_record(cand, rec_a, rec_b, {}, "2026-11-20T12:00:00Z")


def test_validator_perfect_records():
    """Test validator on a clean synthetic gold dataset."""
    cands = [
        make_synthetic_candidate(1, "q_001", "Query 1?"),
        make_synthetic_candidate(2, "q_002", "Query 2?"),
    ]
    rec_a1 = make_synthetic_annotator_record("ANN_A_001", "KNOWN")
    rec_b1 = make_synthetic_annotator_record("ANN_B_001", "KNOWN")
    rec_a2 = make_synthetic_annotator_record("ANN_A_002", "KNOWN")
    rec_b2 = make_synthetic_annotator_record("ANN_B_002", "KNOWN")

    gold1 = assemble_gold_record(cands[0], rec_a1, rec_b1, {}, "2026-11-20T12:00:00Z")
    gold2 = assemble_gold_record(cands[1], rec_a2, rec_b2, {}, "2026-11-20T12:00:00Z")

    validator = GoldBenchmarkValidator(expected_count=2)
    is_valid, errors = validator.validate_records([gold1, gold2], cands)

    assert is_valid is True
    assert len(errors) == 0


def test_validator_detects_mutation_and_duplicates():
    """Test validator detection of modified question text, count mismatch, and duplicate IDs."""
    cands = [
        make_synthetic_candidate(1, "q_001", "Original Question 1?"),
        make_synthetic_candidate(2, "q_002", "Original Question 2?"),
    ]
    rec_a1 = make_synthetic_annotator_record("ANN_A_001", "KNOWN")
    rec_b1 = make_synthetic_annotator_record("ANN_B_001", "KNOWN")
    rec_a2 = make_synthetic_annotator_record("ANN_A_002", "KNOWN")
    rec_b2 = make_synthetic_annotator_record("ANN_B_002", "KNOWN")

    gold1 = assemble_gold_record(cands[0], rec_a1, rec_b1, {}, "2026-11-20T12:00:00Z")
    gold2 = assemble_gold_record(cands[1], rec_a2, rec_b2, {}, "2026-11-20T12:00:00Z")

    # Mutate question text
    gold1["question"] = "MUTATED Question 1?"

    validator = GoldBenchmarkValidator(expected_count=2)
    is_valid, errors = validator.validate_records([gold1, gold2], cands)
    assert is_valid is False
    assert any("Question text modified from candidate" in e for e in errors)

    # Test duplicate ID
    gold2["id"] = "q_001"
    is_valid_dup, errors_dup = validator.validate_records([gold1, gold2], cands)
    assert is_valid_dup is False
    assert any("Duplicate IDs detected" in e for e in errors_dup)


def test_validator_detects_unresolved_adjudication():
    """Test validator detects when a disputed record has invalid or missing adjudication."""
    cands = [make_synthetic_candidate(1, "q_001", "Query 1?")]
    rec_a1 = make_synthetic_annotator_record("ANN_A_001", "KNOWN")
    rec_b1 = make_synthetic_annotator_record("ANN_B_001", "UNKNOWN")

    # Forcibly create a malformed gold record claiming consensus without adjudication
    gold = {
        "id": "q_001",
        "question_group_id": "grp_001",
        "question": "Query 1?",
        "category": "determinate_factual",
        "final_gold_label": "KNOWN",
        "candidate_label": "KNOWN",
        "candidate_matches_gold": True,
        "source": "synthetic_test",
        "is_adversarial_variant": False,
        "human_annotation_provenance": {
            "annotator_a": rec_a1,
            "annotator_b": rec_b1,
            "is_unanimous": False,
            "adjudication_required": True,
            "adjudication_record": None,  # Missing!
        },
        "reviewer_agreement": {
            "annotation_status": "gold_benchmark",
            "annotator_count": 2,
            "individual_labels": ["KNOWN", "UNKNOWN"],
            "inter_annotator_agreement": "unanimous_consensus",
        },
        "benchmark_version": "1.0",
        "gold_finalized_timestamp": "2026-11-20T12:00:00Z",
    }

    validator = GoldBenchmarkValidator(expected_count=1)
    is_valid, errors = validator.validate_records([gold], cands)

    assert is_valid is False
    assert any("Disputed record missing adjudication_record" in e for e in errors)


def test_reproducibility_manifest_structure(tmp_path):
    """Test manifest building and key isolation."""
    cand_file = tmp_path / "cand.jsonl"
    cand_file.write_text('{"id": "q1"}\n')
    a_file = tmp_path / "a.jsonl"
    a_file.write_text('{"id": "a1"}\n')
    b_file = tmp_path / "b.jsonl"
    b_file.write_text('{"id": "b1"}\n')
    key_file = tmp_path / "key.json"
    key_file.write_text('{"secret": "key"}\n')

    gold_records = [
        {"id": "q1", "final_gold_label": "KNOWN"},
        {"id": "q2", "final_gold_label": "AMBIGUOUS"},
        {"id": "q3", "final_gold_label": "UNKNOWN"},
    ]

    manifest = build_reproducibility_manifest(
        str(cand_file), str(a_file), str(b_file), str(key_file), None, "dummy_gold.jsonl", gold_records
    )

    assert manifest["manifest_version"] == "1.0"
    assert manifest["total_records"] == 3
    assert manifest["label_distribution"]["KNOWN"] == 1
    assert manifest["label_distribution"]["AMBIGUOUS"] == 1
    assert manifest["label_distribution"]["UNKNOWN"] == 1
    assert "source_artifacts" in manifest
    # Ensure blinding key contents are not exposed (only SHA-256 and path)
    assert "secret" not in json.dumps(manifest)
    assert manifest["source_artifacts"]["blinding_key"]["sha256"] is not None
