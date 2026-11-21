"""Unit tests for BenchmarkDraftValidator and the 300-question draft benchmark."""

import os
import json
import hashlib
import pytest
from src.evaluation.benchmark_validator import (
    BenchmarkDraftValidator,
    TARGET_QUOTAS,
    VALID_LABELS,
)
from src.evaluation.data_validator import DatasetValidator


@pytest.fixture
def draft_benchmark_records():
    path = "data/benchmarks/uncertainty_benchmark_300_draft.jsonl"
    assert os.path.exists(path), f"Benchmark draft file {path} does not exist"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class TestBenchmarkDraftValidator:
    """Verifies schema, quotas, duplicate detection, group integrity, and evidence checking."""

    def test_schema_validation_catches_missing_fields(self):
        """Validator must fail on missing required fields."""
        incomplete_rec = {
            "id": "test_001",
            "question": "What is 2 + 2?",
            "category": "determinate_factual",
            # missing intended_interpretation, gold_label, rationale, evidence, source, group, reviewer_agreement
        }
        report = BenchmarkDraftValidator.validate_records([incomplete_rec])
        assert report.is_valid is False
        assert any("missing or empty required field" in err for err in report.schema_errors)

    def test_schema_validation_catches_invalid_labels_and_categories(self):
        """Validator must reject unknown labels or categories."""
        bad_rec = {
            "id": "test_002",
            "question": "What is 2 + 2?",
            "category": "invalid_category_xyz",
            "intended_interpretation": "arithmetic",
            "gold_label": "DEFINITELY_KNOWN",  # Invalid
            "rationale": "reason",
            "answerability_evidence": {"evidence_type": "consensus_reference", "citations_or_proof": ["Proof"]},
            "source": "math",
            "question_group_id": "grp_test",
            "reviewer_agreement": {"annotation_status": "needs_human_review"},
        }
        report = BenchmarkDraftValidator.validate_records([bad_rec])
        assert report.is_valid is False
        assert any("invalid gold_label" in err for err in report.schema_errors)
        assert any("invalid category" in err for err in report.schema_errors)

    def test_duplicate_question_detection(self):
        """Validator must identify identical questions after normalization."""
        recs = [
            {
                "id": "q1",
                "question": "What is 2 + 2?",
                "category": "determinate_factual",
                "intended_interpretation": "test",
                "gold_label": "KNOWN",
                "rationale": "test",
                "answerability_evidence": {"evidence_type": "consensus_reference", "citations_or_proof": ["P"]},
                "source": "test",
                "question_group_id": "grp_1",
                "reviewer_agreement": {"annotation_status": "needs_human_review"},
            },
            {
                "id": "q2",
                "question": "  what IS 2 + 2 ??? ",  # Same normalized question
                "category": "determinate_factual",
                "intended_interpretation": "test",
                "gold_label": "KNOWN",
                "rationale": "test",
                "answerability_evidence": {"evidence_type": "consensus_reference", "citations_or_proof": ["P"]},
                "source": "test",
                "question_group_id": "grp_1",
                "reviewer_agreement": {"annotation_status": "needs_human_review"},
            },
        ]
        report = BenchmarkDraftValidator.validate_records(recs)
        assert len(report.duplicate_questions) == 1
        assert "what is 2  2" in report.duplicate_questions[0][0] or "what is 2 2" in report.duplicate_questions[0][0]

    def test_missing_evidence_flagging(self):
        """Validator must flag records with unverified or placeholder evidence."""
        recs = [
            {
                "id": "prov_1",
                "question": "What will Apple stock close at in 2035?",
                "category": "unknowable_future_events",
                "intended_interpretation": "future",
                "gold_label": "UNKNOWN",
                "rationale": "future",
                "answerability_evidence": {
                    "evidence_type": "future_contingency",
                    "citations_or_proof": ["Stock prices undetermined"],
                    "verification_status": "flagged_for_human_verification",
                },
                "source": "financial_future_contingency",
                "question_group_id": "grp_fut",
                "reviewer_agreement": {"annotation_status": "needs_human_review"},
            }
        ]
        report = BenchmarkDraftValidator.validate_records(recs)
        assert report.missing_evidence_count == 1
        assert "prov_1" in report.missing_evidence_ids

    def test_false_claim_of_human_verification_rejected(self):
        """Candidate records must not falsely claim 'human_verified' without adjudication log."""
        bad_draft = {
            "id": "fake_gold_1",
            "question": "What is 2 + 2?",
            "category": "determinate_factual",
            "intended_interpretation": "test",
            "gold_label": "KNOWN",
            "rationale": "test",
            "answerability_evidence": {"evidence_type": "consensus_reference", "citations_or_proof": ["P"]},
            "source": "test",
            "question_group_id": "grp_1",
            "reviewer_agreement": {"annotation_status": "gold_benchmark"},  # Falsely claimed
        }
        report = BenchmarkDraftValidator.validate_records([bad_draft])
        assert report.is_valid is False
        assert any("cannot be marked 'gold_benchmark'" in err for err in report.schema_errors)


class TestDraftBenchmark300Artifact:
    """Verifies that the generated uncertainty_benchmark_300_draft.jsonl meets all requirements."""

    def test_exact_total_count(self, draft_benchmark_records):
        assert len(draft_benchmark_records) == 300

    def test_exact_label_balance(self, draft_benchmark_records):
        labels = [r["gold_label"] for r in draft_benchmark_records]
        assert labels.count("KNOWN") == 100
        assert labels.count("AMBIGUOUS") == 100
        assert labels.count("UNKNOWN") == 100

    def test_exact_category_quotas(self, draft_benchmark_records):
        cats = [r["category"] for r in draft_benchmark_records]
        for cat, expected_quota in TARGET_QUOTAS.items():
            actual = cats.count(cat)
            assert actual == expected_quota, f"Category '{cat}': expected {expected_quota}, got {actual}"

    def test_zero_duplicate_questions(self, draft_benchmark_records):
        report = BenchmarkDraftValidator.validate_records(draft_benchmark_records)
        assert len(report.duplicate_questions) == 0, f"Found duplicate questions: {report.duplicate_questions}"

    def test_all_records_have_valid_group_ids(self, draft_benchmark_records):
        for r in draft_benchmark_records:
            gid = r.get("question_group_id")
            assert gid is not None and len(gid) > 0
            assert gid.startswith("grp_")

    def test_100_percent_marked_needs_human_review(self, draft_benchmark_records):
        report = BenchmarkDraftValidator.validate_records(draft_benchmark_records)
        assert report.unverified_candidate_count == 300
        for r in draft_benchmark_records:
            assert r["reviewer_agreement"]["annotation_status"] == "needs_human_review"

    def test_missing_evidence_is_honestly_flagged(self, draft_benchmark_records):
        report = BenchmarkDraftValidator.validate_records(draft_benchmark_records)
        # 40 subjective + 35 private + 35 future questions = 110 flagged items
        assert report.missing_evidence_count == 110

    def test_complete_artifact_validation_passes(self, draft_benchmark_records):
        report = BenchmarkDraftValidator.validate_records(draft_benchmark_records)
        assert report.is_valid is True
        assert len(report.schema_errors) == 0

    def test_circle_corners_questions_cannot_be_trivially_answered_as_zero(self, draft_benchmark_records):
        """Verifies that circle corners false-premise questions ask for corner angle degrees rather than counting corners."""
        circle_recs = [r for r in draft_benchmark_records if r["id"] in ("draft_unk_false_013", "draft_unk_false_014")]
        assert len(circle_recs) == 2
        for r in circle_recs:
            q = r["question"].lower()
            assert "angle measurement" in q or "degree measurement" in q
            # Must not be a simple "how many corners" count that can be trivially answered as "0"
            assert not q.startswith("how many sharp 90-degree corners")

    def test_california_atlantic_coastline_questions_cannot_be_trivially_answered_as_zero(self, draft_benchmark_records):
        """Verifies that California Atlantic coastline questions ask for seaports rather than coastline miles."""
        ca_recs = [r for r in draft_benchmark_records if r["id"] in ("draft_unk_false_021", "draft_unk_false_022")]
        assert len(ca_recs) == 2
        for r in ca_recs:
            q = r["question"].lower()
            assert "seaport" in q or "port city" in q
            # Must not be a simple "how many miles" count that can be trivially answered as "0"
            assert not q.startswith("how many miles of coastline")

    def test_vacuum_wave_speed_adversarial_pair_shared_group_and_split_isolation(self, draft_benchmark_records):
        """Verifies that speed-of-light and speed-of-sound-in-vacuum share grp_adv_vacuum_wave_speed and are kept together."""
        speed_recs = [
            r for r in draft_benchmark_records
            if r["id"] in ("draft_known_fact_011", "draft_known_fact_012", "draft_unk_false_029", "draft_unk_false_030")
        ]
        assert len(speed_recs) == 4
        groups = {r["question_group_id"] for r in speed_recs}
        assert groups == {"grp_adv_vacuum_wave_speed"}

        for r in speed_recs:
            assert r["is_adversarial_variant"] is True

        labels = {r["gold_label"] for r in speed_recs}
        assert labels == {"KNOWN", "UNKNOWN"}  # Cross-label adversarial group

        # Partition entire 300 draft records using group-aware split
        train, val, test, meta = DatasetValidator.group_train_val_test_split(
            draft_benchmark_records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42, return_metadata=True
        )
        assert meta["is_disjoint"] is True

        # Find which partition contains grp_adv_vacuum_wave_speed
        partitions = {"train": train, "val": val, "test": test}
        part_matches = []
        for pname, part in partitions.items():
            matching_ids = {r["id"] for r in part if r["question_group_id"] == "grp_adv_vacuum_wave_speed"}
            if matching_ids:
                part_matches.append((pname, matching_ids))

        # All 4 items must be in exactly one split partition
        assert len(part_matches) == 1, f"Adversarial vacuum wave speed split across partitions: {part_matches}"
        assert len(part_matches[0][1]) == 4

    def test_first_us_president_boundary_clarification(self, draft_benchmark_records):
        """Verifies that draft_known_fact_048 unambiguously specifies under the US Constitution."""
        rec = [r for r in draft_benchmark_records if r["id"] == "draft_known_fact_048"][0]
        assert "Under the United States Constitution" in rec["question"]

    def test_institutional_future_events_have_temporal_anchors(self, draft_benchmark_records):
        """Verifies that time-sensitive future institutional questions contain explicit temporal anchors."""
        target_ids = {"draft_unk_fut_021", "draft_unk_fut_022", "draft_unk_fut_023", "draft_unk_fut_024", "draft_unk_fut_027", "draft_unk_fut_028"}
        anchored_recs = [r for r in draft_benchmark_records if r["id"] in target_ids]
        assert len(anchored_recs) == 6
        for r in anchored_recs:
            assert r.get("temporal_anchor") == "2026-10"
            assert "As of October 2026" in r["question"]

    def test_model_artifact_checksums_preserved(self):
        """Verifies that existing model artifacts have not been altered or retrained."""
        expected_checksums = {
            "models/uncertainty_classifier.joblib": "b49b0704ad56af8602d218aaac59b3c3",
            "models/uncertainty_classifier.json": "7c3487da15c98246bd0c52efd3c6dce8",
            "models/uncertainty_classifier_metadata.json": "3d2b982c97db3bb010e3e20700639183",
        }
        for path, expected_md5 in expected_checksums.items():
            assert os.path.exists(path), f"Model artifact missing: {path}"
            with open(path, "rb") as f:
                actual_md5 = hashlib.md5(f.read()).hexdigest()
            assert actual_md5 == expected_md5, f"Artifact {path} was modified! Expected {expected_md5}, got {actual_md5}"


@pytest.fixture
def candidate_v1_1_records():
    path = "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl"
    assert os.path.exists(path), f"Candidate benchmark v1.1 file {path} does not exist"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class TestCandidateBenchmarkV1_1Artifact:
    """Verifies that the generated uncertainty_benchmark_300_candidate_v1.1.jsonl meets all requirements."""

    def test_candidate_exact_total_count(self, candidate_v1_1_records):
        assert len(candidate_v1_1_records) == 300

    def test_candidate_exact_label_balance(self, candidate_v1_1_records):
        labels = [r["gold_label"] for r in candidate_v1_1_records]
        assert labels.count("KNOWN") == 100
        assert labels.count("AMBIGUOUS") == 100
        assert labels.count("UNKNOWN") == 100

    def test_candidate_exact_category_quotas(self, candidate_v1_1_records):
        cats = [r["category"] for r in candidate_v1_1_records]
        for cat, expected_quota in TARGET_QUOTAS.items():
            actual = cats.count(cat)
            assert actual == expected_quota, f"Category '{cat}': expected {expected_quota}, got {actual}"

    def test_candidate_zero_duplicate_questions(self, candidate_v1_1_records):
        report = BenchmarkDraftValidator.validate_records(candidate_v1_1_records)
        assert len(report.duplicate_questions) == 0, f"Found duplicate questions: {report.duplicate_questions}"

    def test_candidate_group_integrity_and_adversarial_pair(self, candidate_v1_1_records):
        groups = {r["question_group_id"] for r in candidate_v1_1_records}
        assert len(groups) == 152
        speed_recs = [
            r for r in candidate_v1_1_records
            if r["id"] in ("draft_known_fact_011", "draft_known_fact_012", "draft_unk_false_029", "draft_unk_false_030")
        ]
        assert len(speed_recs) == 4
        assert {r["question_group_id"] for r in speed_recs} == {"grp_adv_vacuum_wave_speed"}

    def test_candidate_pope_leo_xiv_successor_questions(self, candidate_v1_1_records):
        """Verifies draft_unk_fut_023 and 024 query successor to Pope Leo XIV with Nov 2026 anchor."""
        pope_recs = [r for r in candidate_v1_1_records if r["id"] in ("draft_unk_fut_023", "draft_unk_fut_024")]
        assert len(pope_recs) == 2
        for r in pope_recs:
            assert "Pope Leo XIV" in r["question"]
            assert "Pope Francis" not in r["question"]
            assert "As of November 2026" in r["question"]
            assert r.get("temporal_anchor") == "2026-11"
            assert r["gold_label"] == "UNKNOWN"
            assert "audit_trail" in r

    def test_candidate_all_temporal_anchors_standardized_to_november_2026(self, candidate_v1_1_records):
        """Verifies that all 6 anchored future queries use 2026-11 and As of November 2026."""
        anchored_ids = {"draft_unk_fut_021", "draft_unk_fut_022", "draft_unk_fut_023", "draft_unk_fut_024", "draft_unk_fut_027", "draft_unk_fut_028"}
        anchored_recs = [r for r in candidate_v1_1_records if r["id"] in anchored_ids]
        assert len(anchored_recs) == 6
        for r in anchored_recs:
            assert r.get("temporal_anchor") == "2026-11"
            assert "As of November 2026" in r["question"]
            assert "October 2026" not in r["question"]

    def test_candidate_affected_borderline_records_carry_audit_trails(self, candidate_v1_1_records):
        """Verifies that all 13 affected borderline records have audit_trail fields."""
        affected_ids = {
            "draft_unk_false_013", "draft_unk_false_014",
            "draft_unk_false_021", "draft_unk_false_022",
            "draft_known_fact_048",
            "draft_known_fact_029", "draft_known_fact_030",
            "draft_unk_fut_023", "draft_unk_fut_024",
            "draft_unk_fut_027", "draft_unk_fut_028",
            "draft_ambig_poly_055",
            "draft_unk_priv_023",
        }
        affected_recs = [r for r in candidate_v1_1_records if r["id"] in affected_ids]
        assert len(affected_recs) == 13
        for r in affected_recs:
            assert "audit_trail" in r, f"Record {r['id']} missing audit_trail!"

    def test_candidate_and_draft_checksums_match_expectations(self):
        """Verifies that the original draft is untouched and candidate v1.1 matches its hash."""
        with open("data/benchmarks/uncertainty_benchmark_300_draft.jsonl", "rb") as f:
            draft_sha = hashlib.sha256(f.read()).hexdigest()
        assert draft_sha == "9f05fbd3ffcf9b3e76d3720e5d18cd027b4a3e44e0962caeefad7b1ce298d412"

        with open("data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl", "rb") as f:
            cand_sha = hashlib.sha256(f.read()).hexdigest()
        assert cand_sha == "59413b026a8948936961714a1b58fcc3231f9df25bc237b3f046f086962ce220"


