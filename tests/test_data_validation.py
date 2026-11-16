"""Unit tests for dataset schema validation and group-based splitting."""

import pytest
from data.schema import QuestionRecord, ExperimentRecord
from src.evaluation.data_validator import DatasetValidator


class TestDataValidation:
    """Tests schema enforcement, group-based splitting, and leakage prevention."""

    def test_question_record_schema_valid(self):
        rec = QuestionRecord(
            id="q_001",
            question="What is the capital of France?",
            label="KNOWN",
            domain="geography",
            source="test_fixture",
            ground_truth_answers=["Paris"],
        )
        assert rec.label == "KNOWN"
        assert rec.id == "q_001"

    def test_invalid_label_rejected(self):
        with pytest.raises(Exception):
            QuestionRecord(
                id="q_002",
                question="Invalid question?",
                label="INVALID_LABEL",  # must be KNOWN, AMBIGUOUS, or UNKNOWN
                source="test",
            )

    def test_group_train_val_test_split_no_leakage(self):
        """Verifies that duplicate questions are not split between train and test."""
        records = [
            {"id": "1", "question": "What is the capital of France?", "label": "KNOWN", "source": "s"},
            {"id": "2", "question": "What is the capital of Spain?", "label": "KNOWN", "source": "s"},
            {"id": "3", "question": "Who is Jordan?", "label": "AMBIGUOUS", "source": "s"},
            {"id": "4", "question": "Who is Jordan?", "label": "AMBIGUOUS", "source": "s"},  # Duplicate
            {"id": "5", "question": "What number am I thinking of?", "label": "UNKNOWN", "source": "s"},
            {"id": "6", "question": "What number am I thinking of?", "label": "UNKNOWN", "source": "s"},  # Duplicate
            {"id": "7", "question": "What is 2 + 2?", "label": "KNOWN", "source": "s"},
            {"id": "8", "question": "What is 3 + 3?", "label": "KNOWN", "source": "s"},
            {"id": "9", "question": "What is the largest city in the world?", "label": "AMBIGUOUS", "source": "s"},
            {"id": "10", "question": "What will happen tomorrow?", "label": "UNKNOWN", "source": "s"},
        ]

        train, val, test, meta = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42, return_metadata=True
        )

        train_groups = {DatasetValidator.get_question_group_key(r["question"]) for r in train}
        val_groups = {DatasetValidator.get_question_group_key(r["question"]) for r in val}
        test_groups = {DatasetValidator.get_question_group_key(r["question"]) for r in test}

        # Groups must not overlap between any partitions
        assert train_groups.isdisjoint(test_groups), f"Train/Test leakage: {train_groups & test_groups}"
        assert train_groups.isdisjoint(val_groups), f"Train/Val leakage: {train_groups & val_groups}"
        assert val_groups.isdisjoint(test_groups), f"Val/Test leakage: {val_groups & test_groups}"
        assert meta["is_disjoint"] is True

    def test_duplicate_detection_and_conflicting_labels(self):
        """Verifies detection of exact duplicate questions and flags conflicting labels for manual review."""
        records = [
            {"id": "q1", "question": "Is Pluto a planet?", "label": "AMBIGUOUS", "source": "test"},
            {"id": "q2", "question": "is pluto a planet?  ", "label": "KNOWN", "source": "test"},  # Conflicting label
            {"id": "q3", "question": "What is 2+2?", "label": "KNOWN", "source": "test"},
            {"id": "q4", "question": "what is 2+2?", "label": "KNOWN", "source": "test"},  # Consistent duplicate
            {"id": "q5", "question": "Where is my lost key?", "label": "UNKNOWN", "source": "test"},
        ]

        summary = DatasetValidator.validate_records(records)
        assert summary["duplicate_count"] == 2
        assert summary["conflicting_label_count"] == 1
        assert summary["is_valid"] is False

        # Verify conflicting label information is captured without silently selecting a label
        conflict = summary["conflicting_labels"][0]
        assert "pluto a planet" in conflict["normalized_question"]
        assert set(conflict["conflicting_labels"]) == {"AMBIGUOUS", "KNOWN"}
        assert len(conflict["record_instances"]) == 2

    def test_group_split_rejects_conflicting_labels(self):
        """Verifies that group_train_val_test_split refuses to partition records with unresolved conflicting labels."""
        records = [
            {"id": "1", "question": "Who wrote Hamlet?", "label": "KNOWN", "source": "test"},
            {"id": "2", "question": "Who wrote Hamlet?", "label": "AMBIGUOUS", "source": "test"},  # Conflict
            {"id": "3", "question": "What is 2 + 2?", "label": "KNOWN", "source": "test"},
            {"id": "4", "question": "Secret?", "label": "UNKNOWN", "source": "test"},
        ]
        with pytest.raises(ValueError, match="conflicting labels"):
            DatasetValidator.group_train_val_test_split(records)

    def test_global_grouping_prevents_duplicate_partition_split(self):
        """Verifies that all duplicates of a question remain strictly in the same partition."""
        records = [
            {"id": "1", "question": "What is the capital of Georgia?", "label": "AMBIGUOUS", "source": "test"},
            {"id": "2", "question": "What is the capital of Georgia?", "label": "AMBIGUOUS", "source": "test"},
            {"id": "3", "question": "What is the capital of Georgia?", "label": "AMBIGUOUS", "source": "test"},
            {"id": "4", "question": "What is 2 + 2?", "label": "KNOWN", "source": "test"},
            {"id": "5", "question": "What is 3 + 3?", "label": "KNOWN", "source": "test"},
            {"id": "6", "question": "What is 4 + 4?", "label": "KNOWN", "source": "test"},
            {"id": "7", "question": "What will happen in 3000?", "label": "UNKNOWN", "source": "test"},
            {"id": "8", "question": "What will happen in 3001?", "label": "UNKNOWN", "source": "test"},
            {"id": "9", "question": "What will happen in 3002?", "label": "UNKNOWN", "source": "test"},
        ]

        train, val, test = DatasetValidator.group_train_val_test_split(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        partitions = {"train": train, "val": val, "test": test}

        # Check where 'What is the capital of Georgia?' appears
        found_in = []
        for pname, part in partitions.items():
            count = sum(1 for r in part if "capital of georgia" in r["question"].lower())
            if count > 0:
                found_in.append((pname, count))

        # Must appear in exactly one partition with all 3 instances
        assert len(found_in) == 1, f"Question split across multiple partitions: {found_in}"
        assert found_in[0][1] == 3, f"Expected 3 duplicate records in partition, got {found_in[0][1]}"

    def test_class_coverage_reporting(self):
        """Verifies class coverage calculation and honest limitation reporting."""
        records = [
            {"id": "1", "question": "Q1?", "label": "KNOWN", "source": "test"},
            {"id": "2", "question": "Q2?", "label": "KNOWN", "source": "test"},
            {"id": "3", "question": "Q3?", "label": "KNOWN", "source": "test"},
            {"id": "4", "question": "Q4?", "label": "UNKNOWN", "source": "test"},
            {"id": "5", "question": "Q5?", "label": "UNKNOWN", "source": "test"},
            {"id": "6", "question": "Q6?", "label": "UNKNOWN", "source": "test"},
            # Notice: zero AMBIGUOUS records provided
        ]

        _, _, _, meta = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, return_metadata=True
        )

        # Meta must report 0 groups for AMBIGUOUS across all splits and document limitations
        assert meta["train_groups_by_class"]["AMBIGUOUS"] == 0
        assert meta["test_groups_by_class"]["AMBIGUOUS"] == 0
        assert any("AMBIGUOUS" in lim for lim in meta["limitations"])

    def test_conservative_grouping_policy_audit(self):
        """Verifies that the conservative policy distinguishes questions with common 4-word stems."""
        q1 = "What is the capital of France?"
        q2 = "What is the capital of Georgia?"
        q3 = "what is the capital of France?!"

        key1 = DatasetValidator.get_question_group_key(q1)
        key2 = DatasetValidator.get_question_group_key(q2)
        key3 = DatasetValidator.get_question_group_key(q3)

        # Distinct questions must have distinct group keys
        assert key1 != key2, f"Over-grouping detected: {key1} == {key2}"
        # Minor punctuation/casing variations must match
        assert key1 == key3, f"Formatting variation mismatch: {key1} != {key3}"

    def test_schema_variants_label_and_gold_label(self):
        """Verifies normalization function handles both 'label' and 'gold_label' schemas."""
        rec_old = {"id": "1", "question": "What is 2+2?", "label": "KNOWN", "source": "legacy"}
        rec_new = {"id": "2", "question": "What is 3+3?", "gold_label": "KNOWN", "source": "v300"}

        assert DatasetValidator.extract_label(rec_old) == "KNOWN"
        assert DatasetValidator.extract_label(rec_new) == "KNOWN"

        summary = DatasetValidator.validate_records([rec_old, rec_new])
        assert summary["is_valid"] is True
        assert summary["total_records"] == 2
        assert summary["label_distribution"]["KNOWN"] == 2

    def test_schema_variants_group_field_names(self):
        """Verifies support for question_group_id, group_id, and group_key alternatives."""
        r1 = {"id": "1", "question": "Q1?", "label": "KNOWN", "question_group_id": "grp_alpha"}
        r2 = {"id": "2", "question": "Q2?", "label": "KNOWN", "group_id": "grp_beta"}
        r3 = {"id": "3", "question": "Q3?", "label": "KNOWN", "group_key": "grp_gamma"}

        assert DatasetValidator.extract_group_ids(r1) == ["grp_alpha"]
        assert DatasetValidator.extract_group_ids(r2) == ["grp_beta"]
        assert DatasetValidator.extract_group_ids(r3) == ["grp_gamma"]

    def test_conflicting_label_fields_rejected(self):
        """Verifies that records with conflicting 'label' and 'gold_label' are rejected."""
        rec_conflict = {
            "id": "c1",
            "question": "Q?",
            "label": "KNOWN",
            "gold_label": "UNKNOWN",
            "source": "test",
        }
        with pytest.raises(ValueError, match=r"(?i)conflicting label fields"):
            DatasetValidator.extract_label(rec_conflict)

        summary = DatasetValidator.validate_records([rec_conflict])
        assert summary["is_valid"] is False
        assert any("conflicting label fields" in err.lower() for err in summary["schema_errors"])

    def test_missing_label_field_rejected(self):
        """Verifies that records missing both label and gold_label are rejected."""
        rec_missing = {"id": "m1", "question": "Q?", "source": "test"}
        with pytest.raises(ValueError, match=r"(?i)missing required label field"):
            DatasetValidator.extract_label(rec_missing)

        summary = DatasetValidator.validate_records([rec_missing])
        assert summary["is_valid"] is False
        assert any("missing required label field" in err.lower() for err in summary["schema_errors"])

    def test_conflicting_group_fields_rejected(self):
        """Verifies that contradictory group fields on the same record are rejected."""
        rec_bad_group = {
            "id": "bg1",
            "question": "Q?",
            "label": "KNOWN",
            "group_id": "grp_one",
            "question_group_id": "grp_two",
            "source": "test",
        }
        with pytest.raises(ValueError, match=r"(?i)conflicting group fields"):
            DatasetValidator.extract_group_ids(rec_bad_group)

        summary = DatasetValidator.validate_records([rec_bad_group])
        assert summary["is_valid"] is False
        assert any("conflicting group fields" in err.lower() for err in summary["schema_errors"])

    def test_cross_label_adversarial_groups_preserved_together(self):
        """Verifies that groups with different labels (adversarial pairs) are supported and not split across partitions."""
        records = [
            # Group 1: Adversarial vacuum wave speed pair (KNOWN + UNKNOWN in same group)
            {"id": "adv_1", "question": "What is the speed of light in a vacuum?", "label": "KNOWN", "group_id": "grp_adv_vacuum"},
            {"id": "adv_2", "question": "What is the speed of sound in a vacuum?", "label": "UNKNOWN", "group_id": "grp_adv_vacuum"},
            # Group 2: Math facts (KNOWN)
            {"id": "m_1", "question": "What is 2 + 2?", "label": "KNOWN", "group_id": "grp_math"},
            {"id": "m_2", "question": "What is 3 + 3?", "label": "KNOWN", "group_id": "grp_math"},
            # Group 3: Ambiguous queries (AMBIGUOUS)
            {"id": "a_1", "question": "Who is Michael Jordan?", "label": "AMBIGUOUS", "group_id": "grp_jordan"},
            {"id": "a_2", "question": "Tell me about Michael Jordan", "label": "AMBIGUOUS", "group_id": "grp_jordan"},
            # Group 4: Future events (UNKNOWN)
            {"id": "f_1", "question": "Who will win the election in 2060?", "label": "UNKNOWN", "group_id": "grp_fut_elec"},
            {"id": "f_2", "question": "Which candidate will win in 2060?", "label": "UNKNOWN", "group_id": "grp_fut_elec"},
            # Group 5: Country facts (KNOWN)
            {"id": "c_1", "question": "What is the capital of France?", "label": "KNOWN", "group_id": "grp_cap_france"},
            {"id": "c_2", "question": "Which city is the French capital?", "label": "KNOWN", "group_id": "grp_cap_france"},
        ]

        train, val, test, meta = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42, return_metadata=True
        )

        partitions = {"train": train, "val": val, "test": test}
        # Check where adv_1 and adv_2 end up
        adv1_part = [p for p, recs in partitions.items() if any(r["id"] == "adv_1" for r in recs)]
        adv2_part = [p for p, recs in partitions.items() if any(r["id"] == "adv_2" for r in recs)]

        assert len(adv1_part) == 1
        assert len(adv2_part) == 1
        assert adv1_part[0] == adv2_part[0], f"Adversarial pair split across partitions: adv_1 in {adv1_part}, adv_2 in {adv2_part}"
        assert meta["is_disjoint"] is True

    def test_overlapping_composite_groups_merged_into_connected_components(self):
        """Verifies that overlapping composite group membership merges records into a connected component."""
        records = [
            {"id": "r1", "question": "Q1?", "label": "KNOWN", "group_id": ["gA", "gB"]},
            {"id": "r2", "question": "Q2?", "label": "KNOWN", "group_id": ["gB", "gC"]},
            {"id": "r3", "question": "Q3?", "label": "KNOWN", "group_id": ["gC"]},
            {"id": "r4", "question": "Q4?", "label": "AMBIGUOUS", "group_id": "gD"},
            {"id": "r5", "question": "Q5?", "label": "UNKNOWN", "group_id": "gE"},
            {"id": "r6", "question": "Q6?", "label": "UNKNOWN", "group_id": "gF"},
        ]

        train, val, test, meta = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, random_seed=7, return_metadata=True
        )

        partitions = {"train": train, "val": val, "test": test}
        r1_part = [p for p, recs in partitions.items() if any(r["id"] == "r1" for r in recs)][0]
        r2_part = [p for p, recs in partitions.items() if any(r["id"] == "r2" for r in recs)][0]
        r3_part = [p for p, recs in partitions.items() if any(r["id"] == "r3" for r in recs)][0]

        # All three transitively linked records must land in the same partition
        assert r1_part == r2_part == r3_part, f"Connected components split: r1 in {r1_part}, r2 in {r2_part}, r3 in {r3_part}"
        assert meta["is_disjoint"] is True


