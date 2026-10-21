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
        """Verifies that questions with similar stems are not split between train and test."""
        records = [
            {"id": "1", "question": "What is the capital of France?", "label": "KNOWN", "source": "s"},
            {"id": "2", "question": "What is the capital of Spain?", "label": "KNOWN", "source": "s"},
            {"id": "3", "question": "Who is Jordan?", "label": "AMBIGUOUS", "source": "s"},
            {"id": "4", "question": "Who is Jordan (the basketball player)?", "label": "AMBIGUOUS", "source": "s"},
            {"id": "5", "question": "What number am I thinking of?", "label": "UNKNOWN", "source": "s"},
            {"id": "6", "question": "What number am I thinking of right now?", "label": "UNKNOWN", "source": "s"},
            {"id": "7", "question": "What is 2 + 2?", "label": "KNOWN", "source": "s"},
            {"id": "8", "question": "What is 3 + 3?", "label": "KNOWN", "source": "s"},
        ]

        train, val, test = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        train_groups = {DatasetValidator.get_question_group_key(r["question"]) for r in train}
        test_groups = {DatasetValidator.get_question_group_key(r["question"]) for r in test}

        # Stems must not overlap between train and test sets
        overlap = train_groups & test_groups
        assert len(overlap) == 0, f"Data leakage detected! Overlapping groups: {overlap}"
