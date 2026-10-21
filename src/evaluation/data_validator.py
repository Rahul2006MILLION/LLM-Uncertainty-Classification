"""Dataset validation, leakage prevention, and group-based splitting."""

import re
import hashlib
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


class DatasetValidator:
    """Validates question datasets, checks class coverage, and prevents leakage."""

    REQUIRED_FIELDS = ["id", "question", "label", "source"]
    VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}

    @classmethod
    def validate_records(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate an array of question records against schema rules.
        
        Returns validation summary with error counts, class distributions, and duplicate checks.
        """
        errors = []
        class_counts = {lbl: 0 for lbl in cls.VALID_LABELS}
        seen_questions = set()
        duplicates = []

        for idx, rec in enumerate(records):
            for req in cls.REQUIRED_FIELDS:
                if req not in rec or rec[req] is None or str(rec[req]).strip() == "":
                    errors.append(f"Record {idx} missing required field '{req}'")

            label = rec.get("label")
            if label not in cls.VALID_LABELS:
                errors.append(f"Record {idx} has invalid label '{label}'")
            else:
                class_counts[label] += 1

            # Duplicate question text check (normalized)
            q_norm = re.sub(r"\s+", " ", rec.get("question", "").lower().strip())
            if q_norm in seen_questions:
                duplicates.append(rec.get("question"))
            else:
                seen_questions.add(q_norm)

        # Check class coverage
        missing_classes = [c for c, count in class_counts.items() if count == 0]
        if missing_classes:
            errors.append(f"Dataset lacks representation for classes: {missing_classes}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "total_records": len(records),
            "errors": errors,
            "class_counts": class_counts,
            "unique_questions": len(seen_questions),
            "duplicate_count": len(duplicates),
        }

    @staticmethod
    def get_question_group_key(question: str) -> str:
        """Derive a canonical group key for a question to group paraphrases/variants."""
        # Simple stemmed signature to prevent train/test leakage of similar question stems
        clean = re.sub(r"[^\w\s]", "", question.lower()).strip()
        words = clean.split()[:4] # Group by first 4 substantive words
        return "_".join(words) if words else clean

    @classmethod
    def group_train_val_test_split(
        cls,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform group-based split by question group key to strictly prevent train/test leakage."""
        if not records:
            return [], [], []

        groups = [cls.get_question_group_key(r["question"]) for r in records]

        # First split into train and remainder
        gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
        train_idx, rem_idx = next(gss_train.split(records, groups=groups))

        rem_records = [records[i] for i in rem_idx]
        rem_groups = [groups[i] for i in rem_idx]

        # Split remainder into val and test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=random_seed)
        val_sub_idx, test_sub_idx = next(gss_val.split(rem_records, groups=rem_groups))

        train_set = [records[i] for i in train_idx]
        val_set = [rem_records[i] for i in val_sub_idx]
        test_set = [rem_records[i] for i in test_sub_idx]

        return train_set, val_set, test_set
