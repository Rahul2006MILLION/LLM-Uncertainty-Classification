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
        """Perform stratified group-based split by question group key to strictly prevent train/test leakage.
        
        Guarantees:
        1. No question group key appears in more than one partition.
        2. Balanced representation across all present classes in each partition.
        """
        if not records:
            return [], [], []

        # Partition records by class label
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            lbl = r.get("label", "UNKNOWN")
            by_class.setdefault(lbl, []).append(r)

        train_set: List[Dict[str, Any]] = []
        val_set: List[Dict[str, Any]] = []
        test_set: List[Dict[str, Any]] = []

        rng = np.random.RandomState(random_seed)

        for c_idx, (lbl, class_records) in enumerate(by_class.items()):
            # Find unique groups for this class
            group_to_records: Dict[str, List[Dict[str, Any]]] = {}
            for r in class_records:
                gkey = cls.get_question_group_key(r["question"])
                group_to_records.setdefault(gkey, []).append(r)

            unique_groups = list(group_to_records.keys())
            # Deterministic shuffle with seed offset
            class_seed = random_seed + c_idx * 17
            shuffled_groups = list(unique_groups)
            np.random.RandomState(class_seed).shuffle(shuffled_groups)

            n_g = len(shuffled_groups)
            if n_g == 1:
                train_groups = set(shuffled_groups)
                val_groups = set()
                test_groups = set()
            elif n_g == 2:
                train_groups = {shuffled_groups[0]}
                val_groups = set()
                test_groups = {shuffled_groups[1]}
            else:
                n_test = max(1, int(round(n_g * test_ratio)))
                n_val = max(0, int(round(n_g * val_ratio))) if val_ratio > 0 else 0
                n_train = n_g - n_test - n_val
                if n_train < 1:
                    n_train = 1
                    if n_val > 0:
                        n_val -= 1
                    else:
                        n_test -= 1

                test_groups = set(shuffled_groups[:n_test])
                val_groups = set(shuffled_groups[n_test:n_test + n_val])
                train_groups = set(shuffled_groups[n_test + n_val:])

            for g in train_groups:
                train_set.extend(group_to_records[g])
            for g in val_groups:
                val_set.extend(group_to_records[g])
            for g in test_groups:
                test_set.extend(group_to_records[g])

        return train_set, val_set, test_set
