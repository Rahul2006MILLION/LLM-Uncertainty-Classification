"""Reusable dataset loader and rigorous schema/provenance validator."""

import os
import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.experimentation.feature_groups import FEATURE_NAMES

VALID_LABELS: Set[str] = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
LABEL_TO_ID: Dict[str, int] = {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 2}
ID_TO_LABEL: Dict[int, str] = {0: "KNOWN", 1: "AMBIGUOUS", 2: "UNKNOWN"}


@dataclass
class DatasetValidationReport:
    """Detailed validation report for an experimental dataset."""
    is_valid: bool
    total_samples: int
    class_distribution: Dict[str, int]
    feature_names: List[str]
    missing_features_count: int
    non_numeric_values_count: int
    duplicate_ids: List[str]
    duplicate_questions: List[str]
    invalid_labels: List[str]
    errors: List[str]
    warnings: List[str]
    file_path: str
    file_sha256: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_samples": self.total_samples,
            "class_distribution": self.class_distribution,
            "duplicate_id_count": len(self.duplicate_ids),
            "duplicate_question_count": len(self.duplicate_questions),
            "invalid_label_count": len(self.invalid_labels),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "file_sha256": self.file_sha256,
        }


class ExperimentDatasetLoader:
    """Loads and validates JSONL or JSON feature datasets for uncertainty classification."""

    def __init__(self, required_features: Optional[List[str]] = None):
        self.required_features = list(required_features or FEATURE_NAMES)

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize question text by lowercasing, stripping punctuation, and collapsing whitespace."""
        if not question:
            return ""
        clean = re.sub(r"[^\w\s]", "", str(question).lower()).strip()
        return re.sub(r"\s+", " ", clean)

    def load_and_validate(
        self,
        filepath: str,
        allow_warnings: bool = True,
    ) -> Tuple[List[Dict[str, Any]], DatasetValidationReport]:
        """Load records from JSONL file and validate against the schema.
        
        Strictly does NOT alter or repair data. Raises ValueError if critical errors exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file does not exist: {filepath}")

        sha256 = self.compute_sha256(filepath)
        errors: List[str] = []
        warnings: List[str] = []
        records: List[Dict[str, Any]] = []

        seen_ids: Set[str] = set()
        duplicate_ids: List[str] = []
        seen_normalized_questions: Dict[str, str] = {}
        duplicate_questions: List[str] = []
        invalid_labels: List[str] = []
        class_counts: Dict[str, int] = {lbl: 0 for lbl in VALID_LABELS}

        missing_feature_count = 0
        non_numeric_feature_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_no}: Malformed JSON - {e}")
                    continue

                if not isinstance(rec, dict):
                    errors.append(f"Line {line_no}: Record is not a dictionary object.")
                    continue

                rec_id = rec.get("id")
                if not rec_id or not isinstance(rec_id, str):
                    errors.append(f"Line {line_no}: Missing or non-string 'id'.")
                elif rec_id in seen_ids:
                    duplicate_ids.append(rec_id)
                    errors.append(f"Line {line_no}: Duplicate question id '{rec_id}'.")
                else:
                    seen_ids.add(rec_id)

                # Question text
                q_text = rec.get("question", "")
                if not q_text or not isinstance(q_text, str) or len(q_text.strip()) < 3:
                    errors.append(f"Line {line_no} (ID: {rec_id}): Missing or invalid 'question' text.")
                else:
                    norm_q = self.normalize_question(q_text)
                    if norm_q in seen_normalized_questions:
                        duplicate_questions.append(rec_id)
                        warnings.append(
                            f"Line {line_no} (ID: {rec_id}): Normalized question text identical to '{seen_normalized_questions[norm_q]}'."
                        )
                    else:
                        seen_normalized_questions[norm_q] = str(rec_id)

                # Label validation
                label = rec.get("label") or rec.get("target_label") or rec.get("gold_label")
                if not label:
                    errors.append(f"Line {line_no} (ID: {rec_id}): Missing label field.")
                elif str(label).strip() not in VALID_LABELS:
                    inv_lbl = str(label).strip()
                    invalid_labels.append(inv_lbl)
                    errors.append(f"Line {line_no} (ID: {rec_id}): Invalid label '{inv_lbl}'. Must be one of {VALID_LABELS}.")
                else:
                    canonical_label = str(label).strip()
                    class_counts[canonical_label] += 1
                    rec["canonical_label"] = canonical_label

                # Features validation
                features = rec.get("features")
                if features is None or not isinstance(features, dict):
                    errors.append(f"Line {line_no} (ID: {rec_id}): Missing or non-dictionary 'features' object.")
                else:
                    for fname in self.required_features:
                        if fname not in features:
                            missing_feature_count += 1
                            errors.append(f"Line {line_no} (ID: {rec_id}): Missing required feature '{fname}'.")
                        else:
                            val = features[fname]
                            if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                                non_numeric_feature_count += 1
                                errors.append(
                                    f"Line {line_no} (ID: {rec_id}): Non-numeric, NaN, or Inf feature value '{fname}={val}'."
                                )

                records.append(rec)

        if not records and not errors:
            errors.append("Dataset file is empty.")

        is_valid = (len(errors) == 0)

        report = DatasetValidationReport(
            is_valid=is_valid,
            total_samples=len(records),
            class_distribution=class_counts,
            feature_names=list(self.required_features),
            missing_features_count=missing_feature_count,
            non_numeric_values_count=non_numeric_feature_count,
            duplicate_ids=duplicate_ids,
            duplicate_questions=duplicate_questions,
            invalid_labels=invalid_labels,
            errors=errors,
            warnings=warnings,
            file_path=filepath,
            file_sha256=sha256,
        )

        if not is_valid and not allow_warnings:
            raise ValueError(f"Dataset validation failed with {len(errors)} errors:\n" + "\n".join(errors[:10]))

        return records, report

    def extract_matrices(
        self,
        records: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Extract numeric feature matrix X, integer labels y, record IDs, and group IDs.
        
        Returns:
            X: np.ndarray of shape (N, D), dtype float32
            y: np.ndarray of shape (N,), dtype int64 (0: KNOWN, 1: AMBIGUOUS, 2: UNKNOWN)
            ids: List[str] of sample IDs
            group_keys: List[str] for group-aware splitting
        """
        feats_to_use = list(feature_names or self.required_features)
        X_rows = []
        y_rows = []
        ids = []
        group_keys = []

        for r in records:
            lbl = r.get("canonical_label") or r.get("label") or r.get("target_label")
            if lbl not in LABEL_TO_ID:
                continue
            feats = r.get("features", {})
            row = [float(feats.get(fn, 0.0)) for fn in feats_to_use]
            X_rows.append(row)
            y_rows.append(LABEL_TO_ID[lbl])
            ids.append(str(r.get("id", "")))
            
            # Determine group key (explicit question_group_id or fallback to normalized question text)
            grp = r.get("question_group_id") or r.get("group_id")
            if not grp:
                grp = self.normalize_question(r.get("question", ""))
            group_keys.append(str(grp))

        X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, len(feats_to_use)), dtype=np.float32)
        y = np.array(y_rows, dtype=np.int64) if y_rows else np.empty((0,), dtype=np.int64)
        return X, y, ids, group_keys
