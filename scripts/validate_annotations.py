#!/usr/bin/env python3
"""Validation tooling for human annotation records in the LLM Uncertainty Benchmark.

Verifies that completed blinded annotation files conform strictly to the annotation
schema, contain zero missing/null labels, contain no duplicate or missing IDs,
and exhibit no schema violations or data leakage.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple, Set


VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
VALID_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
ALLOWED_FIELDS = {
    "blinded_id",
    "question",
    "assigned_label",
    "confidence",
    "rationale",
    "requires_adjudication",
    "notes",
}
REQUIRED_FIELDS = {
    "blinded_id",
    "question",
    "assigned_label",
    "confidence",
    "rationale",
    "requires_adjudication",
}
FORBIDDEN_LEAKAGE_KEYS = {
    "label",
    "candidate_label",
    "true_label",
    "gold_label",
    "prediction",
    "pred",
    "feature",
    "features",
    "cluster_entropy",
    "entropy",
    "num_clusters",
    "cluster",
    "question_group_id",
    "group_id",
    "category",
    "stratum",
    "id",
    "record_id",
}


class ValidationReport:
    """Stores and formats results from annotation validation."""

    def __init__(self, filename: str):
        self.filename = filename
        self.total_records = 0
        self.valid_records = 0
        self.is_valid = False
        self.annotator_id = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.label_distribution: Dict[str, int] = {k: 0 for k in VALID_LABELS}
        self.confidence_distribution: Dict[str, int] = {k: 0 for k in VALID_CONFIDENCES}
        self.adjudication_flag_count = 0
        self.unannotated_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "annotator_id": self.annotator_id,
            "unannotated_count": self.unannotated_count,
            "adjudication_flag_count": self.adjudication_flag_count,
            "label_distribution": self.label_distribution,
            "confidence_distribution": self.confidence_distribution,
            "error_count": len(self.errors),
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
        }

    def print_summary(self):
        status = "PASSED" if self.is_valid else "FAILED"
        print(f"\n{'=' * 65}")
        print(f" ANNOTATION VALIDATION REPORT: {status}")
        print(f"{'=' * 65}")
        print(f"File:                  {self.filename}")
        print(f"Detected Annotator:    {self.annotator_id or 'UNKNOWN'}")
        print(f"Total Records:         {self.total_records}")
        print(f"Fully Valid Records:   {self.valid_records}")
        print(f"Unannotated Records:   {self.unannotated_count}")
        print(f"Adjudication Flags:    {self.adjudication_flag_count}")
        print(f"\nLabel Distribution:")
        for lbl, count in sorted(self.label_distribution.items()):
            pct = (count / self.total_records * 100) if self.total_records > 0 else 0
            print(f"  - {lbl:<12}: {count:>4} ({pct:5.1f}%)")
        print(f"\nConfidence Distribution:")
        for conf, count in sorted(self.confidence_distribution.items()):
            pct = (count / self.total_records * 100) if self.total_records > 0 else 0
            print(f"  - {conf:<12}: {count:>4} ({pct:5.1f}%)")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:15]:
                print(f"  [WARN] {w}")
            if len(self.warnings) > 15:
                print(f"  ... and {len(self.warnings) - 15} more warnings.")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors[:25]:
                print(f"  [ERROR] {e}")
            if len(self.errors) > 25:
                print(f"  ... and {len(self.errors) - 25} more errors.")
        print(f"{'=' * 65}\n")


class AnnotationValidator:
    """Validates blinded human annotation files against schema and scientific rules."""

    def __init__(self, expected_count: int = 300, allow_unannotated: bool = False):
        self.expected_count = expected_count
        self.allow_unannotated = allow_unannotated

    def validate_file(
        self, filepath: str, expected_annotator: Optional[str] = None
    ) -> ValidationReport:
        report = ValidationReport(filepath)

        if not os.path.exists(filepath):
            report.errors.append(f"File does not exist: {filepath}")
            report.is_valid = False
            return report

        records: List[Dict[str, Any]] = []
        raw_lines: List[str] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except Exception as e:
            report.errors.append(f"Failed to read file: {e}")
            report.is_valid = False
            return report

        report.total_records = len(raw_lines)
        if report.total_records != self.expected_count:
            report.errors.append(
                f"Incorrect record count: found {report.total_records}, expected {self.expected_count}"
            )

        seen_ids: Set[str] = set()
        detected_annotators: Set[str] = set()

        for idx, line in enumerate(raw_lines, start=1):
            line_str = line.strip()
            if not line_str:
                report.errors.append(f"Line {idx}: Empty line in JSONL file")
                continue

            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as jde:
                report.errors.append(f"Line {idx}: Malformed JSON - {jde}")
                continue

            if not isinstance(rec, dict):
                report.errors.append(f"Line {idx}: Record is not a JSON object")
                continue

            line_errors, annotator = self._validate_single_record(
                rec, idx, seen_ids, expected_annotator
            )
            if annotator:
                detected_annotators.add(annotator)

            if line_errors:
                report.errors.extend(line_errors)
            else:
                report.valid_records += 1

            # Stats aggregation
            assigned = rec.get("assigned_label")
            if assigned in VALID_LABELS:
                report.label_distribution[assigned] += 1
            elif assigned is None:
                report.unannotated_count += 1

            conf = rec.get("confidence")
            if conf in VALID_CONFIDENCES:
                report.confidence_distribution[conf] += 1

            if rec.get("requires_adjudication") is True:
                report.adjudication_flag_count += 1

        if len(detected_annotators) == 1:
            report.annotator_id = next(iter(detected_annotators))
        elif len(detected_annotators) > 1:
            report.errors.append(
                f"Mixed annotator IDs in file: found {sorted(detected_annotators)}"
            )

        # Expected ID sequence check (e.g. ANN_A_001 .. ANN_A_300)
        if report.annotator_id and self.expected_count > 0:
            expected_ids = {
                f"ANN_{report.annotator_id}_{i:03d}"
                for i in range(1, self.expected_count + 1)
            }
            missing_ids = expected_ids - seen_ids
            if missing_ids:
                report.errors.append(
                    f"Missing expected IDs ({len(missing_ids)} missing): {sorted(missing_ids)[:10]}"
                )

        if not self.allow_unannotated and report.unannotated_count > 0:
            report.errors.append(
                f"File contains {report.unannotated_count} unannotated records with null assigned_label"
            )

        report.is_valid = len(report.errors) == 0
        return report

    def _validate_single_record(
        self,
        rec: Dict[str, Any],
        line_num: int,
        seen_ids: Set[str],
        expected_annotator: Optional[str],
    ) -> Tuple[List[str], Optional[str]]:
        errors: List[str] = []
        annotator: Optional[str] = None

        # Check required fields present
        missing_fields = REQUIRED_FIELDS - set(rec.keys())
        if missing_fields:
            errors.append(f"Line {line_num}: Missing required fields: {sorted(missing_fields)}")

        # Check unexpected fields (additionalProperties: false)
        extra_fields = set(rec.keys()) - ALLOWED_FIELDS
        if extra_fields:
            errors.append(f"Line {line_num}: Disallowed extra fields: {sorted(extra_fields)}")

        # Check for leakage keys
        leak_keys = {k for k in rec.keys() if k.lower() in FORBIDDEN_LEAKAGE_KEYS}
        if leak_keys:
            errors.append(f"Line {line_num}: CRITICAL LEAKAGE - Prohibited metadata keys found: {sorted(leak_keys)}")

        # Validate blinded_id
        bid = rec.get("blinded_id")
        if not bid or not isinstance(bid, str):
            errors.append(f"Line {line_num}: Missing or non-string 'blinded_id'")
        else:
            match = re.match(r"^ANN_([A-Z])_([0-9]{3})$", bid)
            if not match:
                errors.append(
                    f"Line {line_num}: Invalid 'blinded_id' format '{bid}' (expected ^ANN_[A-Z]_[0-9]{{3}}$)"
                )
            else:
                annotator = match.group(1)
                if expected_annotator and annotator != expected_annotator.upper():
                    errors.append(
                        f"Line {line_num}: Blinded ID '{bid}' does not match expected annotator '{expected_annotator}'"
                    )

            if bid in seen_ids:
                errors.append(f"Line {line_num}: Duplicate blinded_id '{bid}'")
            seen_ids.add(bid)

        # Validate question
        q = rec.get("question")
        if not q or not isinstance(q, str) or len(q.strip()) < 5:
            errors.append(f"Line {line_num}: 'question' must be a non-empty string with >= 5 chars")

        # Validate assigned_label
        label = rec.get("assigned_label")
        if label is None:
            if not self.allow_unannotated:
                errors.append(f"Line {line_num}: 'assigned_label' is null (unannotated)")
        elif label not in VALID_LABELS:
            errors.append(
                f"Line {line_num}: Invalid 'assigned_label' '{label}'. Must be one of {sorted(VALID_LABELS)}"
            )

        # Validate confidence
        conf = rec.get("confidence")
        if conf is None:
            if not self.allow_unannotated and label is not None:
                errors.append(f"Line {line_num}: 'confidence' is null for annotated record")
        elif conf not in VALID_CONFIDENCES:
            errors.append(
                f"Line {line_num}: Invalid 'confidence' '{conf}'. Must be one of {sorted(VALID_CONFIDENCES)}"
            )

        # Validate rationale
        rat = rec.get("rationale")
        if rat is not None and not isinstance(rat, str):
            errors.append(f"Line {line_num}: 'rationale' must be a string")

        # Validate requires_adjudication
        adj = rec.get("requires_adjudication")
        if adj is not None and not isinstance(adj, bool):
            errors.append(f"Line {line_num}: 'requires_adjudication' must be a boolean (true/false)")

        # Validate notes
        notes = rec.get("notes")
        if notes is not None and not isinstance(notes, str):
            errors.append(f"Line {line_num}: 'notes' must be a string")

        return errors, annotator


def main():
    parser = argparse.ArgumentParser(
        description="Validate human annotation JSONL files against schema and operational rules."
    )
    parser.add_argument("filepath", help="Path to the annotation JSONL file to validate.")
    parser.add_argument(
        "--expected-annotator",
        choices=["A", "B", "a", "b"],
        default=None,
        help="Expected annotator identity (A or B).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=300,
        help="Expected total number of records (default: 300).",
    )
    parser.add_argument(
        "--allow-unannotated",
        action="store_true",
        help="Allow null assigned_label for checking uncompleted template sheets.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output report as structured JSON."
    )

    args = parser.parse_args()

    validator = AnnotationValidator(
        expected_count=args.expected_count,
        allow_unannotated=args.allow_unannotated,
    )
    report = validator.validate_file(
        args.filepath, expected_annotator=args.expected_annotator
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
