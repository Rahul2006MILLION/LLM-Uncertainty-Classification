#!/usr/bin/env python3
"""Validation tooling for human annotation records in the LLM Uncertainty Benchmark.

Verifies that blinded human annotation files conform strictly to the annotation
schema, contain zero missing/null labels upon completion, contain no duplicate or missing IDs,
and exhibit no schema violations or metadata/candidate label leakage.
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
    "model_output",
    "logits",
}

DEFAULT_PATHS = {
    "A": "data/annotation/blinded_annotation_annotator_a.jsonl",
    "B": "data/annotation/blinded_annotation_annotator_b.jsonl",
}


class ValidationReport:
    """Stores and formats results from annotation validation."""

    def __init__(self, filename: str, annotator_id: Optional[str] = None, expected_count: int = 300):
        self.filename = filename
        self.annotator_id = annotator_id
        self.expected_count = expected_count
        self.total_records = 0
        self.valid_records = 0
        self.completed_annotations = 0
        self.remaining_annotations = 0
        self.duplicate_id_count = 0
        self.missing_id_count = 0
        self.unexpected_id_count = 0
        self.invalid_label_count = 0
        self.malformed_record_count = 0
        self.leakage_detected = False
        self.is_complete = False
        self.is_valid = False
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.label_distribution: Dict[str, int] = {k: 0 for k in VALID_LABELS}
        self.confidence_distribution: Dict[str, int] = {k: 0 for k in VALID_CONFIDENCES}
        self.adjudication_flag_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "annotator": self.annotator_id,
            "expected_questions": self.expected_count,
            "completed_annotations": self.completed_annotations,
            "remaining_annotations": self.remaining_annotations,
            "duplicate_ids": self.duplicate_id_count,
            "missing_ids": self.missing_id_count,
            "unexpected_ids": self.unexpected_id_count,
            "invalid_labels": self.invalid_label_count,
            "malformed_records": self.malformed_record_count,
            "is_complete": self.is_complete,
            "is_valid": self.is_valid,
            "status": "COMPLETE - ALL CHECKS PASSED" if (self.is_valid and self.is_complete) else "NOT COMPLETE",
            "adjudication_flags": self.adjudication_flag_count,
            "label_distribution": self.label_distribution,
            "confidence_distribution": self.confidence_distribution,
            "error_count": len(self.errors),
            "errors": self.errors,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
        }

    def print_summary(self):
        status_str = "COMPLETE - ALL CHECKS PASSED" if (self.is_valid and self.is_complete) else "NOT COMPLETE"
        print(f"\n{'=' * 65}")
        print(" ANNOTATION VALIDATION REPORT")
        print(f"{'=' * 65}")
        print(f"File:                  {self.filename}")
        print(f"Annotator:             {self.annotator_id or 'UNKNOWN'}")
        print(f"Expected questions:    {self.expected_count}")
        print(f"Completed annotations: {self.completed_annotations}")
        print(f"Remaining annotations: {self.remaining_annotations}")
        print()
        print(f"Duplicate IDs:         {self.duplicate_id_count}")
        print(f"Missing IDs:           {self.missing_id_count}")
        print(f"Unexpected IDs:        {self.unexpected_id_count}")
        print(f"Invalid labels:        {self.invalid_label_count}")
        print(f"Malformed records:     {self.malformed_record_count}")
        print()
        print(f"Status: {status_str}")

        if self.label_distribution and self.completed_annotations > 0:
            print(f"\nLabel Distribution (Completed):")
            for lbl, count in sorted(self.label_distribution.items()):
                pct = (count / self.completed_annotations * 100) if self.completed_annotations > 0 else 0
                print(f"  - {lbl:<12}: {count:>4} ({pct:5.1f}%)")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                print(f"  [WARN] {w}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for e in self.errors[:15]:
                print(f"  [ERROR] {e}")
            if len(self.errors) > 15:
                print(f"  ... and {len(self.errors) - 15} more errors.")
        print(f"{'=' * 65}\n")


class AnnotationValidator:
    """Validates blinded human annotation files against schema and operational rules."""

    def __init__(self, expected_count: int = 300, allow_unannotated: bool = False):
        self.expected_count = expected_count
        self.allow_unannotated = allow_unannotated

    def validate_file(
        self, filepath: str, expected_annotator: Optional[str] = None
    ) -> ValidationReport:
        target_annotator = expected_annotator.upper() if expected_annotator else None
        report = ValidationReport(filepath, annotator_id=target_annotator, expected_count=self.expected_count)

        if not os.path.exists(filepath):
            report.errors.append(f"File does not exist: {filepath}")
            report.is_valid = False
            return report

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
        expected_id_set = (
            {f"ANN_{target_annotator}_{i:03d}" for i in range(1, self.expected_count + 1)}
            if target_annotator else None
        )

        for idx, line in enumerate(raw_lines, start=1):
            line_str = line.strip()
            if not line_str:
                report.malformed_record_count += 1
                report.errors.append(f"Line {idx}: Empty line in JSONL file")
                continue

            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as jde:
                report.malformed_record_count += 1
                report.errors.append(f"Line {idx}: Malformed JSON - {jde}")
                continue

            if not isinstance(rec, dict):
                report.malformed_record_count += 1
                report.errors.append(f"Line {idx}: Record is not a JSON object")
                continue

            line_errors, annotator = self._validate_single_record(
                rec, idx, seen_ids, target_annotator, expected_id_set, report
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
                report.completed_annotations += 1
                report.label_distribution[assigned] += 1
            elif assigned is None:
                report.remaining_annotations += 1

            conf = rec.get("confidence")
            if conf in VALID_CONFIDENCES:
                report.confidence_distribution[conf] += 1

            if rec.get("requires_adjudication") is True:
                report.adjudication_flag_count += 1

        if not report.annotator_id and len(detected_annotators) == 1:
            report.annotator_id = next(iter(detected_annotators))
        elif len(detected_annotators) > 1:
            report.errors.append(
                f"Mixed annotator IDs in file: found {sorted(detected_annotators)}"
            )

        # Sequence completeness check
        if report.annotator_id and self.expected_count > 0:
            full_expected = {
                f"ANN_{report.annotator_id}_{i:03d}"
                for i in range(1, self.expected_count + 1)
            }
            missing = full_expected - seen_ids
            report.missing_id_count = len(missing)
            if missing:
                report.errors.append(
                    f"Missing expected IDs ({len(missing)} missing): {sorted(missing)[:10]}"
                )

        report.is_complete = (report.completed_annotations == self.expected_count and report.remaining_annotations == 0)

        if not self.allow_unannotated and report.remaining_annotations > 0:
            report.errors.append(
                f"File is not complete: {report.remaining_annotations} unannotated records remain."
            )

        report.is_valid = (
            len(report.errors) == 0
            and report.duplicate_id_count == 0
            and report.missing_id_count == 0
            and report.unexpected_id_count == 0
            and report.invalid_label_count == 0
            and report.malformed_record_count == 0
        )
        return report

    def _validate_single_record(
        self,
        rec: Dict[str, Any],
        line_num: int,
        seen_ids: Set[str],
        expected_annotator: Optional[str],
        expected_id_set: Optional[Set[str]],
        report: ValidationReport,
    ) -> Tuple[List[str], Optional[str]]:
        errors: List[str] = []
        annotator: Optional[str] = None

        # Check required fields present
        missing_fields = REQUIRED_FIELDS - set(rec.keys())
        if missing_fields:
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: Missing required fields: {sorted(missing_fields)}")

        # Check unexpected fields (additionalProperties: false)
        extra_fields = set(rec.keys()) - ALLOWED_FIELDS
        if extra_fields:
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: Disallowed extra fields: {sorted(extra_fields)}")

        # Check for leakage keys
        leak_keys = {k for k in rec.keys() if k.lower() in FORBIDDEN_LEAKAGE_KEYS}
        if leak_keys:
            report.leakage_detected = True
            errors.append(f"Line {line_num}: CRITICAL LEAKAGE - Prohibited metadata keys found: {sorted(leak_keys)}")

        # Validate blinded_id
        bid = rec.get("blinded_id")
        if not bid or not isinstance(bid, str):
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: Missing or non-string 'blinded_id'")
        else:
            match = re.match(r"^ANN_([A-Z])_([0-9]{3})$", bid)
            if not match:
                report.unexpected_id_count += 1
                errors.append(
                    f"Line {line_num}: Invalid 'blinded_id' format '{bid}' (expected ^ANN_[A-Z]_[0-9]{{3}}$)"
                )
            else:
                annotator = match.group(1)
                if expected_annotator and annotator != expected_annotator.upper():
                    report.unexpected_id_count += 1
                    errors.append(
                        f"Line {line_num}: Blinded ID '{bid}' does not match expected annotator '{expected_annotator}'"
                    )
                elif expected_id_set and bid not in expected_id_set:
                    report.unexpected_id_count += 1
                    errors.append(f"Line {line_num}: Blinded ID '{bid}' is not in expected ID range")

            if bid in seen_ids:
                report.duplicate_id_count += 1
                errors.append(f"Line {line_num}: Duplicate blinded_id '{bid}'")
            seen_ids.add(bid)

        # Validate question
        q = rec.get("question")
        if not q or not isinstance(q, str) or len(q.strip()) < 5:
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: 'question' must be a non-empty string with >= 5 chars")

        # Validate assigned_label
        label = rec.get("assigned_label")
        if label is None:
            if not self.allow_unannotated:
                errors.append(f"Line {line_num}: 'assigned_label' is null (unannotated)")
        elif label not in VALID_LABELS:
            report.invalid_label_count += 1
            errors.append(
                f"Line {line_num}: Invalid 'assigned_label' '{label}'. Must be one of {sorted(VALID_LABELS)}"
            )

        # Validate confidence
        conf = rec.get("confidence")
        if conf is not None and conf not in VALID_CONFIDENCES:
            report.malformed_record_count += 1
            errors.append(
                f"Line {line_num}: Invalid 'confidence' '{conf}'. Must be one of {sorted(VALID_CONFIDENCES)}"
            )

        # Validate rationale
        rat = rec.get("rationale")
        if rat is not None and not isinstance(rat, str):
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: 'rationale' must be a string")

        # Validate requires_adjudication
        adj = rec.get("requires_adjudication")
        if adj is not None and not isinstance(adj, bool):
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: 'requires_adjudication' must be a boolean (true/false)")

        # Validate notes
        notes = rec.get("notes")
        if notes is not None and not isinstance(notes, str):
            report.malformed_record_count += 1
            errors.append(f"Line {line_num}: 'notes' must be a string")

        return errors, annotator


def main():
    parser = argparse.ArgumentParser(
        description="Validate human annotation JSONL files against schema and operational rules."
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help="Path to the annotation JSONL file to validate.",
    )
    parser.add_argument(
        "--annotator",
        choices=["A", "B", "a", "b"],
        default=None,
        help="Specify annotator (A or B) to validate default file location.",
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

    # Determine filepath and expected annotator
    filepath = args.filepath
    expected_annotator = args.annotator.upper() if args.annotator else None

    if not filepath and expected_annotator:
        filepath = DEFAULT_PATHS.get(expected_annotator)

    if not filepath:
        parser.error("Either a filepath or --annotator [A|B] must be specified.")

    validator = AnnotationValidator(
        expected_count=args.expected_count,
        allow_unannotated=args.allow_unannotated,
    )
    report = validator.validate_file(
        filepath, expected_annotator=expected_annotator
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    sys.exit(0 if (report.is_valid and (report.is_complete or args.allow_unannotated)) else 1)


if __name__ == "__main__":
    main()
