"""Validator for uncertainty benchmark drafts and candidate expansion datasets."""

import re
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field


TARGET_QUOTAS = {
    "determinate_factual": 50,
    "concept_definition": 50,
    "subjective_questions": 40,
    "genuine_ambiguity": 60,
    "private_inaccessible_info": 35,
    "unknowable_future_events": 35,
    "false_premises": 30,
}

CATEGORY_TO_LABEL = {
    "determinate_factual": "KNOWN",
    "concept_definition": "KNOWN",
    "subjective_questions": "AMBIGUOUS",
    "genuine_ambiguity": "AMBIGUOUS",
    "private_inaccessible_info": "UNKNOWN",
    "unknowable_future_events": "UNKNOWN",
    "false_premises": "UNKNOWN",
}

VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}


@dataclass
class ValidationReport:
    is_valid: bool
    total_records: int
    label_counts: Dict[str, int]
    category_counts: Dict[str, int]
    duplicate_questions: List[Tuple[str, List[str]]]
    missing_evidence_count: int
    missing_evidence_ids: List[str]
    group_count: int
    schema_errors: List[str]
    unverified_candidate_count: int
    adversarial_count: int
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkDraftValidator:
    """Validates schema compliance, category quotas, group integrity, and label balance for benchmark drafts."""

    @staticmethod
    def normalize_question(q: str) -> str:
        """Strip casing, extra whitespace, and punctuation for robust duplicate detection."""
        q_clean = q.lower().strip()
        # Remove punctuation
        q_clean = q_clean.translate(str.maketrans("", "", string.punctuation))
        # Collapse multiple spaces
        return " ".join(q_clean.split())

    @classmethod
    def validate_records(cls, records: List[Dict[str, Any]]) -> ValidationReport:
        schema_errors: List[str] = []
        total_records = len(records)
        label_counter = Counter()
        category_counter = Counter()

        seen_normalized: Dict[str, List[str]] = defaultdict(list)
        missing_evidence_ids: List[str] = []
        group_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        unverified_candidates = 0
        adversarial_count = 0

        required_keys = [
            "id",
            "question",
            "category",
            "intended_interpretation",
            "gold_label",
            "rationale",
            "answerability_evidence",
            "source",
            "question_group_id",
            "reviewer_agreement",
        ]

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"record_{idx}")

            # 1. Schema check
            for rk in required_keys:
                if rk not in rec or rec[rk] is None or rec[rk] == "":
                    schema_errors.append(f"Record {rec_id} missing or empty required field: '{rk}'")

            # 2. Label & Category checks
            label = rec.get("gold_label")
            if label not in VALID_LABELS:
                schema_errors.append(f"Record {rec_id} has invalid gold_label: '{label}'")
            else:
                label_counter[label] += 1

            cat = rec.get("category")
            if cat not in TARGET_QUOTAS:
                schema_errors.append(f"Record {rec_id} has invalid category: '{cat}'")
            else:
                category_counter[cat] += 1

            # 3. Check question uniqueness
            q = rec.get("question", "")
            if q:
                norm_q = cls.normalize_question(q)
                seen_normalized[norm_q].append(rec_id)

            # 4. Group tracking
            gid = rec.get("question_group_id")
            if gid:
                group_members[gid].append(rec)
            else:
                schema_errors.append(f"Record {rec_id} missing question_group_id")

            # 5. Reviewer agreement / Draft provenance check
            rev = rec.get("reviewer_agreement", {})
            status = rev.get("annotation_status")
            if status == "needs_human_review":
                unverified_candidates += 1
            elif status in ["human_verified", "gold_benchmark"]:
                schema_errors.append(
                    f"Record {rec_id} cannot be marked '{status}' without completed human adjudication log."
                )

            # 6. Evidence check: Flag missing / placeholder / unverified citations
            evidence = rec.get("answerability_evidence", {})
            cits = evidence.get("citations_or_proof", [])
            needs_human_verification = (
                not cits
                or any("flagged_for_human_verification" in str(c).lower() or "needs_review" in str(c).lower() for c in cits)
                or rec.get("source") == "provisional_needs_verification"
                or evidence.get("verification_status") == "flagged_for_human_verification"
            )
            if needs_human_verification:
                missing_evidence_ids.append(rec_id)

            if rec.get("is_adversarial_variant") or "adversarial" in rec.get("id", ""):
                adversarial_count += 1

        # Check duplicate questions
        duplicates = [(norm_q, ids) for norm_q, ids in seen_normalized.items() if len(ids) > 1]

        # Quota checks
        quota_mismatches = []
        for cat, target_count in TARGET_QUOTAS.items():
            actual = category_counter.get(cat, 0)
            if actual != target_count:
                quota_mismatches.append(f"Category '{cat}': expected {target_count}, got {actual}")

        # Label balance checks (target 100/100/100 for 300 questions)
        expected_label_balance = {"KNOWN": 100, "AMBIGUOUS": 100, "UNKNOWN": 100}
        balance_mismatches = []
        if total_records == 300:
            for lbl, target_lbl_count in expected_label_balance.items():
                actual_lbl = label_counter.get(lbl, 0)
                if actual_lbl != target_lbl_count:
                    balance_mismatches.append(f"Label '{lbl}': expected {target_lbl_count}, got {actual_lbl}")

        is_valid = (
            len(schema_errors) == 0
            and len(duplicates) == 0
            and len(quota_mismatches) == 0
            and len(balance_mismatches) == 0
            and total_records == 300
        )

        return ValidationReport(
            is_valid=is_valid,
            total_records=total_records,
            label_counts=dict(label_counter),
            category_counts=dict(category_counter),
            duplicate_questions=duplicates,
            missing_evidence_count=len(missing_evidence_ids),
            missing_evidence_ids=missing_evidence_ids,
            group_count=len(group_members),
            schema_errors=schema_errors + quota_mismatches + balance_mismatches,
            unverified_candidate_count=unverified_candidates,
            adversarial_count=adversarial_count,
            details={
                "quota_mismatches": quota_mismatches,
                "balance_mismatches": balance_mismatches,
            },
        )
