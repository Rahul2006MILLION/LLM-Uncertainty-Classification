#!/usr/bin/env python3
"""Gold Benchmark Split Generation Pipeline & Leakage-Safe Partitioning Engine.

Milestone 7.5.4: Gold Benchmark Split Generation Infrastructure
Research Project: LLM Uncertainty Classification Using Multi-Response Consistency
                  and Feature-Based Machine Learning

Implements:
1. Strict Gold Benchmark Provenance Validation (Rejects candidate or synthetic unannotated data)
2. Group-Stratified 70/15/15 Train-Validation-Test Splitting
3. StratifiedGroupKFold 5-Fold Cross-Validation
4. Independent Leakage Detection (Train/Val/Test and Fold-level group isolation)
5. Reproducibility using fixed random seeds and deterministic tie-breaking
6. Machine-readable JSON and human-readable Markdown Audit Reports
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
CANONICAL_CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_CV_FOLDS = 5
DEFAULT_SEED = 42

FORBIDDEN_LEAKAGE_KEYS = {
    "prediction", "pred", "cluster_entropy", "entropy", "num_clusters",
    "feature", "features", "logits", "model_output"
}


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def normalize_question_text(question: str) -> str:
    """Normalizes question string for text-collision leakage checks."""
    if not question:
        return ""
    clean = re.sub(r"[^\w\s]", "", str(question).lower()).strip()
    return re.sub(r"\s+", " ", clean)


# =====================================================================
# 1. Gold Benchmark Validator
# =====================================================================

class GoldDatasetValidator:
    """Validates that a dataset is a finalized, human-annotated gold benchmark."""

    def __init__(self, expected_classes: Optional[Set[str]] = None):
        self.expected_classes = expected_classes or VALID_LABELS

    def validate_file(
        self,
        filepath: str,
        manifest_path: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """Validates that a file contains an authoritative gold benchmark.

        Returns: (is_valid, records, errors, metadata)
        """
        errors = []
        records = []
        metadata: Dict[str, Any] = {}

        if not os.path.exists(filepath):
            errors.append(f"File does not exist: {filepath}")
            return False, records, errors, metadata

        # Optional manifest check
        if manifest_path:
            if not os.path.exists(manifest_path):
                errors.append(f"Manifest file does not exist: {manifest_path}")
            else:
                try:
                    with open(manifest_path, "r", encoding="utf-8") as mf:
                        manifest_data = json.load(mf)
                    expected_sha = manifest_data.get("output_artifact", {}).get("gold_benchmark", {}).get("sha256")
                    actual_sha = compute_sha256(filepath)
                    if expected_sha and actual_sha != expected_sha:
                        errors.append(
                            f"Integrity check failed: manifest SHA-256 {expected_sha} "
                            f"does not match actual {actual_sha}"
                        )
                except Exception as ex:
                    errors.append(f"Failed to read/validate manifest: {ex}")

        # Parse JSON/JSONL
        raw_lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    if not isinstance(rec, dict):
                        errors.append(f"Line {line_no}: Record is not a JSON object")
                        continue
                    raw_lines.append((line_no, rec))
                except json.JSONDecodeError as jde:
                    errors.append(f"Line {line_no}: Malformed JSON - {jde}")

        if not raw_lines and not errors:
            errors.append("Dataset is completely empty.")
            return False, records, errors, metadata

        # If syntax errors encountered, return early
        if errors and not raw_lines:
            return False, records, errors, metadata

        records = [r for _, r in raw_lines]

        # Explicit Candidate Rejection: filename check heuristic
        base_name = os.path.basename(filepath).lower()
        if "candidate" in base_name:
            errors.append(
                f"Candidate dataset rejected: filename '{base_name}' indicates candidate/draft data."
            )

        # Record-level verification
        seen_ids: Set[str] = set()
        for line_no, rec in raw_lines:
            rec_id = rec.get("id")
            if not rec_id or not isinstance(rec_id, str):
                errors.append(f"Line {line_no}: Missing or invalid 'id'")
            elif rec_id in seen_ids:
                errors.append(f"Line {line_no}: Duplicate question id '{rec_id}'")
            else:
                seen_ids.add(rec_id)

            # Reviewer agreement / gold status
            rev_agree = rec.get("reviewer_agreement")
            if not rev_agree or not isinstance(rev_agree, dict):
                errors.append(f"Line {line_no} (ID: {rec_id}): Missing 'reviewer_agreement' metadata object")
            else:
                ann_status = rev_agree.get("annotation_status")
                if ann_status == "needs_human_review":
                    errors.append(
                        f"Line {line_no} (ID: {rec_id}): Dataset is unannotated candidate data "
                        f"(annotation_status: 'needs_human_review')"
                    )
                elif ann_status != "gold_benchmark":
                    errors.append(
                        f"Line {line_no} (ID: {rec_id}): Invalid annotation_status '{ann_status}', "
                        f"must be 'gold_benchmark'"
                    )

            # Human annotation provenance
            provenance = rec.get("human_annotation_provenance")
            if not provenance or not isinstance(provenance, dict):
                errors.append(
                    f"Line {line_no} (ID: {rec_id}): Missing 'human_annotation_provenance' "
                    f"- dataset lacks required human ground-truth provenance"
                )

            # Gold label verification
            if "final_gold_label" not in rec or rec["final_gold_label"] is None:
                errors.append(
                    f"Line {line_no} (ID: {rec_id}): Missing authoritative 'final_gold_label'"
                )
            else:
                lbl = str(rec["final_gold_label"]).strip()
                if lbl not in self.expected_classes:
                    errors.append(
                        f"Line {line_no} (ID: {rec_id}): Invalid final_gold_label '{lbl}'. "
                        f"Expected one of {sorted(list(self.expected_classes))}"
                    )

            # Question text
            q_text = rec.get("question")
            if not q_text or not isinstance(q_text, str) or len(q_text.strip()) < 3:
                errors.append(f"Line {line_no} (ID: {rec_id}): Missing or invalid 'question' text")

            # Leakage / forbidden key checks
            forbidden_found = set(rec.keys()).intersection(FORBIDDEN_LEAKAGE_KEYS)
            if forbidden_found:
                errors.append(
                    f"Line {line_no} (ID: {rec_id}): Prohibited model feature or output keys detected: {forbidden_found}"
                )

        # Overall metadata collection
        metadata = {
            "file_path": filepath,
            "sha256": compute_sha256(filepath) if os.path.exists(filepath) else None,
            "record_count": len(records),
            "unique_ids": len(seen_ids),
            "benchmark_version": records[0].get("benchmark_version") if records else None,
            "unique_groups": len(set(r.get("question_group_id") for r in records if r.get("question_group_id"))),
            "label_distribution": dict(Counter(r.get("final_gold_label") for r in records if r.get("final_gold_label"))),
        }

        is_valid = (len(errors) == 0)
        return is_valid, records, errors, metadata


# =====================================================================
# 2. Group-Stratified Train / Validation / Test Splitting (70 / 15 / 15)
# =====================================================================

class GroupStratifiedSplitter:
    """Generates group-disjoint, class-stratified Train / Val / Test partitions."""

    def __init__(
        self,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SEED,
        group_key: str = "question_group_id",
        label_key: str = "final_gold_label",
    ):
        assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5), "Ratios must sum to 1.0"
        self.ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
        self.seed = seed
        self.group_key = group_key
        self.label_key = label_key

    def split(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Partitions records into Train / Val / Test preserving group atomicity and class balance."""
        if not records:
            raise ValueError("Cannot split an empty dataset.")

        # Group records by group_key
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        missing_groups = 0
        for idx, r in enumerate(records):
            gid = r.get(self.group_key)
            if not gid:
                # Fallback group if missing (reported in audit)
                gid = f"singleton_missing_group_{idx}"
                missing_groups += 1
            groups[str(gid)].append(r)

        classes = sorted(list(set(r[self.label_key] for r in records if self.label_key in r)))
        if not classes:
            classes = CANONICAL_CLASSES

        n_total = len(records)
        class_totals = Counter(r[self.label_key] for r in records)

        target_totals = {p: self.ratios[p] * n_total for p in self.ratios}
        target_classes = {
            p: {c: self.ratios[p] * class_totals[c] for c in classes}
            for p in self.ratios
        }

        # Deterministic ordering of groups
        rng = np.random.default_rng(self.seed)
        group_items = list(groups.items())
        rng.shuffle(group_items)
        # Stable sort: larger groups first; within same size, preserve seeded shuffle order
        group_items.sort(key=lambda item: len(item[1]), reverse=True)

        partitions = ["train", "val", "test"]
        allocations: Dict[str, List[str]] = {p: [] for p in partitions}
        current_totals = {p: 0 for p in partitions}
        current_classes = {p: {c: 0 for c in classes} for p in partitions}

        # Greedy multi-objective allocation
        for gid, grecs in group_items:
            g_size = len(grecs)
            g_counts = Counter(r[self.label_key] for r in grecs)

            best_p = None
            best_loss = float("inf")

            for p in partitions:
                loss = 0.0
                for cand_p in partitions:
                    cand_tot = current_totals[cand_p] + (g_size if cand_p == p else 0)
                    tgt_tot = target_totals[cand_p]
                    if tgt_tot > 0:
                        loss += ((cand_tot - tgt_tot) / tgt_tot) ** 2

                    for c in classes:
                        cand_cnt = current_classes[cand_p][c] + (g_counts[c] if cand_p == p else 0)
                        tc = target_classes[cand_p][c]
                        if tc > 0:
                            loss += 2.0 * ((cand_cnt - tc) / tc) ** 2

                if loss < best_loss:
                    best_loss = loss
                    best_p = p

            allocations[best_p].append(gid)
            current_totals[best_p] += g_size
            for c in classes:
                current_classes[best_p][c] += g_counts[c]

        # Build output partition mapping
        splits_result: Dict[str, Any] = {}
        group_assignments: Dict[str, str] = {}
        for p in partitions:
            p_record_ids = []
            for gid in allocations[p]:
                group_assignments[gid] = p
                for r in groups[gid]:
                    p_record_ids.append(r["id"])

            splits_result[p] = {
                "count": current_totals[p],
                "percentage": round((current_totals[p] / n_total) * 100.0, 2),
                "target_percentage": round(self.ratios[p] * 100.0, 2),
                "group_count": len(allocations[p]),
                "ids": sorted(p_record_ids),
                "class_distribution": dict(current_classes[p]),
            }

        warnings = []
        if missing_groups > 0:
            warnings.append(
                f"Grouping limitation: {missing_groups} records lacked explicit '{self.group_key}' "
                f"and were treated as isolated singletons."
            )

        # Check for empty classes in partitions
        for p in partitions:
            for c in classes:
                if class_totals[c] > 0 and current_classes[p][c] == 0:
                    warnings.append(
                        f"Partition '{p}' contains zero samples for class '{c}' "
                        f"due to group constraints or small sample size."
                    )

        return {
            "random_seed": self.seed,
            "split_ratios": self.ratios,
            "total_samples": n_total,
            "total_groups": len(groups),
            "splits": splits_result,
            "group_assignments": group_assignments,
            "warnings": warnings,
        }


# =====================================================================
# 3. StratifiedGroupKFold 5-Fold Cross-Validation
# =====================================================================

class FiveFoldCrossValidator:
    """Generates 5-fold cross-validation assignments with group isolation."""

    def __init__(
        self,
        n_splits: int = DEFAULT_CV_FOLDS,
        seed: int = DEFAULT_SEED,
        group_key: str = "question_group_id",
        label_key: str = "final_gold_label",
    ):
        self.n_splits = n_splits
        self.seed = seed
        self.group_key = group_key
        self.label_key = label_key

    def split(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates 5 folds assigning each sample to exactly one validation fold."""
        if not records:
            raise ValueError("Cannot perform cross-validation on an empty dataset.")

        classes = sorted(list(set(r[self.label_key] for r in records if self.label_key in r)))
        if not classes:
            classes = CANONICAL_CLASSES

        # Build arrays for StratifiedGroupKFold
        X = np.arange(len(records))
        y = np.array([r.get(self.label_key, "UNKNOWN") for r in records])
        groups = np.array([r.get(self.group_key, f"singleton_{i}") for i, r in enumerate(records)])

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        folds_list = []
        sample_to_fold: Dict[str, int] = {}
        warnings = []

        for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(X, y, groups)):
            train_recs = [records[i] for i in train_indices]
            val_recs = [records[i] for i in val_indices]

            train_ids = sorted([r["id"] for r in train_recs])
            val_ids = sorted([r["id"] for r in val_recs])

            train_groups = set(r.get(self.group_key, r.get("id")) for r in train_recs)
            val_groups = set(r.get(self.group_key, r.get("id")) for r in val_recs)

            train_classes = dict(Counter(r.get(self.label_key) for r in train_recs))
            val_classes = dict(Counter(r.get(self.label_key) for r in val_recs))

            for vid in val_ids:
                sample_to_fold[vid] = fold_idx

            for c in classes:
                if val_classes.get(c, 0) == 0:
                    warnings.append(
                        f"Fold {fold_idx}: Validation fold contains 0 samples for class '{c}' "
                        f"due to group constraints."
                    )

            folds_list.append({
                "fold_index": fold_idx,
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "train_group_count": len(train_groups),
                "val_group_count": len(val_groups),
                "train_class_distribution": train_classes,
                "val_class_distribution": val_classes,
                "train_ids": train_ids,
                "val_ids": val_ids,
            })

        # Validate that every sample has exactly one validation assignment
        all_ids = set(r["id"] for r in records)
        assigned_ids = set(sample_to_fold.keys())
        if all_ids != assigned_ids:
            unassigned = all_ids - assigned_ids
            warnings.append(f"Completeness failure: {len(unassigned)} samples were not assigned to any validation fold.")

        return {
            "random_seed": self.seed,
            "n_splits": self.n_splits,
            "total_samples": len(records),
            "folds": folds_list,
            "sample_to_fold_assignment": sample_to_fold,
            "warnings": warnings,
        }


# =====================================================================
# 4. Independent Leakage Validator
# =====================================================================

class LeakageValidator:
    """Independently verifies absence of group, sample, and text leakage."""

    def __init__(self, group_key: str = "question_group_id"):
        self.group_key = group_key

    def validate_splits_and_cv(
        self,
        records: List[Dict[str, Any]],
        split_data: Dict[str, Any],
        cv_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs rigorous, independent leakage audit on train/val/test splits and CV folds."""
        id_to_rec = {r["id"]: r for r in records}
        violations = []
        checks = {}

        # 1. Train / Val / Test Partition Leakage
        splits = split_data.get("splits", {})
        train_ids = set(splits.get("train", {}).get("ids", []))
        val_ids = set(splits.get("val", {}).get("ids", []))
        test_ids = set(splits.get("test", {}).get("ids", []))

        # ID disjointness
        tv_id_overlap = train_ids.intersection(val_ids)
        tt_id_overlap = train_ids.intersection(test_ids)
        vt_id_overlap = val_ids.intersection(test_ids)

        checks["train_val_id_disjoint"] = len(tv_id_overlap) == 0
        checks["train_test_id_disjoint"] = len(tt_id_overlap) == 0
        checks["val_test_id_disjoint"] = len(vt_id_overlap) == 0

        if tv_id_overlap:
            violations.append(f"Sample ID overlap between Train and Val: {tv_id_overlap}")
        if tt_id_overlap:
            violations.append(f"Sample ID overlap between Train and Test: {tt_id_overlap}")
        if vt_id_overlap:
            violations.append(f"Sample ID overlap between Val and Test: {vt_id_overlap}")

        # Group disjointness
        train_groups = set(id_to_rec[i].get(self.group_key) for i in train_ids if i in id_to_rec)
        val_groups = set(id_to_rec[i].get(self.group_key) for i in val_ids if i in id_to_rec)
        test_groups = set(id_to_rec[i].get(self.group_key) for i in test_ids if i in id_to_rec)

        tv_grp_overlap = train_groups.intersection(val_groups) - {None}
        tt_grp_overlap = train_groups.intersection(test_groups) - {None}
        vt_grp_overlap = val_groups.intersection(test_groups) - {None}

        checks["train_val_group_disjoint"] = len(tv_grp_overlap) == 0
        checks["train_test_group_disjoint"] = len(tt_grp_overlap) == 0
        checks["val_test_group_disjoint"] = len(vt_grp_overlap) == 0

        if tv_grp_overlap:
            violations.append(f"Group leakage between Train and Val: {tv_grp_overlap}")
        if tt_grp_overlap:
            violations.append(f"Group leakage between Train and Test: {tt_grp_overlap}")
        if vt_grp_overlap:
            violations.append(f"Group leakage between Val and Test: {vt_grp_overlap}")

        # Normalized question collision check
        train_texts = {normalize_question_text(id_to_rec[i].get("question", "")) for i in train_ids if i in id_to_rec}
        val_texts = {normalize_question_text(id_to_rec[i].get("question", "")) for i in val_ids if i in id_to_rec}
        test_texts = {normalize_question_text(id_to_rec[i].get("question", "")) for i in test_ids if i in id_to_rec}

        train_texts.discard("")
        val_texts.discard("")
        test_texts.discard("")

        tv_text_overlap = train_texts.intersection(val_texts)
        tt_text_overlap = train_texts.intersection(test_texts)
        vt_text_overlap = val_texts.intersection(test_texts)

        checks["train_val_text_disjoint"] = len(tv_text_overlap) == 0
        checks["train_test_text_disjoint"] = len(tt_text_overlap) == 0
        checks["val_test_text_disjoint"] = len(vt_text_overlap) == 0

        if tv_text_overlap:
            violations.append(f"Identical question text appears in Train and Val: {tv_text_overlap}")
        if tt_text_overlap:
            violations.append(f"Identical question text appears in Train and Test: {tt_text_overlap}")
        if vt_text_overlap:
            violations.append(f"Identical question text appears in Val and Test: {vt_text_overlap}")

        # 2. Cross-Validation Leakage Audit
        if cv_data:
            folds = cv_data.get("folds", [])
            checks["cv_fold_count"] = len(folds) == DEFAULT_CV_FOLDS

            val_occurrences: Counter = Counter()
            all_cv_ids = set()

            for fold in folds:
                f_idx = fold.get("fold_index")
                f_train_ids = set(fold.get("train_ids", []))
                f_val_ids = set(fold.get("val_ids", []))

                # ID overlap within fold
                f_id_overlap = f_train_ids.intersection(f_val_ids)
                if f_id_overlap:
                    violations.append(f"Fold {f_idx}: ID overlap between train and val: {f_id_overlap}")

                # Group overlap within fold
                f_train_groups = set(id_to_rec[i].get(self.group_key) for i in f_train_ids if i in id_to_rec)
                f_val_groups = set(id_to_rec[i].get(self.group_key) for i in f_val_ids if i in id_to_rec)
                f_grp_overlap = f_train_groups.intersection(f_val_groups) - {None}

                if f_grp_overlap:
                    violations.append(f"Fold {f_idx}: Group leakage between train and val: {f_grp_overlap}")

                # Accumulate validation occurrences
                for vid in f_val_ids:
                    val_occurrences[vid] += 1
                all_cv_ids.update(f_train_ids)
                all_cv_ids.update(f_val_ids)

            # Completeness check: each sample must appear in validation exactly once
            multiple_val = [vid for vid, count in val_occurrences.items() if count > 1]
            zero_val = [r["id"] for r in records if val_occurrences[r["id"]] == 0]

            checks["cv_exact_single_val_assignment"] = (len(multiple_val) == 0 and len(zero_val) == 0)
            if multiple_val:
                violations.append(f"CV leakage: {len(multiple_val)} samples assigned to multiple validation folds")
            if zero_val:
                violations.append(f"CV incompleteness: {len(zero_val)} samples never assigned to a validation fold")

        passed = (len(violations) == 0)
        return {
            "leakage_audit_passed": passed,
            "checks": checks,
            "violations": violations,
        }


# =====================================================================
# 5. Output Artifacts and Reporting
# =====================================================================

def generate_markdown_audit_report(
    audit_data: Dict[str, Any],
    output_path: str
) -> None:
    """Renders human-readable dataset split audit report in Markdown format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    status_badge = "✅ PASSED" if audit_data.get("overall_validation_status") == "VALID" else "❌ BLOCKED"
    leak_badge = "✅ LEAK-FREE" if audit_data.get("leakage_audit", {}).get("leakage_audit_passed") else "❌ LEAK DETECTED"

    splits = audit_data.get("split_statistics", {}).get("splits", {})
    train_dist = splits.get("train", {}).get("class_distribution", {})
    val_dist = splits.get("val", {}).get("class_distribution", {})
    test_dist = splits.get("test", {}).get("class_distribution", {})

    lines = [
        "# Gold Benchmark Dataset Split Audit Report",
        "",
        f"**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  ",
        f"**Generated:** {audit_data.get('timestamp')}  ",
        f"**Gold Benchmark SHA-256:** `{audit_data.get('dataset_provenance', {}).get('sha256', 'UNKNOWN')}`  ",
        f"**Overall Status:** {status_badge}  ",
        f"**Leakage Audit:** {leak_badge}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Samples:** {audit_data.get('dataset_provenance', {}).get('record_count', 0)}",
        f"- **Total Connected Groups:** {audit_data.get('dataset_provenance', {}).get('unique_groups', 0)}",
        f"- **Random Seed:** `{audit_data.get('reproducibility_configuration', {}).get('seed')}`",
        f"- **Target Split:** 70% Train / 15% Validation / 15% Test",
        f"- **Cross-Validation:** 5-Fold StratifiedGroupKFold",
        "",
        "---",
        "",
        "## 2. Train / Validation / Test Group-Stratified Partitions",
        "",
        "| Partition | Sample Count | Actual % | Target % | Group Count | KNOWN | AMBIGUOUS | UNKNOWN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Train** | {splits.get('train', {}).get('count', 0)} | {splits.get('train', {}).get('percentage', 0)}% | 70.0% | {splits.get('train', {}).get('group_count', 0)} | {train_dist.get('KNOWN', 0)} | {train_dist.get('AMBIGUOUS', 0)} | {train_dist.get('UNKNOWN', 0)} |",
        f"| **Validation** | {splits.get('val', {}).get('count', 0)} | {splits.get('val', {}).get('percentage', 0)}% | 15.0% | {splits.get('val', {}).get('group_count', 0)} | {val_dist.get('KNOWN', 0)} | {val_dist.get('AMBIGUOUS', 0)} | {val_dist.get('UNKNOWN', 0)} |",
        f"| **Test** | {splits.get('test', {}).get('count', 0)} | {splits.get('test', {}).get('percentage', 0)}% | 15.0% | {splits.get('test', {}).get('group_count', 0)} | {test_dist.get('KNOWN', 0)} | {test_dist.get('AMBIGUOUS', 0)} | {test_dist.get('UNKNOWN', 0)} |",
        "",
        "---",
        "",
        "## 3. Five-Fold Cross-Validation Breakdown",
        "",
        "| Fold | Train Samples | Val Samples | Val % | Train Groups | Val Groups | Val KNOWN | Val AMBIGUOUS | Val UNKNOWN |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for fold in audit_data.get("cross_validation_statistics", {}).get("folds", []):
        f_idx = fold.get("fold_index")
        v_dist = fold.get("val_class_distribution", {})
        val_pct = round((fold.get("val_count", 0) / audit_data.get('dataset_provenance', {}).get('record_count', 1)) * 100.0, 1)
        lines.append(
            f"| Fold {f_idx} | {fold.get('train_count')} | {fold.get('val_count')} | {val_pct}% | {fold.get('train_group_count')} | {fold.get('val_group_count')} | {v_dist.get('KNOWN', 0)} | {v_dist.get('AMBIGUOUS', 0)} | {v_dist.get('UNKNOWN', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Independent Leakage Verification Matrix",
        "",
        "| Check Description | Target Constraint | Result |",
        "| :--- | :--- | :---: |",
    ])

    for check, res in audit_data.get("leakage_audit", {}).get("checks", {}).items():
        res_str = "PASSED" if res else "FAILED"
        lines.append(f"| `{check}` | Must be strictly disjoint / complete | **{res_str}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Scientific Limitations & Methodological Notice",
        "",
        "> **Methodological Notice:**  ",
        "> In group-stratified dataset splitting, questions belonging to the same lexical or semantic connected component ",
        "> (`question_group_id`) are held strictly atomic. Because group sizes vary integer-wise, achieved class ",
        "> proportions and partition sizes may deviate slightly from theoretical target ratios (70/15/15). ",
        "> This variance is mathematically required to guarantee zero group-level data leakage.",
        "",
        "- **Zero Information Leakage:** No sample, group, or paraphrased question appears in more than one partition.",
        "- **Strict Provenance Gate:** This split was derived exclusively from an adjudicated, finalized human gold benchmark.",
        "- **Reproducibility:** Splits can be bit-for-bit replicated using the specified random seed and configuration.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =====================================================================
# 6. CLI Driver
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe, group-stratified splits and 5-fold CV assignments for the gold benchmark."
    )
    parser.add_argument(
        "--dataset",
        required=False,
        default="data/benchmarks/uncertainty_benchmark_300_gold_v1.0.jsonl",
        help="Path to finalized gold benchmark JSONL.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to gold benchmark manifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic splitting (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments",
        help="Directory to write gold_split_v1.json and gold_cv5_v1.json.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory to write dataset_split_audit_v1.json and Markdown report.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing split and CV artifacts without regenerating.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Path to existing split JSON file to validate.",
    )
    parser.add_argument(
        "--cv-file",
        default=None,
        help="Path to existing CV JSON file to validate.",
    )

    args = parser.parse_args()

    # Mode A: Validate existing split artifacts
    if args.validate_existing:
        split_path = args.split_file or os.path.join(args.output_dir, "gold_split_v1.json")
        cv_path = args.cv_file or os.path.join(args.output_dir, "gold_cv5_v1.json")
        dataset_path = args.dataset

        print(f"Validating existing artifacts:\n  Dataset: {dataset_path}\n  Split: {split_path}\n  CV: {cv_path}")

        validator = GoldDatasetValidator()
        is_valid_gold, records, gold_errors, meta = validator.validate_file(dataset_path, args.manifest)
        if not is_valid_gold:
            print("Error: Dataset validation failed:")
            for err in gold_errors:
                print(f"  - {err}")
            sys.exit(2)

        if not os.path.exists(split_path):
            print(f"Error: Split file not found: {split_path}")
            sys.exit(1)
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        cv_data = None
        if os.path.exists(cv_path):
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_data = json.load(f)

        leakage_val = LeakageValidator()
        leak_res = leakage_val.validate_splits_and_cv(records, split_data, cv_data)

        if leak_res["leakage_audit_passed"]:
            print("SUCCESS: All split and CV leakage checks passed.")
            sys.exit(0)
        else:
            print("FAIL: Leakage detected in existing artifacts:")
            for viol in leak_res["violations"]:
                print(f"  - {viol}")
            sys.exit(1)

    # Mode B: Standard Generation Workflow
    validator = GoldDatasetValidator()
    is_valid_gold, records, gold_errors, meta = validator.validate_file(args.dataset, args.manifest)

    if not is_valid_gold:
        print("\n" + "=" * 50)
        print("GOLD BENCHMARK SPLIT GENERATION")
        print("=" * 50)
        print("Status:\nBLOCKED\n")
        print("Reason:\nInput dataset is not a finalized gold benchmark or failed verification.")
        for err in gold_errors[:10]:
            print(f"  - {err}")
        if len(gold_errors) > 10:
            print(f"  ... and {len(gold_errors) - 10} additional errors.")
        print("\nSafe failure safeguard active: Split and CV artifacts were NOT generated.")
        print("=" * 50 + "\n")
        sys.exit(2)

    # 1. Generate Group-Stratified 70/15/15 Split
    splitter = GroupStratifiedSplitter(seed=args.seed)
    split_result = splitter.split(records)

    # 2. Generate 5-Fold Cross-Validation
    cv_gen = FiveFoldCrossValidator(seed=args.seed)
    cv_result = cv_gen.split(records)

    # 3. Independent Leakage Audit
    leak_validator = LeakageValidator()
    leak_audit = leak_validator.validate_splits_and_cv(records, split_result, cv_result)

    if not leak_audit["leakage_audit_passed"]:
        print("CRITICAL ERROR: Leakage detected during split generation!")
        for viol in leak_audit["violations"]:
            print(f"  - {viol}")
        sys.exit(1)

    # 4. Prepare Output Artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    gold_sha = meta["sha256"]
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    split_artifact = {
        "dataset_version": meta["benchmark_version"],
        "gold_benchmark_path": args.dataset,
        "gold_benchmark_sha256": gold_sha,
        "random_seed": args.seed,
        "split_configuration": {
            "train_ratio": DEFAULT_TRAIN_RATIO,
            "val_ratio": DEFAULT_VAL_RATIO,
            "test_ratio": DEFAULT_TEST_RATIO,
            "group_variable": "question_group_id",
        },
        "splits": split_result["splits"],
        "group_information": {
            "total_groups": split_result["total_groups"],
            "group_assignments": split_result["group_assignments"],
        },
        "leakage_audit_passed": leak_audit["leakage_audit_passed"],
        "generated_timestamp": timestamp,
    }

    cv_artifact = {
        "dataset_version": meta["benchmark_version"],
        "gold_benchmark_path": args.dataset,
        "gold_benchmark_sha256": gold_sha,
        "random_seed": args.seed,
        "n_splits": DEFAULT_CV_FOLDS,
        "group_variable": "question_group_id",
        "folds": cv_result["folds"],
        "sample_to_fold_assignment": cv_result["sample_to_fold_assignment"],
        "leakage_audit_passed": leak_audit["leakage_audit_passed"],
        "generated_timestamp": timestamp,
    }

    audit_artifact = {
        "timestamp": timestamp,
        "dataset_integrity_status": "PASSED" if is_valid_gold else "FAILED",
        "dataset_provenance": meta,
        "gold_label_validation_results": {
            "classes_verified": list(VALID_LABELS),
            "label_distribution": meta["label_distribution"],
        },
        "class_distributions": {
            "overall": meta["label_distribution"],
            "train": split_result["splits"]["train"]["class_distribution"],
            "val": split_result["splits"]["val"]["class_distribution"],
            "test": split_result["splits"]["test"]["class_distribution"],
        },
        "group_distributions": {
            "total_groups": split_result["total_groups"],
        },
        "leakage_audit": leak_audit,
        "reproducibility_configuration": {
            "seed": args.seed,
            "split_ratios": [DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO],
            "cv_folds": DEFAULT_CV_FOLDS,
        },
        "split_statistics": split_result,
        "cross_validation_statistics": cv_result,
        "warnings": split_result["warnings"] + cv_result["warnings"],
        "overall_validation_status": "VALID" if (is_valid_gold and leak_audit["leakage_audit_passed"]) else "INVALID",
    }

    # Write files
    split_file_out = os.path.join(args.output_dir, "gold_split_v1.json")
    cv_file_out = os.path.join(args.output_dir, "gold_cv5_v1.json")
    audit_json_out = os.path.join(args.reports_dir, "dataset_split_audit_v1.json")
    audit_md_out = os.path.join(args.reports_dir, "DATASET_SPLIT_AUDIT_REPORT.md")

    with open(split_file_out, "w", encoding="utf-8") as f:
        json.dump(split_artifact, f, indent=2)

    with open(cv_file_out, "w", encoding="utf-8") as f:
        json.dump(cv_artifact, f, indent=2)

    with open(audit_json_out, "w", encoding="utf-8") as f:
        json.dump(audit_artifact, f, indent=2)

    generate_markdown_audit_report(audit_artifact, audit_md_out)

    print("\n" + "=" * 50)
    print("GOLD BENCHMARK SPLIT GENERATION: SUCCESS")
    print("=" * 50)
    print(f"Generated Split Artifact: {split_file_out}")
    print(f"Generated CV Artifact:    {cv_file_out}")
    print(f"Generated Audit JSON:     {audit_json_out}")
    print(f"Generated Audit Report:   {audit_md_out}")
    print(f"Leakage Checks:           {len(leak_audit['checks'])} passed, 0 violations")
    print("=" * 50 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
