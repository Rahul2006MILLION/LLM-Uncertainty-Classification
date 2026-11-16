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
    LABEL_FIELDS = ("label", "gold_label")
    GROUP_FIELDS = ("question_group_id", "group_id", "group_key")

    @classmethod
    def extract_label(cls, rec: Dict[str, Any]) -> str:
        """Extract canonical class label from record with disagreement detection.

        Supports alternative field names 'label' and 'gold_label'.
        Raises ValueError if both are provided and disagree.
        Raises ValueError if neither is provided.
        Returns the non-empty canonical label string.
        """
        has_label = "label" in rec and rec["label"] is not None and str(rec["label"]).strip() != ""
        has_gold = "gold_label" in rec and rec["gold_label"] is not None and str(rec["gold_label"]).strip() != ""
        rec_id = rec.get("id", "unknown")

        if has_label and has_gold:
            lbl_val = str(rec["label"]).strip()
            gold_val = str(rec["gold_label"]).strip()
            if lbl_val != gold_val:
                raise ValueError(
                    f"Conflicting label fields in record {rec_id}: "
                    f"'label'='{lbl_val}' vs 'gold_label'='{gold_val}'"
                )
            return lbl_val
        elif has_label:
            return str(rec["label"]).strip()
        elif has_gold:
            return str(rec["gold_label"]).strip()
        raise ValueError(f"Record {rec_id} missing required label field ('label' or 'gold_label')")

    @classmethod
    def extract_group_ids(cls, rec: Dict[str, Any]) -> List[str]:
        """Extract declared group identifiers from record with disagreement detection.

        Supports alternative field names 'question_group_id', 'group_id', 'group_key'.
        Supports single scalar string or composite list/set of group strings.
        Raises ValueError if multiple alternative scalar group fields disagree.
        Falls back to normalized question text if no explicit group is declared.
        """
        scalar_fields: Dict[str, str] = {}
        composite_groups: set = set()

        for field_name in cls.GROUP_FIELDS:
            val = rec.get(field_name)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                for item in val:
                    if item is not None and str(item).strip() != "":
                        composite_groups.add(str(item).strip())
            elif isinstance(val, str):
                sval = val.strip()
                if sval:
                    scalar_fields[field_name] = sval
            elif isinstance(val, (int, float)):
                scalar_fields[field_name] = str(val)

        # Disagreement check between alternative scalar group fields
        if len(scalar_fields) > 1:
            unique_scalars = set(scalar_fields.values())
            if len(unique_scalars) > 1:
                rec_id = rec.get("id", "unknown")
                raise ValueError(
                    f"Conflicting group fields in record {rec_id}: {scalar_fields}"
                )

        if composite_groups:
            if scalar_fields:
                composite_groups.update(scalar_fields.values())
            return sorted(list(composite_groups))
        elif scalar_fields:
            return [next(iter(scalar_fields.values()))]
        else:
            q_text = rec.get("question", "")
            return [cls.get_question_group_key(q_text)]

    @classmethod
    def extract_record_fields(cls, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract canonical dictionary with standardized keys: id, question, normalized_question, label, group_ids, source."""
        return {
            "id": rec.get("id"),
            "question": rec.get("question", ""),
            "normalized_question": cls.normalize_question(rec.get("question", "")),
            "label": cls.extract_label(rec),
            "group_ids": cls.extract_group_ids(rec),
            "source": rec.get("source", ""),
        }

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize question text by lowercasing, stripping punctuation, and collapsing whitespace.
        
        Guarantees that identical questions with formatting, punctuation, or casing
        differences map to the identical canonical representation.
        """
        if not question:
            return ""
        clean = re.sub(r"[^\w\s]", "", str(question).lower()).strip()
        return re.sub(r"\s+", " ", clean)

    @classmethod
    def validate_records(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate an array of question records against schema rules.

        Performs:
        1. Required field checks (supports 'label' or 'gold_label'; 'id', 'question').
        2. Detection of conflicting alternative fields (label vs gold_label; conflicting group keys).
        3. Exact duplicate question detection via normalized question text.
        4. Conflicting label detection across identical duplicate questions.
        5. Class coverage verification across dataset.
        """
        errors = []
        class_counts = {lbl: 0 for lbl in cls.VALID_LABELS}
        norm_to_records: Dict[str, List[Dict[str, Any]]] = {}
        duplicates: List[Dict[str, Any]] = []
        conflicting_labels: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"record_{idx}")
            if not rec.get("id"):
                errors.append(f"Record {idx} missing required field 'id'")
            if not rec.get("question"):
                errors.append(f"Record {idx} missing required field 'question'")

            # Check label with conflict detection
            try:
                label = cls.extract_label(rec)
            except ValueError as e:
                errors.append(str(e))
                label = ""

            if label and label not in cls.VALID_LABELS:
                errors.append(f"Record {rec_id} has invalid label '{label}'")
            elif label in cls.VALID_LABELS:
                class_counts[label] += 1

            # Check group fields with conflict detection
            try:
                _ = cls.extract_group_ids(rec)
            except ValueError as e:
                errors.append(str(e))

            q_raw = rec.get("question", "")
            q_norm = cls.normalize_question(q_raw)
            if not q_norm:
                errors.append(f"Record {rec_id} has empty normalized question text")
                continue

            rec_entry = {
                "id": rec.get("id"),
                "question": q_raw,
                "normalized": q_norm,
                "label": label,
                "source": rec.get("source"),
            }

            if q_norm in norm_to_records:
                duplicates.append({
                    "id": rec.get("id"),
                    "question": q_raw,
                    "normalized": q_norm,
                    "label": label,
                    "first_seen_id": norm_to_records[q_norm][0].get("id"),
                })
            norm_to_records.setdefault(q_norm, []).append(rec_entry)

        # Check for conflicting labels across duplicate normalized questions
        for q_norm, recs in norm_to_records.items():
            if len(recs) > 1:
                labels_found = set(r.get("label") for r in recs if r.get("label") in cls.VALID_LABELS)
                if len(labels_found) > 1:
                    conflict_info = {
                        "normalized_question": q_norm,
                        "conflicting_labels": sorted(list(labels_found)),
                        "record_instances": [
                            {"id": r.get("id"), "label": r.get("label"), "question": r.get("question")}
                            for r in recs
                        ],
                    }
                    conflicting_labels.append(conflict_info)
                    errors.append(
                        f"Conflicting labels detected for duplicate question '{q_norm}': "
                        f"labels={sorted(list(labels_found))} across records {[r.get('id') for r in recs]}. "
                        f"Manual review required; label will not be silently selected."
                    )

        # Check class coverage across dataset if sample count permits representation of all classes
        if len(records) >= len(cls.VALID_LABELS):
            missing_classes = [c for c, count in class_counts.items() if count == 0]
            if missing_classes:
                errors.append(f"Dataset lacks representation for classes: {missing_classes}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "total_records": len(records),
            "errors": errors,
            "schema_errors": errors,
            "class_counts": class_counts,
            "label_distribution": class_counts,
            "unique_questions": len(norm_to_records),
            "duplicate_count": len(duplicates),
            "duplicates": duplicates,
            "conflicting_labels": conflicting_labels,
            "conflicting_label_count": len(conflicting_labels),
        }

    @classmethod
    def get_question_group_key(cls, question: str) -> str:
        """Derive a canonical group key for a question.
        
        Conservative Grouping Policy:
        Groups records strictly by their normalized question text (lower-cased, punctuation-stripped,
        whitespace-collapsed).
        
        Audit and Limitations of Grouping Policy:
        - Prior heuristic: The previous implementation grouped by the first four words of the
          question. This heuristic was flawed: it over-grouped distinct questions sharing common
          initial prefixes (e.g., 'What is the capital of France?' vs 'What is the capital of
          Georgia?' both mapped to 'what_is_the_capital'), forcing unrelated queries into one
          partition, while failing to group true paraphrases that had different word order
          (e.g., 'What is France's capital?').
        - Conservative Policy: The updated policy ensures that all records with the identical
          normalized question remain strictly in the same partition.
        - Documented Limitation: Lexical normalization guarantees that exact duplicate questions
          and formatting/casing variations never leak across train/val/test partitions. However,
          purely lexical normalization CANNOT reliably detect semantic paraphrases (e.g. 'What is
          the capital of France?' vs 'Tell me France's capital city'). In datasets where semantic
          paraphrases are present, records should provide an explicit 'group_id' or 'group_key'
          derived from curated annotations or semantic cluster embeddings.
        """
        return cls.normalize_question(question)

    @classmethod
    def group_train_val_test_split(
        cls,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        return_metadata: bool = False,
    ) -> Any:
        """Perform leak-free group-based train/val/test split supporting cross-label adversarial groups.

        Guarantees:
        1. Connected Components: Any records sharing a declared group identifier or identical question text
           are merged into a single indivisible connected component before splitting. Overlapping
           composite groups are strictly unified.
        2. Cross-Label Adversarial Support: Groups containing questions with different labels (e.g.
           adversarial pairs testing boundary conditions) are fully supported and held out together.
           Group label homogeneity is NOT required.
        3. Duplicate Integrity & Conflicting Label Detection: Raises ValueError if any identical
           normalized duplicate questions carry conflicting labels.
        4. Mutual Disjointness: Programmatically verifies that train, val, and test group sets
           are strictly disjoint. No group or question crosses partition boundaries.
        5. Stratified Allocation: Allocates multi-label and single-label groups to balance class
           quotas and partition sizes.
        6. Distribution Reporting & Warnings: Reports class distribution per split and documents
           warnings when the number of groups makes stratification unreliable.
        """
        if not records:
            empty_meta = {
                "total_records": 0,
                "total_groups": 0,
                "train_record_count": 0,
                "val_record_count": 0,
                "test_record_count": 0,
                "train_group_count": 0,
                "val_group_count": 0,
                "test_group_count": 0,
                "train_class_distribution": {lbl: 0 for lbl in cls.VALID_LABELS},
                "val_class_distribution": {lbl: 0 for lbl in cls.VALID_LABELS},
                "test_class_distribution": {lbl: 0 for lbl in cls.VALID_LABELS},
                "train_groups_by_class": {lbl: 0 for lbl in cls.VALID_LABELS},
                "val_groups_by_class": {lbl: 0 for lbl in cls.VALID_LABELS},
                "test_groups_by_class": {lbl: 0 for lbl in cls.VALID_LABELS},
                "is_disjoint": True,
                "limitations": ["Dataset contains no records."],
            }
            if return_metadata:
                return [], [], [], empty_meta
            return [], [], []

        # Step 1: Extract canonical fields for all records
        canonical_records = []
        for idx, rec in enumerate(records):
            c_fields = cls.extract_record_fields(rec)
            lbl = c_fields["label"]
            if lbl not in cls.VALID_LABELS:
                rec_id = c_fields["id"] or f"idx_{idx}"
                raise ValueError(f"Record {rec_id} has invalid or missing label '{lbl}'")
            canonical_records.append(c_fields)

        # Step 2: Detect duplicate questions with conflicting labels
        norm_to_labels: Dict[str, set] = {}
        for c_rec in canonical_records:
            nq = c_rec["normalized_question"]
            norm_to_labels.setdefault(nq, set()).add(c_rec["label"])

        conflicting_dups = {nq: sorted(list(lbls)) for nq, lbls in norm_to_labels.items() if len(lbls) > 1}
        if conflicting_dups:
            raise ValueError(
                f"Global group split failed: conflicting labels detected on duplicate question text: {conflicting_dups}"
            )

        # Step 3: Build Connected Components (Union-Find)
        n = len(records)
        parent = list(range(n))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Union identical normalized questions
        norm_first_seen: Dict[str, int] = {}
        for idx, c_rec in enumerate(canonical_records):
            nq = c_rec["normalized_question"]
            if nq in norm_first_seen:
                union(idx, norm_first_seen[nq])
            else:
                norm_first_seen[nq] = idx

        # Union overlapping group identifiers (composite group support)
        group_first_seen: Dict[str, int] = {}
        for idx, c_rec in enumerate(canonical_records):
            for gid in c_rec["group_ids"]:
                if gid in group_first_seen:
                    union(idx, group_first_seen[gid])
                else:
                    group_first_seen[gid] = idx

        # Step 4: Assemble Connected Components
        comp_map: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            comp_map.setdefault(root, []).append(idx)

        components = []
        for root, member_indices in comp_map.items():
            comp_records = [records[i] for i in member_indices]
            comp_gids = set()
            class_counts = {lbl: 0 for lbl in cls.VALID_LABELS}
            for i in member_indices:
                comp_gids.update(canonical_records[i]["group_ids"])
                lbl = canonical_records[i]["label"]
                class_counts[lbl] += 1

            components.append({
                "root": root,
                "indices": member_indices,
                "records": comp_records,
                "group_ids": comp_gids,
                "class_counts": class_counts,
                "total_records": len(member_indices),
                "num_classes": sum(1 for c, cnt in class_counts.items() if cnt > 0),
                "comp_key": sorted(list(comp_gids))[0] if comp_gids else f"comp_{root}",
            })

        # Step 5: Stratified Multi-Label Split Allocation
        m_components = len(components)
        split_ratios = [train_ratio, val_ratio, test_ratio]
        eligible_splits = [s for s, r in enumerate(split_ratios) if r > 0]

        train_components: List[Dict[str, Any]] = []
        val_components: List[Dict[str, Any]] = []
        test_components: List[Dict[str, Any]] = []
        split_comp_lists = [train_components, val_components, test_components]

        # Target class counts per partition
        total_class_counts = {
            lbl: sum(comp["class_counts"][lbl] for comp in components)
            for lbl in cls.VALID_LABELS
        }
        target_counts = [
            {lbl: total_class_counts[lbl] * r for lbl in cls.VALID_LABELS}
            for r in split_ratios
        ]

        if m_components == 1:
            train_components.append(components[0])
        elif m_components == 2:
            train_components.append(components[0])
            if test_ratio > 0:
                test_components.append(components[1])
            elif val_ratio > 0:
                val_components.append(components[1])
            else:
                train_components.append(components[1])
        else:
            # Deterministic ordering: multi-label / larger components first
            shuffled_components = list(components)
            shuffled_components.sort(key=lambda c: (-c["num_classes"], -c["total_records"], c["comp_key"]))

            assigned_class_counts = [
                {lbl: 0 for lbl in cls.VALID_LABELS}
                for _ in range(3)
            ]
            assigned_total_counts = [0, 0, 0]

            for comp_idx, comp in enumerate(shuffled_components):
                # Ensure all eligible splits receive at least one component
                empty_splits = [s for s in eligible_splits if len(split_comp_lists[s]) == 0]
                remaining_comps = len(shuffled_components) - comp_idx
                if empty_splits and remaining_comps <= len(empty_splits):
                    best_s = empty_splits[0]
                else:
                    best_s = eligible_splits[0]
                    best_score = float("inf")
                    for s in eligible_splits:
                        r_s = split_ratios[s]
                        class_cost = 0.0
                        for lbl in cls.VALID_LABELS:
                            tgt = target_counts[s][lbl]
                            curr = assigned_class_counts[s][lbl] + comp["class_counts"][lbl]
                            class_cost += ((curr - tgt) / (tgt + 1.0)) ** 2

                        size_tgt = n * r_s
                        size_curr = assigned_total_counts[s] + comp["total_records"]
                        size_cost = ((size_curr - size_tgt) / (size_tgt + 1.0)) ** 2

                        score = class_cost + 0.5 * size_cost
                        if score < best_score:
                            best_score = score
                            best_s = s

                split_comp_lists[best_s].append(comp)
                for lbl in cls.VALID_LABELS:
                    assigned_class_counts[best_s][lbl] += comp["class_counts"][lbl]
                assigned_total_counts[best_s] += comp["total_records"]

        # Step 6: Assemble partition records and group sets
        train_set = [r for comp in train_components for r in comp["records"]]
        val_set = [r for comp in val_components for r in comp["records"]]
        test_set = [r for comp in test_components for r in comp["records"]]

        train_groups = {gid for comp in train_components for gid in comp["group_ids"]}
        val_groups = {gid for comp in val_components for gid in comp["group_ids"]}
        test_groups = {gid for comp in test_components for gid in comp["group_ids"]}

        # Step 7: Verify mutual disjointness
        train_val_overlap = train_groups & val_groups
        train_test_overlap = train_groups & test_groups
        val_test_overlap = val_groups & test_groups

        is_disjoint = (len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0)
        if not is_disjoint:
            raise RuntimeError(
                f"Data leakage detected! Non-disjoint groups across partitions: "
                f"train/val={train_val_overlap}, train/test={train_test_overlap}, val/test={val_test_overlap}"
            )

        # Step 8: Class coverage and group distribution reporting
        train_class_dist = {lbl: sum(1 for r in train_set if cls.extract_label(r) == lbl) for lbl in cls.VALID_LABELS}
        val_class_dist = {lbl: sum(1 for r in val_set if cls.extract_label(r) == lbl) for lbl in cls.VALID_LABELS}
        test_class_dist = {lbl: sum(1 for r in test_set if cls.extract_label(r) == lbl) for lbl in cls.VALID_LABELS}

        train_groups_by_class = {
            lbl: sum(1 for comp in train_components if comp["class_counts"][lbl] > 0)
            for lbl in cls.VALID_LABELS
        }
        val_groups_by_class = {
            lbl: sum(1 for comp in val_components if comp["class_counts"][lbl] > 0)
            for lbl in cls.VALID_LABELS
        }
        test_groups_by_class = {
            lbl: sum(1 for comp in test_components if comp["class_counts"][lbl] > 0)
            for lbl in cls.VALID_LABELS
        }

        limitations = []
        for lbl in cls.VALID_LABELS:
            if train_groups_by_class[lbl] == 0:
                limitations.append(f"Train partition lacks representation for class '{lbl}' (0 groups).")
            if val_ratio > 0 and val_groups_by_class[lbl] == 0:
                limitations.append(f"Validation partition lacks representation for class '{lbl}' (0 groups).")
            if test_ratio > 0 and test_groups_by_class[lbl] == 0:
                limitations.append(f"Test partition lacks representation for class '{lbl}' (0 groups).")

            # Warn if number of groups makes stratification unreliable
            if m_components >= 6:
                if test_ratio > 0 and 0 < test_groups_by_class[lbl] < 2:
                    limitations.append(
                        f"Low group count ({test_groups_by_class[lbl]}) for class '{lbl}' in test partition may make evaluation variance high."
                    )
                if val_ratio > 0 and 0 < val_groups_by_class[lbl] < 2:
                    limitations.append(
                        f"Low group count ({val_groups_by_class[lbl]}) for class '{lbl}' in validation partition may make evaluation variance high."
                    )

        metadata = {
            "total_records": len(records),
            "total_groups": len(components),
            "train_record_count": len(train_set),
            "val_record_count": len(val_set),
            "test_record_count": len(test_set),
            "train_group_count": len(train_components),
            "val_group_count": len(val_components),
            "test_group_count": len(test_components),
            "train_class_distribution": train_class_dist,
            "val_class_distribution": val_class_dist,
            "test_class_distribution": test_class_dist,
            "train_groups_by_class": train_groups_by_class,
            "val_groups_by_class": val_groups_by_class,
            "test_groups_by_class": test_groups_by_class,
            "is_disjoint": is_disjoint,
            "limitations": limitations,
        }

        if return_metadata:
            return train_set, val_set, test_set, metadata
        return train_set, val_set, test_set

    @classmethod
    def verify_split(
        cls,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Programmatically verify split disjointness and class coverage across partitions."""
        train_groups = {gid for r in train_records for gid in cls.extract_group_ids(r)}
        val_groups = {gid for r in val_records for gid in cls.extract_group_ids(r)}
        test_groups = {gid for r in test_records for gid in cls.extract_group_ids(r)}

        train_val_overlap = train_groups & val_groups
        train_test_overlap = train_groups & test_groups
        val_test_overlap = val_groups & test_groups

        is_disjoint = (len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0)

        def _groups_by_class(recs):
            res = {lbl: 0 for lbl in cls.VALID_LABELS}
            seen_group_class = set()
            for r in recs:
                lbl = cls.extract_label(r)
                if lbl in res:
                    for gid in cls.extract_group_ids(r):
                        key = (gid, lbl)
                        if key not in seen_group_class:
                            res[lbl] += 1
                            seen_group_class.add(key)
            return res

        def _class_dist(recs):
            dist = {lbl: 0 for lbl in cls.VALID_LABELS}
            for r in recs:
                lbl = cls.extract_label(r)
                if lbl in dist:
                    dist[lbl] += 1
            return dist

        return {
            "is_disjoint": is_disjoint,
            "train_val_overlap": list(train_val_overlap),
            "train_test_overlap": list(train_test_overlap),
            "val_test_overlap": list(val_test_overlap),
            "train_group_count": len(train_groups),
            "val_group_count": len(val_groups),
            "test_group_count": len(test_groups),
            "train_class_distribution": _class_dist(train_records),
            "val_class_distribution": _class_dist(val_records),
            "test_class_distribution": _class_dist(test_records),
            "train_groups_by_class": _groups_by_class(train_records),
            "val_groups_by_class": _groups_by_class(val_records),
            "test_groups_by_class": _groups_by_class(test_records),
        }
