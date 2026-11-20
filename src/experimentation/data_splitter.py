"""Reproducible, leakage-safe data splitting with group disjointness guarantees."""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Encapsulates split indices, sample IDs, class distributions, and leakage audit."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    train_distribution: Dict[str, int]
    val_distribution: Dict[str, int]
    test_distribution: Dict[str, int]
    group_disjoint: bool
    split_ratios: Tuple[float, float, float]
    random_seed: int
    is_small_sample: bool
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_count": len(self.train_indices),
            "val_count": len(self.val_indices),
            "test_count": len(self.test_indices),
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "train_distribution": self.train_distribution,
            "val_distribution": self.val_distribution,
            "test_distribution": self.test_distribution,
            "group_disjoint": self.group_disjoint,
            "split_ratios": list(self.split_ratios),
            "random_seed": self.random_seed,
            "is_small_sample": self.is_small_sample,
            "limitations": self.limitations,
        }

    def save(self, output_path: str) -> None:
        """Serialize split information to JSON for auditability."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class LeakageSafeSplitter:
    """Partitions dataset into train/val/test splits with group disjointness and stratified class balance."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        total = round(train_ratio + val_ratio + test_ratio, 4)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total})")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self,
        records: List[Dict[str, Any]],
        y: np.ndarray,
        ids: List[str],
        groups: Optional[List[str]] = None,
    ) -> SplitResult:
        """Perform stratified, group-aware train/val/test partition with zero leakage.
        
        Guarantees:
        1. Connected Groups: Identical question groups never cross partition boundaries.
        2. Class Stratification: StratifiedGroupKFold is leveraged so each split balances class proportions.
        3. Class Coverage Check: Verifies all classes appear in all splits, or documents limitation.
        """
        n_samples = len(records)
        limitations: List[str] = []
        is_small = n_samples < 100

        if is_small:
            limitations.append(
                f"Small sample size limitation: Total records ({n_samples}) < 100. "
                "Test partition performance carries high empirical variance. "
                "Recommend stratified cross-validation or expanding development dataset."
            )

        if groups is None:
            groups = [str(r.get("question_group_id") or r.get("id")) for r in records]

        unique_groups = sorted(list(set(groups)))
        n_groups = len(unique_groups)

        grp_to_idx = {g: i for i, g in enumerate(unique_groups)}
        group_indices = np.array([grp_to_idx[g] for g in groups])

        # Calculate fold counts for outer and inner stratified group splits
        n_outer_folds = max(2, min(n_groups, int(round(1.0 / self.test_ratio))))
        
        final_train_idx = np.empty(0, dtype=int)
        final_val_idx = np.empty(0, dtype=int)
        final_test_idx = np.empty(0, dtype=int)

        split_success = False

        # Attempt StratifiedGroupKFold
        if n_groups >= n_outer_folds and len(np.unique(y)) > 1:
            try:
                sgkf_outer = StratifiedGroupKFold(
                    n_splits=n_outer_folds,
                    shuffle=True,
                    random_state=self.random_seed,
                )
                
                # Search for an outer fold where test has all classes
                for train_val_idx, test_idx in sgkf_outer.split(records, y, group_indices):
                    # Check inner split
                    n_inner_folds = max(2, min(len(set(group_indices[train_val_idx])), int(round((self.train_ratio + self.val_ratio) / self.val_ratio))))
                    sgkf_inner = StratifiedGroupKFold(
                        n_splits=n_inner_folds,
                        shuffle=True,
                        random_state=self.random_seed,
                    )
                    
                    sub_records = [records[i] for i in train_val_idx]
                    sub_y = y[train_val_idx]
                    sub_groups = group_indices[train_val_idx]
                    
                    for train_sub_idx, val_sub_idx in sgkf_inner.split(sub_records, sub_y, sub_groups):
                        cand_train = train_val_idx[train_sub_idx]
                        cand_val = train_val_idx[val_sub_idx]
                        cand_test = test_idx

                        # Check if all classes present in train, val, and test
                        classes_present = set(np.unique(y))
                        if (
                            set(np.unique(y[cand_train])) == classes_present and
                            set(np.unique(y[cand_val])) == classes_present and
                            set(np.unique(y[cand_test])) == classes_present
                        ):
                            final_train_idx = cand_train
                            final_val_idx = cand_val
                            final_test_idx = cand_test
                            split_success = True
                            break
                    if split_success:
                        break

                if not split_success:
                    # Take first valid fold assignment if perfect class representation in all 3 was impossible
                    for train_val_idx, test_idx in sgkf_outer.split(records, y, group_indices):
                        n_inner_folds = max(2, min(len(set(group_indices[train_val_idx])), int(round((self.train_ratio + self.val_ratio) / self.val_ratio))))
                        sgkf_inner = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=self.random_seed)
                        sub_records = [records[i] for i in train_val_idx]
                        sub_y = y[train_val_idx]
                        sub_groups = group_indices[train_val_idx]
                        train_sub_idx, val_sub_idx = next(sgkf_inner.split(sub_records, sub_y, sub_groups))
                        final_train_idx = train_val_idx[train_sub_idx]
                        final_val_idx = train_val_idx[val_sub_idx]
                        final_test_idx = test_idx
                        split_success = True
                        break
            except Exception as e:
                logger.warning("StratifiedGroupKFold failed (%s); falling back to GroupShuffleSplit", e)
                split_success = False

        # Fallback to GroupShuffleSplit if StratifiedGroupKFold cannot be run
        if not split_success:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_seed)
            train_val_idx, test_idx = next(gss_test.split(records, y, group_indices))

            val_relative_size = self.val_ratio / (self.train_ratio + self.val_ratio)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=self.random_seed)
            train_sub_idx, val_sub_idx = next(
                gss_val.split(
                    [records[i] for i in train_val_idx],
                    y[train_val_idx],
                    group_indices[train_val_idx],
                )
            )
            final_train_idx = train_val_idx[train_sub_idx]
            final_val_idx = train_val_idx[val_sub_idx]
            final_test_idx = test_idx

        # Leakage verification
        train_set_ids = set(ids[i] for i in final_train_idx)
        val_set_ids = set(ids[i] for i in final_val_idx)
        test_set_ids = set(ids[i] for i in final_test_idx)

        id_disjoint = (
            len(train_set_ids.intersection(val_set_ids)) == 0 and
            len(train_set_ids.intersection(test_set_ids)) == 0 and
            len(val_set_ids.intersection(test_set_ids)) == 0
        )

        train_grps = set(groups[i] for i in final_train_idx)
        val_grps = set(groups[i] for i in final_val_idx)
        test_grps = set(groups[i] for i in final_test_idx)

        group_disjoint = (
            id_disjoint and
            len(train_grps.intersection(val_grps)) == 0 and
            len(train_grps.intersection(test_grps)) == 0 and
            len(val_grps.intersection(test_grps)) == 0
        )

        if not group_disjoint:
            limitations.append("CRITICAL: Group leakage detected across partitions!")

        from src.experimentation.dataset_loader import ID_TO_LABEL
        def get_dist(indices: np.ndarray) -> Dict[str, int]:
            cnts = Counter(y[i] for i in indices)
            return {ID_TO_LABEL[k]: cnts.get(k, 0) for k in [0, 1, 2]}

        train_dist = get_dist(final_train_idx)
        val_dist = get_dist(final_val_idx)
        test_dist = get_dist(final_test_idx)

        # Check if any partition lacks a class
        dataset_classes = set(ID_TO_LABEL[k] for k in np.unique(y))
        for pname, pdist in [("train", train_dist), ("val", val_dist), ("test", test_dist)]:
            missing = [c for c in dataset_classes if pdist.get(c, 0) == 0]
            if missing:
                limitations.append(f"Partition '{pname}' lacks representation for class(es): {missing}")

        return SplitResult(
            train_indices=final_train_idx,
            val_indices=final_val_idx,
            test_indices=final_test_idx,
            train_ids=[ids[i] for i in final_train_idx],
            val_ids=[ids[i] for i in final_val_idx],
            test_ids=[ids[i] for i in final_test_idx],
            train_distribution=train_dist,
            val_distribution=val_dist,
            test_distribution=test_dist,
            group_disjoint=group_disjoint,
            split_ratios=(self.train_ratio, self.val_ratio, self.test_ratio),
            random_seed=self.random_seed,
            is_small_sample=is_small,
            limitations=limitations,
        )
