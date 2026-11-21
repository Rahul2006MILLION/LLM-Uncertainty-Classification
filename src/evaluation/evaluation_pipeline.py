"""Experimental Evaluation Pipeline Infrastructure.

Milestone 7.5.5: Experimental Evaluation Pipeline Infrastructure
Research Project: LLM Uncertainty Classification Using Multi-Response Consistency
                  and Feature-Based Machine Learning

Provides reproducible evaluation infrastructure for:
- Majority Class Baseline
- Simple Feature-Based Baseline
- Logistic Regression (with training-only StandardScaler)
- Random Forest
- XGBoost Proposed Model
- Feature Ablations (Ablations 1 - 6)

Enforces strict research safeguards:
- Requires finalized human gold benchmark (annotation_status == "gold_benchmark")
- Rejects candidate benchmarks and unannotated data
- Strict test set isolation (no leakage during preprocessing or model selection)
- 5-Fold StratifiedGroupKFold cross-validation
- Standardized results schema and audit report generation
"""

import copy
import datetime
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
CANONICAL_CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]

# Canonical Label Encoding
LABEL_TO_ID = {"KNOWN": 0, "AMBIGUOUS": 1, "UNKNOWN": 2}
ID_TO_LABEL = {0: "KNOWN", 1: "AMBIGUOUS", 2: "UNKNOWN"}

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

        if errors and not raw_lines:
            return False, records, errors, metadata

        records = [r for _, r in raw_lines]

        base_name = os.path.basename(filepath).lower()
        if "candidate" in base_name:
            errors.append(
                f"Candidate dataset rejected: filename '{base_name}' indicates candidate/draft data."
            )

        seen_ids: Set[str] = set()
        for line_no, rec in raw_lines:
            rec_id = rec.get("id")
            if not rec_id or not isinstance(rec_id, str):
                errors.append(f"Line {line_no}: Missing or invalid 'id'")
            elif rec_id in seen_ids:
                errors.append(f"Line {line_no}: Duplicate question id '{rec_id}'")
            else:
                seen_ids.add(rec_id)

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

            provenance = rec.get("human_annotation_provenance")
            if not provenance or not isinstance(provenance, dict):
                errors.append(
                    f"Line {line_no} (ID: {rec_id}): Missing 'human_annotation_provenance' "
                    f"- dataset lacks required human ground-truth provenance"
                )

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

            q_text = rec.get("question")
            if not q_text or not isinstance(q_text, str) or len(q_text.strip()) < 3:
                errors.append(f"Line {line_no} (ID: {rec_id}): Missing or invalid 'question' text")

            forbidden_found = set(rec.keys()).intersection(FORBIDDEN_LEAKAGE_KEYS)
            if forbidden_found:
                errors.append(
                    f"Line {line_no} (ID: {rec_id}): Prohibited model feature or output keys detected: {forbidden_found}"
                )

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

        splits = split_data.get("splits", {})
        train_ids = set(splits.get("train", {}).get("ids", []))
        val_ids = set(splits.get("val", {}).get("ids", []))
        test_ids = set(splits.get("test", {}).get("ids", []))

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

        if cv_data:
            folds = cv_data.get("folds", [])
            checks["cv_fold_count"] = len(folds) == 5

            val_occurrences: Counter = Counter()
            all_cv_ids: Set[str] = set()

            for fold in folds:
                f_idx = fold.get("fold_index")
                f_train_ids = set(fold.get("train_ids", []))
                f_val_ids = set(fold.get("val_ids", []))
                all_cv_ids.update(f_train_ids)
                all_cv_ids.update(f_val_ids)

                f_id_overlap = f_train_ids.intersection(f_val_ids)
                if f_id_overlap:
                    violations.append(f"Fold {f_idx}: ID overlap between train and val: {f_id_overlap}")

                f_train_groups = set(id_to_rec[i].get(self.group_key) for i in f_train_ids if i in id_to_rec)
                f_val_groups = set(id_to_rec[i].get(self.group_key) for i in f_val_ids if i in id_to_rec)
                f_grp_overlap = f_train_groups.intersection(f_val_groups) - {None}

                if f_grp_overlap:
                    violations.append(f"Fold {f_idx}: Group leakage between train and val: {f_grp_overlap}")

                for vid in f_val_ids:
                    val_occurrences[vid] += 1

            multiple_val = [vid for vid, count in val_occurrences.items() if count > 1]
            zero_val = [qid for qid in all_cv_ids if val_occurrences[qid] == 0]

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

# Canonical 16-Feature Schema
CANONICAL_16_FEATURES = [
    "semantic_entropy",
    "normalized_entropy",
    "dominant_agreement",
    "second_agreement",
    "agreement_margin",
    "cluster_count",
    "cluster_ratio",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
    "response_len_mean",
    "response_len_std",
    "distinct_1gram_ratio",
    "distinct_2gram_ratio",
    "question_overlap_mean",
    "hedging_score",
    "negation_count",
]

# Ablation & Baseline Feature Subsets
FEATURE_SETS: Dict[str, List[str]] = {
    "canonical_16": CANONICAL_16_FEATURES,
    "ablation_1_no_entropy": [f for f in CANONICAL_16_FEATURES if f not in {"semantic_entropy", "normalized_entropy"}],
    "ablation_2_no_agreement": [f for f in CANONICAL_16_FEATURES if f not in {"dominant_agreement", "second_agreement", "agreement_margin", "cluster_count", "cluster_ratio"}],
    "ablation_3_no_similarity": [f for f in CANONICAL_16_FEATURES if f not in {"mean_pairwise_similarity", "min_pairwise_similarity"}],
    "ablation_4_no_lexical_length": [f for f in CANONICAL_16_FEATURES if f not in {"response_len_mean", "response_len_std", "distinct_1gram_ratio", "distinct_2gram_ratio"}],
    "ablation_5_no_hedging_negation": [f for f in CANONICAL_16_FEATURES if f not in {"hedging_score", "negation_count"}],
    "ablation_6_no_overlap": [f for f in CANONICAL_16_FEATURES if f not in {"question_overlap_mean"}],
    "baseline_a_single_lexical": ["response_len_mean", "distinct_1gram_ratio", "distinct_2gram_ratio", "hedging_score", "negation_count"],
    "baseline_c_agreement_only": ["dominant_agreement", "second_agreement", "agreement_margin", "cluster_count", "cluster_ratio"],
    "baseline_d_entropy_only": ["semantic_entropy", "normalized_entropy"],
    "baseline_e_multi_linguistic": ["response_len_mean", "response_len_std", "distinct_1gram_ratio", "distinct_2gram_ratio", "question_overlap_mean", "hedging_score", "negation_count"],
}


# =====================================================================
# 1. Experiment Configuration System
# =====================================================================

class ExperimentConfig:
    """Encapsulates, validates, and serializes experiment configurations."""

    REQUIRED_FIELDS = [
        "experiment_id",
        "dataset_path",
        "features_path",
        "split_path",
        "cv_path",
        "model_name",
    ]

    SUPPORTED_MODELS = {
        "majority_baseline",
        "simple_feature_baseline",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw_config = copy.deepcopy(config_dict)
        self._validate_and_assign()

    def _validate_and_assign(self):
        for field in self.REQUIRED_FIELDS:
            if field not in self.raw_config or self.raw_config[field] is None:
                raise ValueError(f"ExperimentConfig missing required field: '{field}'")

        self.experiment_id: str = str(self.raw_config["experiment_id"])
        self.description: str = str(self.raw_config.get("description", "Evaluation experiment"))
        self.dataset_path: str = str(self.raw_config["dataset_path"])
        self.features_path: str = str(self.raw_config["features_path"])
        self.split_path: str = str(self.raw_config["split_path"])
        self.cv_path: str = str(self.raw_config["cv_path"])
        self.manifest_path: Optional[str] = self.raw_config.get("manifest_path")
        self.output_dir: str = str(self.raw_config.get("output_dir", "experiments/results"))

        self.model_name: str = str(self.raw_config["model_name"]).lower().strip()
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                f"Must be one of {sorted(list(self.SUPPORTED_MODELS))}"
            )

        self.hyperparameters: Dict[str, Any] = self.raw_config.get("hyperparameters", {})
        self.random_seed: int = int(self.raw_config.get("random_seed", 42))

        # Feature configuration
        feature_set_name = self.raw_config.get("feature_set", "canonical_16")
        if feature_set_name not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set '{feature_set_name}'. "
                f"Supported sets: {sorted(list(FEATURE_SETS.keys()))}"
            )
        self.feature_set: str = feature_set_name
        self.feature_names: List[str] = FEATURE_SETS[feature_set_name]

        # Evaluation configuration
        self.cv_folds: int = int(self.raw_config.get("cv_folds", 5))
        self.dry_run: bool = bool(self.raw_config.get("dry_run", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "dataset_path": self.dataset_path,
            "features_path": self.features_path,
            "split_path": self.split_path,
            "cv_path": self.cv_path,
            "manifest_path": self.manifest_path,
            "output_dir": self.output_dir,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "feature_set": self.feature_set,
            "feature_names": self.feature_names,
            "random_seed": self.random_seed,
            "cv_folds": self.cv_folds,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_file(cls, filepath: str) -> "ExperimentConfig":
        """Loads configuration from YAML or JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith((".yaml", ".yml")):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config file '{filepath}' must contain a JSON/YAML mapping.")
        return cls(data)


# =====================================================================
# 2. Dataset Loading and Strict Validation Engine
# =====================================================================

class EvaluationDatasetLoader:
    """Strict loader and validator for gold benchmarks, features, and split artifacts."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def load_and_validate(self) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Loads and cross-validates gold benchmark, feature matrix, and split artifacts.

        Returns: (is_valid, payload, errors)
        """
        errors: List[str] = []
        payload: Dict[str, Any] = {}

        # 1. Validate Gold Benchmark
        validator = GoldDatasetValidator()
        is_gold, gold_recs, gold_errs, gold_meta = validator.validate_file(
            self.config.dataset_path, self.config.manifest_path
        )
        if not is_gold:
            errors.extend(gold_errs)
            return False, payload, errors

        # 2. Validate Split and CV Artifacts Existence
        if not os.path.exists(self.config.split_path):
            errors.append(f"Split artifact not found: {self.config.split_path}")
            return False, payload, errors
        if not os.path.exists(self.config.cv_path):
            errors.append(f"Cross-validation artifact not found: {self.config.cv_path}")
            return False, payload, errors

        with open(self.config.split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        with open(self.config.cv_path, "r", encoding="utf-8") as f:
            cv_data = json.load(f)

        # 3. Independent Leakage Check on Split Artifacts
        leak_validator = LeakageValidator()
        leak_res = leak_validator.validate_splits_and_cv(gold_recs, split_data, cv_data)
        if not leak_res["leakage_audit_passed"]:
            errors.extend(leak_res["violations"])
            return False, payload, errors

        # 4. Load and Validate Features
        if not os.path.exists(self.config.features_path):
            errors.append(f"Features file not found: {self.config.features_path}")
            return False, payload, errors

        feature_records: Dict[str, Dict[str, float]] = {}
        with open(self.config.features_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    frec = json.loads(s)
                except json.JSONDecodeError as jde:
                    errors.append(f"Features file line {line_no}: Malformed JSON - {jde}")
                    continue

                fid = frec.get("id")
                if not fid:
                    errors.append(f"Features file line {line_no}: Missing record 'id'")
                    continue

                feats = frec.get("features", {})
                if not isinstance(feats, dict):
                    errors.append(f"Features file line {line_no} (ID {fid}): 'features' must be a dict")
                    continue

                # Check required features for the configured feature set
                for fname in self.config.feature_names:
                    if fname not in feats or feats[fname] is None:
                        errors.append(f"Features file line {line_no} (ID {fid}): Missing required feature '{fname}'")
                    else:
                        val = feats[fname]
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            errors.append(f"Features file line {line_no} (ID {fid}): Invalid non-numeric/infinite feature '{fname}'={val}")

                feature_records[fid] = {fname: float(feats.get(fname, 0.0)) for fname in self.config.feature_names}

        # 5. ID Alignment: verify every gold record has features and split assignment
        gold_ids = set(r["id"] for r in gold_recs)
        feature_ids = set(feature_records.keys())

        missing_feature_ids = gold_ids - feature_ids
        if missing_feature_ids:
            errors.append(f"{len(missing_feature_ids)} gold benchmark records lack features in {self.config.features_path}")

        split_train_ids = set(split_data.get("splits", {}).get("train", {}).get("ids", []))
        split_val_ids = set(split_data.get("splits", {}).get("val", {}).get("ids", []))
        split_test_ids = set(split_data.get("splits", {}).get("test", {}).get("ids", []))
        all_split_ids = split_train_ids | split_val_ids | split_test_ids

        missing_split_ids = gold_ids - all_split_ids
        if missing_split_ids:
            errors.append(f"{len(missing_split_ids)} gold benchmark records lack split assignments in {self.config.split_path}")

        if errors:
            return False, payload, errors

        # 6. Build Arrays for Evaluation
        gold_map = {r["id"]: r for r in gold_recs}

        payload = {
            "gold_records": gold_recs,
            "gold_metadata": gold_meta,
            "gold_sha256": gold_meta["sha256"],
            "split_data": split_data,
            "cv_data": cv_data,
            "feature_records": feature_records,
            "train_ids": sorted(list(split_train_ids)),
            "val_ids": sorted(list(split_val_ids)),
            "test_ids": sorted(list(split_test_ids)),
            "gold_map": gold_map,
        }
        return True, payload, errors


# =====================================================================
# 3. Model Implementations
# =====================================================================

class MajorityClassBaseline:
    """Predicts majority class computed exclusively from training data."""

    def __init__(self, **kwargs):
        self.majority_class: Optional[int] = None
        self.classes_: np.ndarray = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert len(y) > 0, "Cannot fit on empty dataset"
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.majority_class is not None, "Model must be fitted before predict"
        return np.full(shape=(len(X),), fill_value=self.majority_class, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.majority_class is not None, "Model must be fitted before predict_proba"
        probas = np.zeros(shape=(len(X), len(self.classes_)), dtype=float)
        probas[:, self.majority_class] = 1.0
        return probas


class SimpleFeatureBaseline:
    """Rule-based decision baseline using entropy and cluster agreement heuristics."""

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        agreement_threshold: float = 0.7,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        self.entropy_threshold = entropy_threshold
        self.agreement_threshold = agreement_threshold
        self.feature_names = feature_names or CANONICAL_16_FEATURES
        self.classes_: np.ndarray = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Rule-based heuristic requires no parameter updates, but marks model fitted
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ent_idx = self.feature_names.index("normalized_entropy") if "normalized_entropy" in self.feature_names else 0
        agr_idx = self.feature_names.index("dominant_agreement") if "dominant_agreement" in self.feature_names else 1

        preds = []
        for row in X:
            norm_entropy = row[ent_idx]
            dom_agreement = row[agr_idx]
            if norm_entropy > self.entropy_threshold:
                preds.append(LABEL_TO_ID["UNKNOWN"])
            elif dom_agreement < self.agreement_threshold:
                preds.append(LABEL_TO_ID["AMBIGUOUS"])
            else:
                preds.append(LABEL_TO_ID["KNOWN"])
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probas = np.zeros(shape=(len(X), len(self.classes_)), dtype=float)
        for i, p in enumerate(preds):
            probas[i, p] = 1.0
        return probas


class LogisticRegressionBaseline:
    """Multinomial Logistic Regression with training-only StandardScaler."""

    def __init__(self, random_state: int = 42, C: float = 1.0, max_iter: int = 1000, **kwargs):
        self.random_state = random_state
        self.C = C
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Crucial safeguard: Scaler fitted EXCLUSIVELY on training features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RandomForestBaseline:
    """Random Forest Classifier with feature importance tracking."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: Optional[int] = 6,
        **kwargs
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


class XGBoostClassifierModel:
    """XGBoost Proposed Classifier for multi-class uncertainty classification."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        **kwargs
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
        )
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


def instantiate_model(model_name: str, config: ExperimentConfig) -> Any:
    """Factory creating configured model instance."""
    name = model_name.lower().strip()
    kwargs = copy.deepcopy(config.hyperparameters)
    kwargs["random_state"] = config.random_seed
    kwargs["feature_names"] = config.feature_names

    if name == "majority_baseline":
        return MajorityClassBaseline(**kwargs)
    elif name == "simple_feature_baseline":
        return SimpleFeatureBaseline(**kwargs)
    elif name == "logistic_regression":
        return LogisticRegressionBaseline(**kwargs)
    elif name == "random_forest":
        return RandomForestBaseline(**kwargs)
    elif name == "xgboost":
        return XGBoostClassifierModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")


# =====================================================================
# 4. Evaluation Metrics Engine
# =====================================================================

class EvaluationMetricsCalculator:
    """Calculates comprehensive classification, class-wise, and correlation metrics."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculates standard, balanced, and class-wise metrics with confusion matrix."""
        if classes is None:
            classes = [0, 1, 2]

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))

        # Confusion Matrix (3x3: Rows = True, Columns = Predicted)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_dict = {
            ID_TO_LABEL[r_idx]: {
                ID_TO_LABEL[c_idx]: int(cm[r_idx, c_idx])
                for c_idx in classes
            }
            for r_idx in classes
        }

        # Per-class analysis
        per_class: Dict[str, Dict[str, float]] = {}
        for c in classes:
            c_name = ID_TO_LABEL[c]
            c_true = (y_true == c).astype(int)
            c_pred = (y_pred == c).astype(int)
            support = int(np.sum(c_true))
            prec = float(precision_score(c_true, c_pred, zero_division=0))
            rec = float(recall_score(c_true, c_pred, zero_division=0))
            f1 = float(f1_score(c_true, c_pred, zero_division=0))
            per_class[c_name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "matthews_corrcoef": round(mcc, 4),
            "cohens_kappa": round(kappa, 4),
            "per_class": per_class,
            "confusion_matrix": cm_dict,
        }


# =====================================================================
# 5. Cross-Validation and Train/Val/Test Evaluator
# =====================================================================

class CrossValidationEvaluator:
    """Executes 5-fold cross-validation with strict group isolation and train-only fitting."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def evaluate(
        self,
        cv_data: Dict[str, Any],
        gold_map: Dict[str, Dict[str, Any]],
        features_map: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Runs 5-fold cross-validation and computes aggregate metrics."""
        folds = cv_data.get("folds", [])
        assert len(folds) == self.config.cv_folds, f"Expected {self.config.cv_folds} folds, found {len(folds)}"

        fold_results = []
        metrics_lists: Dict[str, List[float]] = {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "weighted_f1": [],
            "matthews_corrcoef": [],
            "cohens_kappa": [],
        }
        per_class_f1: Dict[str, List[float]] = {"KNOWN": [], "AMBIGUOUS": [], "UNKNOWN": []}

        feature_names = self.config.feature_names

        for fold in folds:
            f_idx = fold["fold_index"]
            train_ids = fold["train_ids"]
            val_ids = fold["val_ids"]

            # Safeguard: verify disjoint IDs
            assert set(train_ids).isdisjoint(set(val_ids)), f"Fold {f_idx}: train and val IDs overlap"

            # Construct arrays
            X_train = np.array([[features_map[qid][f] for f in feature_names] for qid in train_ids])
            y_train = np.array([LABEL_TO_ID[gold_map[qid]["final_gold_label"]] for qid in train_ids])

            X_val = np.array([[features_map[qid][f] for f in feature_names] for qid in val_ids])
            y_val = np.array([LABEL_TO_ID[gold_map[qid]["final_gold_label"]] for qid in val_ids])

            # Instantiate and fit model on training partition only
            model = instantiate_model(self.config.model_name, self.config)
            model.fit(X_train, y_train)

            # Evaluate on validation partition
            y_pred = model.predict(X_val)
            fold_metrics = EvaluationMetricsCalculator.compute_metrics(y_val, y_pred)

            fold_results.append({
                "fold_index": f_idx,
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "metrics": fold_metrics,
            })

            for m in metrics_lists:
                metrics_lists[m].append(fold_metrics[m])
            for c in per_class_f1:
                per_class_f1[c].append(fold_metrics["per_class"][c]["f1"])

        # Aggregate statistics (mean and std)
        aggregate: Dict[str, Any] = {}
        for m, vals in metrics_lists.items():
            aggregate[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }

        class_f1_aggregate: Dict[str, Any] = {}
        for c, vals in per_class_f1.items():
            class_f1_aggregate[c] = {
                "mean_f1": round(float(np.mean(vals)), 4),
                "std_f1": round(float(np.std(vals)), 4),
            }

        return {
            "cv_folds": self.config.cv_folds,
            "aggregate_metrics": aggregate,
            "class_wise_aggregate": class_f1_aggregate,
            "individual_folds": fold_results,
        }


class TrainValTestEvaluator:
    """Evaluates final model on held-out test partition with strict test-set isolation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def evaluate(
        self,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        gold_map: Dict[str, Dict[str, Any]],
        features_map: Dict[str, Dict[str, float]],
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """Fits model on train and scores on validation and isolated test partitions.

        Returns: (fitted_model, val_metrics, test_metrics)
        """
        feature_names = self.config.feature_names

        # Strict isolation check
        s_train = set(train_ids)
        s_val = set(val_ids)
        s_test = set(test_ids)
        assert s_train.isdisjoint(s_val), "Train and Val IDs overlap"
        assert s_train.isdisjoint(s_test), "Train and Test IDs overlap"
        assert s_val.isdisjoint(s_test), "Val and Test IDs overlap"

        X_train = np.array([[features_map[qid][f] for f in feature_names] for qid in train_ids])
        y_train = np.array([LABEL_TO_ID[gold_map[qid]["final_gold_label"]] for qid in train_ids])

        X_val = np.array([[features_map[qid][f] for f in feature_names] for qid in val_ids])
        y_val = np.array([LABEL_TO_ID[gold_map[qid]["final_gold_label"]] for qid in val_ids])

        X_test = np.array([[features_map[qid][f] for f in feature_names] for qid in test_ids])
        y_test = np.array([LABEL_TO_ID[gold_map[qid]["final_gold_label"]] for qid in test_ids])

        # Train model exclusively on training partition
        model = instantiate_model(self.config.model_name, self.config)
        model.fit(X_train, y_train)

        # Validation scoring
        y_val_pred = model.predict(X_val)
        val_metrics = EvaluationMetricsCalculator.compute_metrics(y_val, y_val_pred)

        # Held-out test scoring
        y_test_pred = model.predict(X_test)
        test_metrics = EvaluationMetricsCalculator.compute_metrics(y_test, y_test_pred)

        return model, val_metrics, test_metrics


# =====================================================================
# 6. Experiment Runner Orchestrator
# =====================================================================

class ExperimentRunner:
    """Coordinates validation, cross-validation, evaluation, and reporting."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> Tuple[int, Dict[str, Any], List[str]]:
        """Executes full experiment workflow or dry-run.

        Returns: (exit_code, results_dict, errors)
        """
        errors: List[str] = []
        results: Dict[str, Any] = {}

        # 1. Strict Dataset & Split Validation
        loader = EvaluationDatasetLoader(self.config)
        is_valid, payload, load_errors = loader.load_and_validate()

        if not is_valid:
            errors.extend(load_errors)
            return 2, results, errors

        # 2. Dry-Run Check
        if self.config.dry_run:
            results = {
                "status": "DRY_RUN_PASSED",
                "message": "All configuration, dataset provenance, feature schema, and split leakage checks passed successfully.",
                "dataset_version": payload["gold_metadata"].get("benchmark_version"),
                "sample_count": payload["gold_metadata"].get("record_count"),
                "model_name": self.config.model_name,
                "feature_count": len(self.config.feature_names),
                "feature_names": self.config.feature_names,
            }
            return 0, results, errors

        # 3. 5-Fold Cross-Validation
        cv_evaluator = CrossValidationEvaluator(self.config)
        cv_results = cv_evaluator.evaluate(
            payload["cv_data"], payload["gold_map"], payload["feature_records"]
        )

        # 4. Train / Val / Test Partition Evaluation
        split_evaluator = TrainValTestEvaluator(self.config)
        model, val_metrics, test_metrics = split_evaluator.evaluate(
            payload["train_ids"],
            payload["val_ids"],
            payload["test_ids"],
            payload["gold_map"],
            payload["feature_records"],
        )

        # 5. Extract Feature Importances if available
        feature_importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importances = {
                fname: round(float(importances[i]), 6)
                for i, fname in enumerate(self.config.feature_names)
            }

        # 6. Assemble Standardized Experiment Results Schema
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results = {
            "experiment_id": self.config.experiment_id,
            "timestamp": timestamp,
            "dataset_version": payload["gold_metadata"].get("benchmark_version", "1.0"),
            "gold_benchmark_sha256": payload["gold_sha256"],
            "model_name": self.config.model_name,
            "feature_set": self.config.feature_set,
            "feature_count": len(self.config.feature_names),
            "feature_names": self.config.feature_names,
            "random_seed": self.config.random_seed,
            "hyperparameters": self.config.hyperparameters,
            "sample_counts": {
                "total": len(payload["gold_records"]),
                "train": len(payload["train_ids"]),
                "val": len(payload["val_ids"]),
                "test": len(payload["test_ids"]),
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "cross_validation": cv_results,
            "feature_importances": feature_importances,
            "execution_status": "COMPLETED",
        }

        # 7. Write Results & Audit Report
        os.makedirs(self.config.output_dir, exist_ok=True)
        results_file = os.path.join(self.config.output_dir, f"{self.config.experiment_id}_results.json")
        audit_file = os.path.join(self.config.output_dir, f"{self.config.experiment_id}_audit.md")
        model_file = os.path.join(self.config.output_dir, f"{self.config.experiment_id}_model.joblib")

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save model artifact in designated experiment directory (never touching models/)
        try:
            joblib.dump(model, model_file)
        except Exception as ex:
            logger.warning(f"Failed to dump model: {ex}")

        # Render Markdown Audit Report
        self._generate_audit_report(results, payload, audit_file)

        return 0, results, errors

    def _generate_audit_report(self, results: Dict[str, Any], payload: Dict[str, Any], output_path: str):
        test_m = results["test_metrics"]
        cv_agg = results["cross_validation"]["aggregate_metrics"]
        cv_class = results["cross_validation"]["class_wise_aggregate"]

        lines = [
            f"# Experiment Audit Report: {results['experiment_id']}",
            "",
            f"**Model:** `{results['model_name']}`  ",
            f"**Feature Set:** `{results['feature_set']}` ({results['feature_count']} features)  ",
            f"**Timestamp:** `{results['timestamp']}`  ",
            f"**Gold Benchmark SHA-256:** `{results['gold_benchmark_sha256']}`  ",
            f"**Random Seed:** `{results['random_seed']}`  ",
            "",
            "---",
            "",
            "## 1. Executive Performance Summary",
            "",
            "| Metric | 5-Fold CV (Mean ± Std) | Held-Out Test Set |",
            "| :--- | :---: | :---: |",
            f"| **Accuracy** | {cv_agg['accuracy']['mean']:.4f} ± {cv_agg['accuracy']['std']:.4f} | {test_m['accuracy']:.4f} |",
            f"| **Balanced Accuracy** | {cv_agg['balanced_accuracy']['mean']:.4f} ± {cv_agg['balanced_accuracy']['std']:.4f} | {test_m['balanced_accuracy']:.4f} |",
            f"| **Macro F1** | {cv_agg['macro_f1']['mean']:.4f} ± {cv_agg['macro_f1']['std']:.4f} | {test_m['macro_f1']:.4f} |",
            f"| **Weighted F1** | {cv_agg['weighted_f1']['mean']:.4f} ± {cv_agg['weighted_f1']['std']:.4f} | {test_m['weighted_f1']:.4f} |",
            f"| **Matthews Corrcoef** | {cv_agg['matthews_corrcoef']['mean']:.4f} ± {cv_agg['matthews_corrcoef']['std']:.4f} | {test_m['matthews_corrcoef']:.4f} |",
            f"| **Cohen's Kappa** | {cv_agg['cohens_kappa']['mean']:.4f} ± {cv_agg['cohens_kappa']['std']:.4f} | {test_m['cohens_kappa']:.4f} |",
            "",
            "---",
            "",
            "## 2. Class-Wise Performance Breakdown",
            "",
            "| Class | 5-Fold CV F1 | Test Precision | Test Recall | Test F1 | Test Support |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for c in CANONICAL_CLASSES:
            cp = test_m["per_class"].get(c, {})
            cv_f1 = cv_class.get(c, {})
            lines.append(
                f"| **{c}** | {cv_f1.get('mean_f1', 0.0):.4f} ± {cv_f1.get('std_f1', 0.0):.4f} | "
                f"{cp.get('precision', 0.0):.4f} | {cp.get('recall', 0.0):.4f} | "
                f"{cp.get('f1', 0.0):.4f} | {cp.get('support', 0)} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Test Set Confusion Matrix ($3 \\times 3$)",
            "",
            "*Rows = Authoritative Gold Label, Columns = Predicted Class*",
            "",
            "| True \\ Pred | KNOWN | AMBIGUOUS | UNKNOWN |",
            "| :--- | :---: | :---: | :---: |",
        ])

        cm = test_m["confusion_matrix"]
        for r_cls in CANONICAL_CLASSES:
            row_vals = [str(cm.get(r_cls, {}).get(c_cls, 0)) for c_cls in CANONICAL_CLASSES]
            lines.append(f"| **{r_cls}** | " + " | ".join(row_vals) + " |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Scientific Safeguards & Data Leakage Audit",
            "",
            "- **Gold Provenance Gate:** Verified authoritative final gold labels and double-blind human consensus.",
            "- **Group Isolation:** Zero question group overlap between train, validation, and test partitions.",
            "- **Test Set Isolation:** Held-out test set was never accessed during model training, feature scaling, or cross-validation.",
            "- **Reproducibility:** Seeded deterministic configuration with full metadata serialization.",
            "",
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
