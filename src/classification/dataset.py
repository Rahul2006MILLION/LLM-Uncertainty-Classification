"""Dataset loading, validation, and leakage prevention for uncertainty classification.

Provides rigorous validation:
- Documented label definitions: KNOWN, AMBIGUOUS, UNKNOWN.
- Split generation with strict isolation: Train / Validation / Test.
- No-leakage verification: checks for question overlap across splits and feature leakage.
- Class distribution reporting.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}

LABEL_DEFINITIONS = {
    "KNOWN": (
        "The question has a sufficiently clear interpretation and the model's answer "
        "is supported by reliable evidence. High semantic agreement and low entropy."
    ),
    "AMBIGUOUS": (
        "The question has multiple materially different plausible interpretations "
        "(e.g., polysemy, distinct entities sharing a name). Evidenced by distinct semantic clusters."
    ),
    "UNKNOWN": (
        "The question is sufficiently clear, but the system lacks reliable information "
        "to answer it (e.g., future events, private facts, unanswerable queries). "
        "Note: A wrong answer alone must NOT automatically be classified as UNKNOWN."
    ),
}

FEATURE_COLUMNS = [
    "num_clusters",
    "majority_cluster_agreement",
    "second_cluster_agreement",
    "agreement_margin",
    "raw_semantic_entropy",
    "normalized_semantic_entropy",
    "mean_pairwise_similarity",
    "min_pairwise_similarity",
    "mean_length_chars",
    "mean_length_words",
    "length_variance_words",
    "lexical_diversity",
    "question_word_count",
    "question_has_disjunction",
    "question_has_which",
    "question_has_what",
    "question_is_short",
]


@dataclass
class DatasetExample:
    question: str
    label: str
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class DatasetValidationError(Exception):
    """Raised when dataset validation or leakage checks fail."""
    pass


def validate_examples(examples: List[DatasetExample]) -> None:
    """Validate labels and check for basic integrity.

    Raises:
        DatasetValidationError: If invalid labels or malformed features are detected.
    """
    if not examples:
        raise DatasetValidationError("Dataset is empty.")

    for i, ex in enumerate(examples):
        if not ex.question or not ex.question.strip():
            raise DatasetValidationError(f"Example index {i} has an empty question.")

        if ex.label not in VALID_LABELS:
            raise DatasetValidationError(
                f"Example index {i} has invalid label '{ex.label}'. "
                f"Allowed labels: {sorted(list(VALID_LABELS))}"
            )

        # Check for target leakage in features
        for feat_name, feat_val in ex.features.items():
            feat_lower = feat_name.lower()
            if any(leak_word in feat_lower for leak_word in ["label", "target", "ground_truth", "gt_"]):
                raise DatasetValidationError(
                    f"Target label leakage detected in feature '{feat_name}' for example {i}."
                )
            if not isinstance(feat_val, (int, float)) or np.isnan(feat_val):
                raise DatasetValidationError(
                    f"Feature '{feat_name}' has non-numeric/NaN value '{feat_val}' in example {i}."
                )


def check_split_leakage(
    train_examples: List[DatasetExample],
    val_examples: List[DatasetExample],
    test_examples: List[DatasetExample],
) -> None:
    """Ensure zero question overlap between train, validation, and test splits."""
    train_questions: Set[str] = {ex.question.strip().lower() for ex in train_examples}
    val_questions: Set[str] = {ex.question.strip().lower() for ex in val_examples}
    test_questions: Set[str] = {ex.question.strip().lower() for ex in test_examples}

    train_val_overlap = train_questions.intersection(val_questions)
    if train_val_overlap:
        raise DatasetValidationError(
            f"Data leakage detected: {len(train_val_overlap)} questions overlap between train and val splits."
        )

    train_test_overlap = train_questions.intersection(test_questions)
    if train_test_overlap:
        raise DatasetValidationError(
            f"Data leakage detected: {len(train_test_overlap)} questions overlap between train and test splits."
        )

    val_test_overlap = val_questions.intersection(test_questions)
    if val_test_overlap:
        raise DatasetValidationError(
            f"Data leakage detected: {len(val_test_overlap)} questions overlap between val and test splits."
        )


def compute_class_distribution(examples: List[DatasetExample]) -> Dict[str, Dict[str, Any]]:
    """Compute counts and proportions for each class."""
    total = len(examples)
    counts = {lbl: 0 for lbl in sorted(VALID_LABELS)}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1

    return {
        lbl: {
            "count": count,
            "percentage": (count / total * 100.0) if total > 0 else 0.0,
        }
        for lbl, count in counts.items()
    }


def split_dataset(
    examples: List[DatasetExample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[DatasetExample], List[DatasetExample], List[DatasetExample]]:
    """Split dataset into train, validation, and test splits with strict isolation.

    Uses grouping by normalized question text to guarantee that identical/paraphrased questions
    never cross split boundaries.
    """
    validate_examples(examples)

    # Group by question to prevent leakage
    question_map: Dict[str, List[DatasetExample]] = {}
    for ex in examples:
        q_key = ex.question.strip().lower()
        question_map.setdefault(q_key, []).append(ex)

    rng = np.random.RandomState(random_seed)
    unique_questions = list(question_map.keys())
    rng.shuffle(unique_questions)

    n_total = len(unique_questions)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio))) if (n_total - n_train) > 1 else 0

    train_keys = unique_questions[:n_train]
    val_keys = unique_questions[n_train : n_train + n_val]
    test_keys = unique_questions[n_train + n_val :]

    train_set = [ex for q in train_keys for ex in question_map[q]]
    val_set = [ex for q in val_keys for ex in question_map[q]]
    test_set = [ex for q in test_keys for ex in question_map[q]]

    # Rigorous check
    if val_set and test_set:
        check_split_leakage(train_set, val_set, test_set)

    return train_set, val_set, test_set


def load_dataset_from_json(path: Path) -> List[DatasetExample]:
    """Load and validate dataset examples from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        ex = DatasetExample(
            question=item["question"],
            label=item["label"],
            features=item.get("features", {}),
            metadata=item.get("metadata", {}),
        )
        examples.append(ex)

    validate_examples(examples)
    return examples


def save_dataset_to_json(examples: List[DatasetExample], path: Path) -> None:
    """Save dataset examples to a JSON file."""
    data = [
        {
            "question": ex.question,
            "label": ex.label,
            "features": ex.features,
            "metadata": ex.metadata or {},
        }
        for ex in examples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
