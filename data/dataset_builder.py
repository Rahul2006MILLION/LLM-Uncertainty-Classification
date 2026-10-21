"""Utilities to load, filter, and construct benchmark datasets from local sources."""

import json
import logging
from typing import List, Dict, Any, Optional
from data.schema import QuestionRecord
from src.evaluation.data_validator import DatasetValidator

logger = logging.getLogger(__name__)


# Curated, defensible UNKNOWN questions with clear non-epistemic/unanswerability rationales
CURATED_UNKNOWN_QUESTIONS = [
    {
        "id": "unk_001",
        "question": "What number am I thinking of right now?",
        "label": "UNKNOWN",
        "domain": "private_cognition",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private mental state inaccessible to any model or external observer.",
    },
    {
        "id": "unk_002",
        "question": "What will be the exact closing price of Apple stock on December 31, 2035?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future economic contingency fundamentally undetermined at present time.",
    },
    {
        "id": "unk_003",
        "question": "What exact words did Socrates whisper as he took his final breath?",
        "label": "UNKNOWN",
        "domain": "unrecorded_history",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Historical fact not recorded in any extant historical record or evidence.",
    },
    {
        "id": "unk_004",
        "question": "What is the exact mass of the pebble located at 45.1234 N, 12.5678 E on Mars?",
        "label": "UNKNOWN",
        "domain": "unmeasured_physical_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Specific physical measurement that has never been observed or documented.",
    },
    {
        "id": "unk_005",
        "question": "What did John Smith of London eat for breakfast on March 14, 1842?",
        "label": "UNKNOWN",
        "domain": "unrecorded_history",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private historical event for which no documentation exists.",
    },
    {
        "id": "unk_006",
        "question": "What will be the exact temperature in Tokyo at noon on January 1, 2100?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future weather conditions beyond deterministic predictability.",
    },
    {
        "id": "unk_007",
        "question": "What is the secret passphrase written on the note inside my locked safe?",
        "label": "UNKNOWN",
        "domain": "private_information",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private inaccessible information not present in any training corpus.",
    },
    {
        "id": "unk_008",
        "question": "How many grains of sand are currently on the beach at Copacabana?",
        "label": "UNKNOWN",
        "domain": "unmeasured_physical_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Immeasurable real-time physical quantity lacking empirical data.",
    },
]


def extract_from_ambignq(
    filepath: str,
    max_known: int = 15,
    max_ambiguous: int = 15,
) -> List[Dict[str, Any]]:
    """Extract verified KNOWN (singleAnswer) and AMBIGUOUS (multipleQAs) records from AmbigNQ."""
    records: List[Dict[str, Any]] = []
    known_count = 0
    ambig_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        qid = str(item.get("id"))
        qtext = item.get("question", "").strip()
        if not qtext:
            continue

        annotations = item.get("annotations", [])
        if not annotations:
            continue

        ann_type = annotations[0].get("type")

        # AMBIGUOUS: Annotated with multipleQAs indicating multiple distinct interpretations
        if ann_type == "multipleQAs" and ambig_count < max_ambiguous:
            qa_pairs = annotations[0].get("qaPairs", [])
            interpretations = [p.get("question") for p in qa_pairs if p.get("question")]
            answers = []
            for p in qa_pairs:
                answers.extend(p.get("answer", []))

            records.append({
                "id": f"ambig_{qid}",
                "question": qtext,
                "label": "AMBIGUOUS",
                "domain": "general",
                "source": "ambignq_light",
                "license": "CC BY-SA 3.0",
                "ground_truth_answers": answers,
                "interpretations": interpretations,
                "annotation_type": "gold_benchmark",
                "annotation_rationale": "AmbigNQ gold benchmark annotation with multiple distinct QA pairs.",
            })
            ambig_count += 1

        # KNOWN: Annotated with singleAnswer indicating a single clear factual answer
        elif ann_type == "singleAnswer" and known_count < max_known:
            answers = annotations[0].get("answer", [])
            records.append({
                "id": f"known_{qid}",
                "question": qtext,
                "label": "KNOWN",
                "domain": "general",
                "source": "ambignq_light",
                "license": "CC BY-SA 3.0",
                "ground_truth_answers": answers,
                "interpretations": None,
                "annotation_type": "gold_benchmark",
                "annotation_rationale": "AmbigNQ gold benchmark annotation with a single unambiguous factual answer.",
            })
            known_count += 1

        if known_count >= max_known and ambig_count >= max_ambiguous:
            break

    return records


def build_pilot_benchmark(
    ambignq_path: str = "ambignq_light/train_light.json",
    output_path: str = "data/benchmarks/pilot_uncertainty_benchmark.jsonl",
    n_per_class: int = 8,
) -> List[QuestionRecord]:
    """Build and save a balanced, validated pilot benchmark covering KNOWN, AMBIGUOUS, and UNKNOWN."""
    ambig_records = extract_from_ambignq(ambignq_path, max_known=n_per_class, max_ambiguous=n_per_class)

    known_subset = [r for r in ambig_records if r["label"] == "KNOWN"][:n_per_class]
    ambig_subset = [r for r in ambig_records if r["label"] == "AMBIGUOUS"][:n_per_class]
    unknown_subset = CURATED_UNKNOWN_QUESTIONS[:n_per_class]

    # Combine all subsets
    all_raw = known_subset + ambig_subset + unknown_subset

    # Validate against schema
    validated: List[QuestionRecord] = []
    for r in all_raw:
        validated.append(QuestionRecord(**r))

    # Validate integrity
    summary = DatasetValidator.validate_records([r.model_dump() for r in validated])
    logger.info("Pilot benchmark validation summary: %s", summary)

    # Save to jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in validated:
            f.write(json.dumps(rec.model_dump()) + "\n")

    return validated
