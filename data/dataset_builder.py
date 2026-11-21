"""Utilities to load, filter, and construct scientifically grounded benchmark datasets."""

import json
import logging
from typing import List, Dict, Any, Optional
from data.schema import QuestionRecord
from src.evaluation.data_validator import DatasetValidator

logger = logging.getLogger(__name__)


# Curated, defensible UNKNOWN questions with clear non-epistemic/unanswerability rationales
CURATED_UNKNOWN_QUESTIONS: List[Dict[str, Any]] = [
    # 1. Private Cognition & Inaccessible Local State (5 items)
    {
        "id": "unk_001",
        "question": "What number am I thinking of right now?",
        "label": "UNKNOWN",
        "domain": "private_cognition",
        "subcategory": "private_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private mental state inaccessible to any model or external observer.",
    },
    {
        "id": "unk_002",
        "question": "What color shirt is the user wearing today?",
        "label": "UNKNOWN",
        "domain": "private_information",
        "subcategory": "private_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Real-time user physical state not transmitted or observed in the prompt.",
    },
    {
        "id": "unk_003",
        "question": "What is the secret passphrase written on the note inside my locked safe?",
        "label": "UNKNOWN",
        "domain": "private_information",
        "subcategory": "private_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private encrypted/secret information absent from any training corpus.",
    },
    {
        "id": "unk_004",
        "question": "What did John Smith of London eat for breakfast on March 14, 1842?",
        "label": "UNKNOWN",
        "domain": "unrecorded_history",
        "subcategory": "unrecorded_history",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Private historical event for which no documentation exists.",
    },
    {
        "id": "unk_005",
        "question": "What is the exact current weight of my pet cat?",
        "label": "UNKNOWN",
        "domain": "private_information",
        "subcategory": "private_state",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Unstated personal physical measurement inaccessible to the system.",
    },

    # 2. Indeterminate Future Contingencies (5 items)
    {
        "id": "unk_006",
        "question": "What will be the exact closing price of Apple stock on December 31, 2035?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "subcategory": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future economic contingency fundamentally undetermined at present time.",
    },
    {
        "id": "unk_007",
        "question": "What will be the exact temperature in Tokyo at noon on January 1, 2100?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "subcategory": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future weather conditions beyond deterministic predictability.",
    },
    {
        "id": "unk_008",
        "question": "Who will win the FIFA Men's World Cup in 2062?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "subcategory": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future athletic outcome decades in the future.",
    },
    {
        "id": "unk_009",
        "question": "What will be the exact population of Paris on January 1, 2200?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "subcategory": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Demographic projection beyond knowable empirical certainty.",
    },
    {
        "id": "unk_010",
        "question": "What will the headline of the New York Times be on November 15, 2048?",
        "label": "UNKNOWN",
        "domain": "future_contingency",
        "subcategory": "future_contingency",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Future editorial decision for an unpublished issue.",
    },

    # 3. False Premises & Impossible Assumptions (5 items)
    {
        "id": "unk_011",
        "question": "Which organization organizes the FIFA Ice Hockey World Cup?",
        "label": "UNKNOWN",
        "domain": "false_premise",
        "subcategory": "false_premise",
        "source": "curated_false_premise",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "False premise: FIFA does not organize ice hockey; IIHF organizes ice hockey.",
    },
    {
        "id": "unk_012",
        "question": "When did Thomas Jefferson sign the United States Constitution?",
        "label": "UNKNOWN",
        "domain": "false_premise",
        "subcategory": "false_premise",
        "source": "curated_false_premise",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "False premise: Jefferson never signed the US Constitution (he was US Minister to France at the time).",
    },
    {
        "id": "unk_013",
        "question": "What was the name of Napoleon Bonaparte's nuclear submarine?",
        "label": "UNKNOWN",
        "domain": "false_premise",
        "subcategory": "false_premise",
        "source": "curated_false_premise",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "False premise: Nuclear submarines did not exist during Napoleon's era.",
    },
    {
        "id": "unk_014",
        "question": "In what year did Abraham Lincoln announce the release of the first Apple iPhone?",
        "label": "UNKNOWN",
        "domain": "false_premise",
        "subcategory": "false_premise",
        "source": "curated_false_premise",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "False premise: Anachronistic confusion of 19th-century president and 21st-century phone.",
    },
    {
        "id": "unk_015",
        "question": "Which country has won the most Olympic gold medals in underwater basket weaving?",
        "label": "UNKNOWN",
        "domain": "false_premise",
        "subcategory": "false_premise",
        "source": "curated_false_premise",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "False premise: Underwater basket weaving is not an Olympic sport.",
    },

    # 4. Immeasurable Physical States & Unrecorded History (5 items)
    {
        "id": "unk_016",
        "question": "What is the exact mass of the pebble located at 45.1234 N, 12.5678 E on Mars?",
        "label": "UNKNOWN",
        "domain": "unmeasured_physical_state",
        "subcategory": "immeasurable_physical",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Specific physical measurement that has never been observed or documented.",
    },
    {
        "id": "unk_017",
        "question": "How many grains of sand are currently on the beach at Copacabana?",
        "label": "UNKNOWN",
        "domain": "unmeasured_physical_state",
        "subcategory": "immeasurable_physical",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Immeasurable real-time physical quantity lacking empirical data.",
    },
    {
        "id": "unk_018",
        "question": "What exact words did Socrates whisper as he took his final breath?",
        "label": "UNKNOWN",
        "domain": "unrecorded_history",
        "subcategory": "unrecorded_history",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Historical fact not recorded in any extant historical record or evidence.",
    },
    {
        "id": "unk_019",
        "question": "How many leaves were on the tallest oak tree in Sherwood Forest on July 4, 1776?",
        "label": "UNKNOWN",
        "domain": "unrecorded_history",
        "subcategory": "unrecorded_history",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Transient historical physical quantity never measured.",
    },
    {
        "id": "unk_020",
        "question": "How many raindrops fell on Central Park during the storm on May 3, 1928?",
        "label": "UNKNOWN",
        "domain": "unmeasured_physical_state",
        "subcategory": "immeasurable_physical",
        "source": "curated_unanswerable",
        "license": "CC0",
        "annotation_type": "human_verified",
        "annotation_rationale": "Immeasurable aggregate meteorological quantity.",
    },
]

# Curated High-Certainty Definitional & Arithmetic Questions (KNOWN)
CURATED_KNOWN_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "known_def_001",
        "question": "What is 2 + 2?",
        "label": "KNOWN",
        "domain": "mathematics",
        "subcategory": "concept_definition",
        "source": "elementary_arithmetic",
        "license": "CC0",
        "ground_truth_answers": ["4", "four"],
        "annotation_type": "gold_benchmark",
        "annotation_rationale": "Elementary arithmetic identity with single objective answer.",
    },
    {
        "id": "known_def_002",
        "question": "What is Data Science?",
        "label": "KNOWN",
        "domain": "computer_science",
        "subcategory": "concept_definition",
        "source": "academic_taxonomy",
        "license": "CC0",
        "ground_truth_answers": ["An interdisciplinary field combining statistics, computation, and domain analysis."],
        "annotation_type": "gold_benchmark",
        "annotation_rationale": "Standard established academic discipline with clear consensus definition.",
    },
    {
        "id": "known_def_003",
        "question": "What is the speed of light in a vacuum in meters per second?",
        "label": "KNOWN",
        "domain": "physics",
        "subcategory": "concept_definition",
        "source": "si_standard",
        "license": "CC0",
        "ground_truth_answers": ["299,792,458 m/s", "299792458"],
        "annotation_type": "gold_benchmark",
        "annotation_rationale": "Exact defined physical constant in the International System of Units.",
    },
    {
        "id": "known_def_004",
        "question": "What is the chemical formula for water?",
        "label": "KNOWN",
        "domain": "chemistry",
        "subcategory": "concept_definition",
        "source": "iupac",
        "license": "CC0",
        "ground_truth_answers": ["H2O", "H₂O"],
        "annotation_type": "gold_benchmark",
        "annotation_rationale": "Universal chemical identity of water.",
    },
]

# Curated Entity Polysemy & Criterion Ambiguity (AMBIGUOUS)
CURATED_AMBIGUOUS_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "ambig_cur_001",
        "question": "Who is Jordan?",
        "label": "AMBIGUOUS",
        "domain": "entity_polysemy",
        "subcategory": "entity_polysemy",
        "source": "curated_polysemy",
        "license": "CC0",
        "ground_truth_answers": ["Michael Jordan", "Jordan (country)", "Michael B. Jordan", "Jordan River"],
        "interpretations": ["Michael Jordan (basketball player)", "Jordan (Middle Eastern nation)", "Michael B. Jordan (actor)", "Jordan River (geographic feature)"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Polysemic single name with multiple prominent real-world referents.",
    },
    {
        "id": "ambig_cur_002",
        "question": "What is the capital of Georgia?",
        "label": "AMBIGUOUS",
        "domain": "geography",
        "subcategory": "entity_polysemy",
        "source": "curated_polysemy",
        "license": "CC0",
        "ground_truth_answers": ["Tbilisi", "Atlanta"],
        "interpretations": ["What is the capital of Georgia (the sovereign country in the Caucasus)?", "What is the capital of Georgia (the US state)?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Geographic polysemy between sovereign nation (Tbilisi) and US state (Atlanta).",
    },
    {
        "id": "ambig_cur_003",
        "question": "What is the capital of Washington?",
        "label": "AMBIGUOUS",
        "domain": "geography",
        "subcategory": "entity_polysemy",
        "source": "curated_polysemy",
        "license": "CC0",
        "ground_truth_answers": ["Olympia", "Washington, D.C."],
        "interpretations": ["What is the capital of Washington State?", "What is the capital of the United States (Washington, D.C.)?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Geographic polysemy between Washington State (Olympia) and the national capital district.",
    },
    {
        "id": "ambig_cur_004",
        "question": "Who played Spider-Man in the live-action films?",
        "label": "AMBIGUOUS",
        "domain": "pop_culture",
        "subcategory": "entity_polysemy",
        "source": "curated_polysemy",
        "license": "CC0",
        "ground_truth_answers": ["Tobey Maguire", "Andrew Garfield", "Tom Holland"],
        "interpretations": ["Who played Spider-Man in the Sam Raimi trilogy?", "Who played Spider-Man in the Amazing Spider-Man duology?", "Who played Spider-Man in the Marvel Cinematic Universe?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Role portrayed by three distinct lead actors across separate live-action film series.",
    },
    {
        "id": "ambig_cur_005",
        "question": "What is the largest city in the world?",
        "label": "AMBIGUOUS",
        "domain": "geography",
        "subcategory": "criterion_ambiguity",
        "source": "curated_criterion_ambiguity",
        "license": "CC0",
        "ground_truth_answers": ["Tokyo (by metropolitan population)", "Shanghai (by city proper population)", "New York (by urban area)", "Chongqing (by administrative area)"],
        "interpretations": ["What is the largest city in the world by metropolitan population?", "What is the largest city by city proper population?", "What is the largest city by urban land area?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Criterion ambiguity: Answer depends on whether size is measured by metropolitan population, city-proper population, or land area.",
    },
    {
        "id": "ambig_cur_006",
        "question": "What is the fastest animal on earth?",
        "label": "AMBIGUOUS",
        "domain": "biology",
        "subcategory": "criterion_ambiguity",
        "source": "curated_criterion_ambiguity",
        "license": "CC0",
        "ground_truth_answers": ["Peregrine falcon (in a dive)", "Cheetah (land sprint)", "Black marlin (swimming)"],
        "interpretations": ["What is the fastest animal on earth during a diving flight?", "What is the fastest land animal running?", "What is the fastest swimming marine animal?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Criterion ambiguity: depends on whether flight dive, land locomotion, or swimming speed is measured.",
    },
    {
        "id": "ambig_cur_007",
        "question": "What is the oldest university in the world?",
        "label": "AMBIGUOUS",
        "domain": "history",
        "subcategory": "criterion_ambiguity",
        "source": "curated_criterion_ambiguity",
        "license": "CC0",
        "ground_truth_answers": ["University of al-Qarawiyyin", "University of Bologna", "Nalanda"],
        "interpretations": ["What is the oldest continuously operating educational institution?", "What is the oldest institution founded specifically as a medieval university? (Bologna)"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Criterion ambiguity over definition of university versus madrasa / ancient academy.",
    },
    {
        "id": "ambig_cur_008",
        "question": "What is the best-selling book of all time?",
        "label": "AMBIGUOUS",
        "domain": "literature",
        "subcategory": "criterion_ambiguity",
        "source": "curated_criterion_ambiguity",
        "license": "CC0",
        "ground_truth_answers": ["The Bible", "Don Quixote", "A Tale of Two Cities"],
        "interpretations": ["What is the best-selling book including religious texts?", "What is the best-selling single-volume work of fiction?"],
        "annotation_type": "human_verified",
        "annotation_rationale": "Criterion ambiguity: depends on whether religious / historical non-copyrighted texts are counted.",
    },
]


def extract_from_ambignq(
    filepath: str,
    max_known: int = 16,
    max_ambiguous: int = 12,
    existing_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Extract verified KNOWN (singleAnswer) and AMBIGUOUS (multipleQAs) records from AmbigNQ."""
    records: List[Dict[str, Any]] = []
    known_count = 0
    ambig_count = 0
    seen = existing_ids or set()

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
            out_id = f"ambig_{qid}"
            if out_id in seen:
                continue
            seen.add(out_id)
            qa_pairs = annotations[0].get("qaPairs", [])
            interpretations = [p.get("question") for p in qa_pairs if p.get("question")]
            answers = []
            for p in qa_pairs:
                answers.extend(p.get("answer", []))

            records.append({
                "id": out_id,
                "question": qtext,
                "label": "AMBIGUOUS",
                "domain": "general",
                "subcategory": "temporal_contextual",
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
            out_id = f"known_{qid}"
            if out_id in seen:
                continue
            seen.add(out_id)
            answers = annotations[0].get("answer", [])
            records.append({
                "id": out_id,
                "question": qtext,
                "label": "KNOWN",
                "domain": "general",
                "subcategory": "factual",
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


def build_evaluation_benchmark(
    train_ambignq_path: str = "ambignq_light/train_light.json",
    dev_ambignq_path: str = "ambignq_light/dev_light.json",
    output_path: str = "data/benchmarks/uncertainty_benchmark_v2.jsonl",
    target_per_class: int = 20,
) -> List[QuestionRecord]:
    """Build and save a balanced, diverse 60-question benchmark (20 KNOWN, 20 AMBIGUOUS, 20 UNKNOWN)."""
    # 1. KNOWN: 4 curated definitions + 8 from train_light + 8 from dev_light = 20
    known_from_train = extract_from_ambignq(train_ambignq_path, max_known=8, max_ambiguous=0)
    seen_ids = {r["id"] for r in known_from_train}
    known_from_dev = extract_from_ambignq(dev_ambignq_path, max_known=8, max_ambiguous=0, existing_ids=seen_ids)
    known_pool = CURATED_KNOWN_DEFINITIONS + known_from_train + known_from_dev
    known_subset = known_pool[:target_per_class]

    # 2. AMBIGUOUS: 8 curated entity/criterion + 6 from train_light + 6 from dev_light = 20
    ambig_from_train = extract_from_ambignq(train_ambignq_path, max_known=0, max_ambiguous=6)
    seen_ambig_ids = {r["id"] for r in ambig_from_train}
    ambig_from_dev = extract_from_ambignq(dev_ambignq_path, max_known=0, max_ambiguous=6, existing_ids=seen_ambig_ids)
    ambig_pool = CURATED_AMBIGUOUS_QUESTIONS + ambig_from_train + ambig_from_dev
    ambig_subset = ambig_pool[:target_per_class]

    # 3. UNKNOWN: exactly 20 curated unanswerable questions across 4 subcategories
    unknown_subset = CURATED_UNKNOWN_QUESTIONS[:target_per_class]

    all_raw = known_subset + ambig_subset + unknown_subset

    # Validate against Pydantic QuestionRecord
    validated: List[QuestionRecord] = []
    for r in all_raw:
        validated.append(QuestionRecord(**r))

    summary = DatasetValidator.validate_records([r.model_dump() for r in validated])
    logger.info("Evaluation benchmark validation summary: %s", summary)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in validated:
            f.write(json.dumps(rec.model_dump()) + "\n")

    logger.info("Saved %d validated records to %s", len(validated), output_path)
    return validated


# Backward compatibility alias
build_pilot_benchmark = build_evaluation_benchmark
