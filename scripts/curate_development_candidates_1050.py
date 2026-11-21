"""Curation and candidate construction script for the 1,050-Question Development Dataset.

Enforces:
1. Exact class quotas: 350 KNOWN, 350 AMBIGUOUS, 350 UNKNOWN (1,050 total).
2. Strict operational taxonomy mappings from AmbigNQ and SelfAware.
3. Strict benchmark isolation: Every candidate is embedded via all-minilm and verified to have max cosine similarity < 0.75 with the 300-question benchmark.
4. Exact and semantic deduplication.
5. Stable ID and question_group_id assignment.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Set
import numpy as np

from src.preprocessing.embeddings import EmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_curator_1050")


def clean_text(text: str) -> str:
    """Normalize text for exact duplicate matching."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_benchmark_embeddings(engine: EmbeddingEngine, benchmark_path: str):
    logger.info("Loading and embedding 300-question benchmark from %s...", benchmark_path)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_records = [json.loads(line) for line in f]
    bench_qs = [clean_text(r["question"]) for r in bench_records]
    bench_embs = engine.embed_batch(bench_qs)
    logger.info("Computed embeddings for %d benchmark questions.", len(bench_qs))
    return bench_qs, bench_embs


def curate_dataset():
    os.makedirs("data/datasets", exist_ok=True)
    engine = EmbeddingEngine(embedding_model="all-minilm")
    
    benchmark_path = "data/benchmarks/uncertainty_benchmark_300_draft.jsonl"
    bench_qs, bench_embs = load_benchmark_embeddings(engine, benchmark_path)
    bench_qs_set = set(q.lower() for q in bench_qs)

    seen_questions: Set[str] = set()
    rejected_reasons: Dict[str, int] = {
        "benchmark_exact_match": 0,
        "benchmark_semantic_similarity_ge_075": 0,
        "corpus_exact_duplicate": 0,
        "philosophical_scientific_debate": 0,
        "trick_counting_zero_none": 0,
        "empty_or_too_short": 0,
    }

    def is_isolated(q: str) -> bool:
        norm_q = clean_text(q).lower()
        if norm_q in bench_qs_set:
            rejected_reasons["benchmark_exact_match"] += 1
            return False
        if norm_q in seen_questions:
            rejected_reasons["corpus_exact_duplicate"] += 1
            return False
            
        # Semantic similarity check
        emb = engine.embed_single(q)
        sims = np.dot(bench_embs, emb)
        max_sim = float(np.max(sims))
        if max_sim >= 0.75:
            rejected_reasons["benchmark_semantic_similarity_ge_075"] += 1
            return False
            
        seen_questions.add(norm_q)
        return True

    # -------------------------------------------------------------
    # 1. Harvest AMBIGUOUS (Target: 350)
    # Source: AmbigNQ multipleQAs (multiple distinct interpretations)
    # -------------------------------------------------------------
    logger.info("Harvesting AMBIGUOUS candidates from AmbigNQ...")
    ambig_candidates: List[Dict[str, Any]] = []
    
    for ambig_file in ["ambignq_light/train_light.json", "ambignq_light/dev_light.json"]:
        if len(ambig_candidates) >= 350:
            break
        with open(ambig_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            if len(ambig_candidates) >= 350:
                break
            anns = item.get("annotations", [])
            if not anns or anns[0].get("type") != "multipleQAs":
                continue
                
            qtext = clean_text(item.get("question", ""))
            if len(qtext.split()) < 4:
                rejected_reasons["empty_or_too_short"] += 1
                continue
                
            qa_pairs = anns[0].get("qaPairs", [])
            if len(qa_pairs) < 2:
                continue
                
            if not is_isolated(qtext):
                continue
                
            sub_qs = [p.get("question") for p in qa_pairs if p.get("question")]
            ambig_candidates.append({
                "id": f"dev_ambig_{len(ambig_candidates)+1:04d}",
                "question": qtext,
                "target_label": "AMBIGUOUS",
                "question_group_id": f"grp_ambig_{item.get('id')}",
                "source_dataset": "AmbigNQ",
                "source_id": str(item.get("id")),
                "original_source_label": "multipleQAs",
                "mapping_rationale": "AmbigNQ verified multipleQAs: query lacks disambiguating constraints and admits multiple verified QA interpretations.",
                "interpretations": sub_qs,
                "domain": "open_domain",
            })
            
    logger.info("Harvested %d verified AMBIGUOUS candidates.", len(ambig_candidates))

    # -------------------------------------------------------------
    # 2. Harvest KNOWN (Target: 350)
    # Source: AmbigNQ singleAnswer & SelfAware answerable
    # -------------------------------------------------------------
    logger.info("Harvesting KNOWN candidates from AmbigNQ and SelfAware...")
    known_candidates: List[Dict[str, Any]] = []
    
    # 2a. From AmbigNQ singleAnswer (200 candidates)
    with open("ambignq_light/train_light.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if len(known_candidates) >= 200:
            break
        anns = item.get("annotations", [])
        if not anns or anns[0].get("type") != "singleAnswer":
            continue
        answers = anns[0].get("answer", [])
        if not answers or not answers[0]:
            continue
            
        qtext = clean_text(item.get("question", ""))
        if len(qtext.split()) < 4:
            rejected_reasons["empty_or_too_short"] += 1
            continue
        if not is_isolated(qtext):
            continue
            
        known_candidates.append({
            "id": f"dev_known_{len(known_candidates)+1:04d}",
            "question": qtext,
            "target_label": "KNOWN",
            "question_group_id": f"grp_known_{item.get('id')}",
            "source_dataset": "AmbigNQ",
            "source_id": str(item.get("id")),
            "original_source_label": "singleAnswer",
            "mapping_rationale": "AmbigNQ verified singleAnswer: sufficiently clear factual query with singular documented ground-truth answer.",
            "ground_truth_answers": answers,
            "domain": "open_domain",
        })

    # 2b. From SelfAware answerable (150 candidates)
    with open("data/raw/SelfAware.json", "r", encoding="utf-8") as f:
        sa_data = json.load(f)
    for item in sa_data.get("example", []):
        if len(known_candidates) >= 350:
            break
        if not item.get("answerable"):
            continue
        qtext = clean_text(item.get("question", ""))
        if len(qtext.split()) < 4:
            rejected_reasons["empty_or_too_short"] += 1
            continue
        if not is_isolated(qtext):
            continue
            
        known_candidates.append({
            "id": f"dev_known_{len(known_candidates)+1:04d}",
            "question": qtext,
            "target_label": "KNOWN",
            "question_group_id": f"grp_known_sa_{item.get('question_id')}",
            "source_dataset": "SelfAware",
            "source_id": str(item.get("question_id")),
            "original_source_label": f"answerable_{item.get('source', '')}",
            "mapping_rationale": "SelfAware answerable factual benchmark query with verifiable consensus answers.",
            "ground_truth_answers": item.get("answer", []),
            "domain": "open_domain",
        })
        
    logger.info("Harvested %d verified KNOWN candidates.", len(known_candidates))

    # -------------------------------------------------------------
    # 3. Harvest UNKNOWN (Target: 350)
    # Source: SelfAware unanswerable (filtered for epistemic barriers)
    # -------------------------------------------------------------
    logger.info("Harvesting UNKNOWN candidates from SelfAware unanswerable...")
    unknown_candidates: List[Dict[str, Any]] = []
    
    # Filter regex for philosophical debates to reject
    phil_pattern = re.compile(r"\b(consciousness|free will|multiverse|soul|god|meaning of life|freedom|moral|ethical|afterlife|why do humans exist)\b", re.I)
    # Filter regex for counting questions that can be answered by "Zero" or "None"
    count_zero_pattern = re.compile(r"^(how many|how much|what number of)\b", re.I)
    
    for item in sa_data.get("example", []):
        if len(unknown_candidates) >= 350:
            break
        if item.get("answerable"):
            continue
            
        qtext = clean_text(item.get("question", ""))
        if len(qtext.split()) < 4:
            rejected_reasons["empty_or_too_short"] += 1
            continue
            
        # Filter out philosophical debates
        if phil_pattern.search(qtext):
            rejected_reasons["philosophical_scientific_debate"] += 1
            continue
            
        # Filter out counting questions answerable by zero
        if count_zero_pattern.search(qtext):
            rejected_reasons["trick_counting_zero_none"] += 1
            continue
            
        if not is_isolated(qtext):
            continue
            
        unknown_candidates.append({
            "id": f"dev_unk_{len(unknown_candidates)+1:04d}",
            "question": qtext,
            "target_label": "UNKNOWN",
            "question_group_id": f"grp_unk_sa_{item.get('question_id')}",
            "source_dataset": "SelfAware",
            "source_id": str(item.get("question_id")),
            "original_source_label": "unanswerable",
            "mapping_rationale": "SelfAware unanswerable item with documented epistemic barrier (inaccessible state, unrecorded history, future contingency, or false presupposition).",
            "domain": "epistemic_uncertainty",
        })
        
    logger.info("Harvested %d verified UNKNOWN candidates.", len(unknown_candidates))

    # -------------------------------------------------------------
    # 4. Final Aggregation & Verification
    # -------------------------------------------------------------
    all_candidates = known_candidates + ambig_candidates + unknown_candidates
    logger.info("Total harvested dataset: %d samples (KNOWN: %d, AMBIGUOUS: %d, UNKNOWN: %d)",
                len(all_candidates), len(known_candidates), len(ambig_candidates), len(unknown_candidates))
                
    assert len(known_candidates) == 350, f"Expected 350 KNOWN, got {len(known_candidates)}"
    assert len(ambig_candidates) == 350, f"Expected 350 AMBIGUOUS, got {len(ambig_candidates)}"
    assert len(unknown_candidates) == 350, f"Expected 350 UNKNOWN, got {len(unknown_candidates)}"

    # Save development questions
    dev_questions_path = "data/datasets/development_questions.jsonl"
    with open(dev_questions_path, "w", encoding="utf-8") as f:
        for item in all_candidates:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved 1,050 development questions to %s", dev_questions_path)

    # Save manifest
    manifest_path = "data/datasets/dataset_manifest.json"
    manifest_data = {
        "dataset_name": "Uncertainty_Classification_Development_Corpus_1050",
        "total_samples": len(all_candidates),
        "class_distribution": {
            "KNOWN": len(known_candidates),
            "AMBIGUOUS": len(ambig_candidates),
            "UNKNOWN": len(unknown_candidates),
        },
        "source_distribution": {
            "AmbigNQ": len([c for c in all_candidates if c["source_dataset"] == "AmbigNQ"]),
            "SelfAware": len([c for c in all_candidates if c["source_dataset"] == "SelfAware"]),
        },
        "isolation_standard": "all-minilm cosine similarity < 0.75 against 300-question gold benchmark",
        "rejection_statistics": rejected_reasons,
        "files": {
            "questions": "data/datasets/development_questions.jsonl",
            "responses_target": "data/datasets/development_responses.jsonl",
            "features_target": "data/features/development_features.jsonl",
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved dataset manifest to %s", manifest_path)
    print("\nDataset Manifest Summary:")
    print(json.dumps(manifest_data, indent=2))


if __name__ == "__main__":
    curate_dataset()
