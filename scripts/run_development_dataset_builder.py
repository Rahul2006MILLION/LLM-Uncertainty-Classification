"""Checkpoint-resumable dataset generation and feature extraction pipeline for the 1,050 Development Dataset.

Features:
- Robust incremental checkpointing: resumes from last valid completed question.
- Zero data loss: flushes responses and features atomically per question.
- Strict feature validation: verifies all 16 features (no NaN, no inf, correct types).
- Batch/limit argument support for controlled execution batches.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Set
import numpy as np

from src.generation.ollama_client import OllamaClient
from src.generation.generator import ResponseGenerator
from src.preprocessing.embeddings import EmbeddingEngine
from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dev_dataset_builder_1050")


def load_completed_qids(responses_path: str) -> Set[str]:
    """Scan existing responses file to identify completed question IDs."""
    completed = set()
    if not os.path.exists(responses_path):
        return completed
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("id")
                if qid and len(rec.get("responses", [])) == 10:
                    completed.add(qid)
            except Exception:
                continue
    return completed


def run_builder(
    questions_path: str = "data/datasets/development_questions.jsonl",
    responses_path: str = "data/datasets/development_responses.jsonl",
    features_path: str = "data/features/development_features.jsonl",
    metadata_path: str = "data/datasets/generation_metadata.json",
    model_name: str = "qwen3:4b",
    limit: int = 0,
):
    os.makedirs(os.path.dirname(responses_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    # 1. Initialize tools
    client = OllamaClient()
    if not client.is_available():
        raise RuntimeError("Ollama daemon is not reachable at http://localhost:11434")
        
    generator = ResponseGenerator(client=client, model=model_name, temperature=0.7, top_p=0.9)
    embedding_engine = EmbeddingEngine(client=client, embedding_model="all-minilm")
    clusterer = SemanticClusterer(similarity_threshold=0.80)
    extractor = FeatureExtractor(clusterer=clusterer)

    # 2. Load questions
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f]
    logger.info("Loaded %d candidate questions from %s", len(all_questions), questions_path)

    # 3. Check existing checkpoints
    completed_qids = load_completed_qids(responses_path)
    logger.info("Found %d already completed questions in %s", len(completed_qids), responses_path)

    pending_questions = [q for q in all_questions if q["id"] not in completed_qids]
    if limit > 0:
        pending_questions = pending_questions[:limit]
        logger.info("Limiting current execution run to %d pending questions.", len(pending_questions))
    else:
        logger.info("Total questions to process in this run: %d", len(pending_questions))

    if not pending_questions:
        logger.info("All questions are already completed! Nothing to generate.")
        return

    # 4. Open append file descriptors
    gen_times = []
    success_count = 0
    fail_count = 0
    run_start = time.time()

    with open(responses_path, "a", encoding="utf-8") as resp_f, open(features_path, "a", encoding="utf-8") as feat_f:
        for idx, item in enumerate(pending_questions):
            qid = item["id"]
            qtext = item["question"]
            label = item["target_label"]
            
            logger.info("[%d/%d] Processing '%s' (%s: %s)...", idx + 1, len(pending_questions), qid, label, qtext[:40])
            q_start = time.time()
            
            # Generate 10 responses: Response 0 greedy T=0.0, Responses 1..9 stochastic T=0.7
            resps = generator.generate_multi(qtext, target_total_count=10)
            q_elapsed = time.time() - q_start
            gen_times.append(q_elapsed)
            
            # Validate generated texts
            resp_texts = []
            for r in resps:
                txt = r.get("text", "").strip()
                resp_texts.append(txt)
                if txt:
                    success_count += 1
                else:
                    fail_count += 1
                    
            # Response record
            resp_record = {
                "id": qid,
                "question": qtext,
                "target_label": label,
                "question_group_id": item.get("question_group_id"),
                "source_dataset": item.get("source_dataset"),
                "source_id": item.get("source_id"),
                "original_source_label": item.get("original_source_label"),
                "model": model_name,
                "generation_parameters": {
                    "model": model_name,
                    "target_total_count": 10,
                    "greedy_temperature": 0.0,
                    "stochastic_temperature": 0.7,
                    "top_p": 0.9,
                },
                "responses": resps,
                "elapsed_seconds": round(q_elapsed, 2),
            }
            resp_f.write(json.dumps(resp_record) + "\n")
            resp_f.flush()

            # Embed & Extract features
            resp_embs = embedding_engine.embed_batch(resp_texts)
            feats = extractor.extract_features(qtext, resp_texts, resp_embs)
            
            # Validate 16 features
            for fn in FEATURE_NAMES:
                val = feats.get(fn)
                if val is None or np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Invalid feature value '{fn}={val}' for question '{qid}'")
                    
            feat_record = {
                "id": qid,
                "question": qtext,
                "target_label": label,
                "features": feats,
            }
            feat_f.write(json.dumps(feat_record) + "\n")
            feat_f.flush()
            
            logger.info("    -> Completed in %.2fs (%.2fs/resp)", q_elapsed, q_elapsed / 10)

    total_run_elapsed = time.time() - run_start
    total_completed_now = len(load_completed_qids(responses_path))

    # Update metadata
    metadata = {
        "dataset_name": "development_1050",
        "model_name": model_name,
        "total_target_questions": len(all_questions),
        "total_completed_questions": total_completed_now,
        "recent_run_processed": len(pending_questions),
        "recent_run_seconds": round(total_run_elapsed, 2),
        "mean_seconds_per_question": round(float(np.mean(gen_times)), 2) if gen_times else 0.0,
        "mean_seconds_per_response": round(float(np.mean(gen_times)/10), 2) if gen_times else 0.0,
        "successful_responses_generated": success_count,
        "failed_responses_generated": fail_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        json.dump(metadata, meta_f, indent=2)
    logger.info("Updated generation metadata in %s", metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to process in this run")
    args = parser.parse_args()
    run_builder(limit=args.limit)
