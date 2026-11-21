#!/usr/bin/env python3
"""Feature extraction and response analysis pipeline across generated response sets."""

import sys
import os
import json
import argparse
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.ollama_client import OllamaClient
from src.preprocessing.embeddings import EmbeddingEngine
from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FeatureExtractor
from src.evaluation.data_validator import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_responses")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract semantic consistency features from generated responses.")
    parser.add_argument("--input", "-i", type=str, default="data/raw/generated_responses.jsonl",
                        help="Input JSONL with generated responses")
    parser.add_argument("--output", "-o", type=str, default="data/features.jsonl",
                        help="Output JSONL with extracted features")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Override similarity threshold for clustering")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ollama_cfg = cfg.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    emb_model = ollama_cfg.get("embedding_model", "all-minilm")

    clust_cfg = cfg.get("clustering", {})
    sim_threshold = args.threshold if args.threshold is not None else float(clust_cfg.get("similarity_threshold", 0.80))

    client = OllamaClient(host=host)
    embedder = EmbeddingEngine(client=client, embedding_model=emb_model)
    clusterer = SemanticClusterer(similarity_threshold=sim_threshold)
    extractor = FeatureExtractor(clusterer=clusterer)

    if not os.path.exists(args.input):
        logger.error("Input file %s does not exist. Run generate_responses.py first.", args.input)
        sys.exit(1)

    records: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Extracting features for %d items using similarity threshold %.2f...", len(records), sim_threshold)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in tqdm(records, desc="Extracting features"):
            q_text = item["question"]
            responses = [r["text"] for r in item.get("responses", [])]

            embeddings = embedder.embed_batch(responses)
            cluster_res = clusterer.cluster_responses(responses, embeddings)

            features = extractor.extract_features(
                question=q_text,
                responses=responses,
                embeddings=embeddings,
                cluster_result=cluster_res,
            )

            # Robust label extraction supporting gold_label, ground_truth_label, and label
            label = None
            if "label" in item or "gold_label" in item:
                try:
                    label = DatasetValidator.extract_label(item)
                except Exception:
                    label = item.get("gold_label") or item.get("label") or item.get("ground_truth_label")
            else:
                label = item.get("ground_truth_label")

            # Group preservation
            try:
                group_ids = DatasetValidator.extract_group_ids(item)
                primary_group_id = group_ids[0] if group_ids else DatasetValidator.get_question_group_key(q_text)
            except Exception:
                primary_group_id = item.get("question_group_id") or DatasetValidator.get_question_group_key(q_text)
                group_ids = item.get("question_group_ids") or [primary_group_id]

            # Join features with metadata and target label (preventing any leakage during feature calculation)
            out_record = {
                "id": item["id"],
                "question": q_text,
                "label": label,
                "gold_label": label,
                "question_group_id": primary_group_id,
                "question_group_ids": group_ids,
                "category": item.get("category"),
                "intended_interpretation": item.get("intended_interpretation"),
                "temporal_anchor": item.get("temporal_anchor"),
                "domain": item.get("domain", "general"),
                "source": item.get("source"),
                "similarity_threshold": sim_threshold,
                "features": features,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    logger.info("Feature extraction complete. Saved to %s", args.output)


if __name__ == "__main__":
    main()
