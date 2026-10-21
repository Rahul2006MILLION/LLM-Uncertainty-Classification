#!/usr/bin/env python3
"""Batch response generation script for uncertainty benchmarks with caching and provenance."""

import sys
import os
import json
import time
import argparse
import datetime
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.ollama_client import OllamaClient, OllamaClientError
from src.generation.generator import ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_responses")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate stochastic responses for benchmark questions.")
    parser.add_argument("--input", "-i", type=str, default="data/benchmarks/pilot_uncertainty_benchmark.jsonl",
                        help="Input JSONL questions file")
    parser.add_argument("--output", "-o", type=str, default="data/raw/generated_responses.jsonl",
                        help="Output JSONL generations file")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--n-responses", "-n", type=int, default=10, help="Number of responses per question")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ollama_cfg = cfg.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    model = ollama_cfg.get("generation_model", "qwen2.5:3b")
    temp = float(ollama_cfg.get("temperature", 0.7))
    top_p = float(ollama_cfg.get("top_p", 0.9))

    client = OllamaClient(host=host, request_timeout=float(ollama_cfg.get("request_timeout", 45.0)))
    if not client.is_available():
        logger.error("Ollama service at %s is not reachable.", host)
        sys.exit(1)

    generator = ResponseGenerator(client=client, model=model, temperature=temp, top_p=top_p)

    # Load questions
    questions: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    # Load existing cached outputs if file exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_ids.add(rec.get("id"))
        logger.info("Found %d already generated records in %s. Resuming...", len(existing_ids), args.output)

    # Process items
    with open(args.output, "a", encoding="utf-8") as out_f:
        for q_item in tqdm(questions, desc="Generating responses"):
            qid = q_item["id"]
            if qid in existing_ids:
                continue

            q_text = q_item["question"]
            start_t = time.time()
            responses = generator.generate_multi(question=q_text, target_total_count=args.n_responses)
            elapsed = time.time() - start_t

            record = {
                "id": qid,
                "question": q_text,
                "ground_truth_label": q_item.get("label"),
                "domain": q_item.get("domain", "general"),
                "source": q_item.get("source"),
                "model": model,
                "temperature": temp,
                "top_p": top_p,
                "n_responses": len(responses),
                "elapsed_seconds": elapsed,
                "timestamp": datetime.datetime.now().isoformat(),
                "responses": responses,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    logger.info("Generation complete. Output saved to %s", args.output)


if __name__ == "__main__":
    main()
