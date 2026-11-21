#!/usr/bin/env python3
"""Hardened batch response generation script for uncertainty benchmarks with caching, provenance, and failure recovery."""

import sys
import os
import json
import time
import fcntl
import argparse
import datetime
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.ollama_client import OllamaClient, OllamaClientError
from src.generation.generator import ResponseGenerator
from src.evaluation.data_validator import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_responses")

DEFAULT_BENCHMARK_PATH = "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl"
DEFAULT_OUTPUT_PATH = "data/raw/generated_responses.jsonl"


class ProcessLock:
    """Acquires an exclusive advisory file lock using fcntl.flock to prevent concurrent generation runs."""

    def __init__(self, lock_file_path: str):
        self.lock_file_path = lock_file_path
        self.fd: Optional[Any] = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_file_path)), exist_ok=True)
        self.fd = open(self.lock_file_path, "w")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, IOError, OSError) as e:
            self.fd.close()
            self.fd = None
            raise RuntimeError(
                f"Exclusive lock could not be acquired for {self.lock_file_path}. "
                "Another response generation process is currently executing on this target."
            ) from e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except Exception:
                pass
            self.fd.close()
            self.fd = None


def recover_and_load_checkpoint(output_path: str) -> Tuple[Set[str], int]:
    """Recovers valid records from output_path, discarding any trailing interrupted line.
    
    Returns:
        Tuple of (completed_question_ids, count_valid_records).
    """
    if not os.path.exists(output_path):
        return set(), 0

    valid_records: List[Dict[str, Any]] = []
    completed_ids: Set[str] = set()
    last_valid_offset = 0
    corrupted_lines_found = False

    with open(output_path, "rb") as f:
        offset = 0
        for raw_line in f:
            line_len = len(raw_line)
            stripped = raw_line.strip()
            if stripped:
                try:
                    line_str = stripped.decode("utf-8")
                    rec = json.loads(line_str)
                    if isinstance(rec, dict) and "id" in rec and "responses" in rec:
                        # Verify responses list is non-empty
                        valid_records.append(rec)
                        completed_ids.add(str(rec["id"]))
                        last_valid_offset = offset + line_len
                    else:
                        raise ValueError(f"Record missing required keys: {list(rec.keys()) if isinstance(rec, dict) else type(rec)}")
                except Exception as e:
                    logger.warning(
                        "Corrupted/incomplete JSON line detected at byte offset %d: %s. Discarding trailing content.",
                        offset, e
                    )
                    corrupted_lines_found = True
                    break
            else:
                last_valid_offset = offset + line_len
            offset += line_len

    if corrupted_lines_found:
        with open(output_path, "r+b") as f:
            f.seek(last_valid_offset)
            f.truncate()
        logger.info("Output file truncated to byte offset %d to recover clean state.", last_valid_offset)

    return completed_ids, len(valid_records)


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate stochastic responses for benchmark questions.")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_BENCHMARK_PATH,
                        help=f"Input JSONL questions file (default: {DEFAULT_BENCHMARK_PATH})")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"Output JSONL generations file (default: {DEFAULT_OUTPUT_PATH})")
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
    timeout = float(ollama_cfg.get("request_timeout", 45.0))

    client = OllamaClient(host=host, request_timeout=timeout)
    if not client.is_available():
        logger.error("Ollama service at %s is not reachable.", host)
        sys.exit(1)

    ollama_ver = client.get_version()
    model_info = client.get_model_info(model)
    model_digest = model_info.get("digest") or model_info.get("details", {}).get("parent_model", "unknown")
    logger.info("Ollama version: %s | Model: %s (digest: %s)", ollama_ver, model, model_digest[:16] if model_digest else "none")

    generator = ResponseGenerator(client=client, model=model, temperature=temp, top_p=top_p)

    # Load questions
    questions: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    q_data = json.loads(line)
                    questions.append(q_data)
                except json.JSONDecodeError as e:
                    logger.error("Malformed JSON in %s at line %d: %s", args.input, line_num, e)
                    sys.exit(1)

    if args.limit:
        questions = questions[:args.limit]

    logger.info("Loaded %d questions from %s", len(questions), args.input)

    # Acquire exclusive process lock
    lock_file = args.output + ".lock"
    with ProcessLock(lock_file):
        # Recover checkpoint and existing completed question IDs
        completed_ids, recovered_count = recover_and_load_checkpoint(args.output)
        if recovered_count > 0:
            logger.info("Recovered %d completed records from %s. Resuming...", recovered_count, args.output)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        with open(args.output, "a", encoding="utf-8") as out_f:
            for q_item in tqdm(questions, desc="Generating responses"):
                qid = str(q_item["id"])
                q_text = q_item["question"]

                if qid in completed_ids:
                    continue

                # Canonical label and group extraction using DatasetValidator
                try:
                    gold_label = DatasetValidator.extract_label(q_item)
                except Exception as e:
                    logger.error("Failed to extract canonical label for question %s: %s", qid, e)
                    raise

                try:
                    group_ids = DatasetValidator.extract_group_ids(q_item)
                    primary_group_id = group_ids[0] if group_ids else DatasetValidator.get_question_group_key(q_text)
                except Exception as e:
                    logger.error("Failed to extract group ID for question %s: %s", qid, e)
                    raise

                # Outer retry loop for connection resilience
                max_q_retries = 3
                responses = None
                elapsed = 0.0

                for q_attempt in range(max_q_retries):
                    try:
                        start_t = time.time()
                        responses = generator.generate_multi(
                            question=q_text,
                            target_total_count=args.n_responses,
                            stochastic_temperature=temp,
                            greedy_temperature=0.0
                        )
                        elapsed = time.time() - start_t
                        break
                    except (OllamaClientError, ConnectionError, TimeoutError) as e:
                        logger.warning(
                            "Transient connection failure generating for %s (attempt %d/%d): %s",
                            qid, q_attempt + 1, max_q_retries, e
                        )
                        if q_attempt < max_q_retries - 1:
                            backoff = 2.0 ** q_attempt
                            time.sleep(backoff)
                            # Verify client availability before retrying
                            if not client.is_available():
                                logger.warning("Ollama daemon temporarily unavailable, waiting for reconnection...")
                                time.sleep(2.0)
                        else:
                            logger.error("Exhausted retries for question %s", qid)
                            raise

                record = {
                    "id": qid,
                    "question": q_text,
                    "gold_label": gold_label,
                    "ground_truth_label": gold_label,
                    "question_group_id": primary_group_id,
                    "question_group_ids": group_ids,
                    "category": q_item.get("category"),
                    "intended_interpretation": q_item.get("intended_interpretation"),
                    "temporal_anchor": q_item.get("temporal_anchor"),
                    "domain": q_item.get("domain", "general"),
                    "source": q_item.get("source"),
                    "model": model,
                    "temperature": temp,
                    "top_p": top_p,
                    "n_responses": len(responses) if responses else 0,
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "provenance": {
                        "ollama_version": ollama_ver,
                        "model": model,
                        "model_digest": str(model_digest),
                        "system_prompt": generator.system_prompt,
                        "seed": None,
                        "sampling_protocol": "r1: T=0.0 (greedy), r2-r10: T=0.7, top_p=0.9 (stochastic)",
                    },
                    "responses": responses,
                }

                # Atomic append and fsync for crash durability
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
                completed_ids.add(qid)

    logger.info("Generation complete. Output saved to %s", args.output)


if __name__ == "__main__":
    main()
