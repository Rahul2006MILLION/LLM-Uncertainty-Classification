#!/usr/bin/env python3
"""Automated, reproducible experiment runner across test suites and benchmark datasets."""

import sys
import os
import json
import time
import argparse
import datetime
import logging
from typing import Dict, Any, List
from tabulate import tabulate
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.ask import run_pipeline, load_config
from data.schema import ExperimentRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmarks")

PREDEFINED_TEST_CASES = [
    {"question": "What is Data Science?", "expected_path": "FAST_PATH", "type": "Concept Definition"},
    {"question": "What is 2 + 2?", "expected_path": "FAST_PATH", "type": "Arithmetic"},
    {"question": "What is the capital of Georgia?", "expected_path": "MULTI_RESPONSE", "type": "Geographic Polysemy"},
    {"question": "Who is Jordan?", "expected_path": "MULTI_RESPONSE", "type": "Ambiguous Entity"},
    {"question": "What is the largest city in the world?", "expected_path": "MULTI_RESPONSE", "type": "Criterion Ambiguity"},
    {"question": "What number am I thinking of right now?", "expected_path": "MULTI_RESPONSE", "type": "Insufficient Information"},
    {"question": "Which organization organizes the FIFA Ice Hockey World Cup?", "expected_path": "MULTI_RESPONSE", "type": "False Premise Factual"},
]


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark test suite.")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml")
    parser.add_argument("--output", "-o", type=str, default="experiments/results/benchmark_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results: List[Dict[str, Any]] = []

    logger.info("Executing predefined test suite against live model...")
    table_rows = []

    for test_case in PREDEFINED_TEST_CASES:
        q = test_case["question"]
        exp_path = test_case["expected_path"]
        q_type = test_case["type"]

        t_start = time.time()
        try:
            record: ExperimentRecord = run_pipeline(question=q, config=cfg)
            match = "PASS" if record.path == exp_path else f"FAIL (got {record.path})"
            status = "SUCCESS"
            err_msg = None
        except Exception as e:
            status = "ERROR"
            err_msg = str(e)
            match = "ERROR"
            record = None

        row = [
            q[:32] + ("..." if len(q) > 32 else ""),
            q_type,
            exp_path,
            record.path if record else "ERROR",
            f"{record.cluster_count}" if record else "-",
            f"{record.semantic_entropy:.2f}" if record else "-",
            f"{record.elapsed_seconds:.2f}s" if record else "-",
            match,
        ]
        table_rows.append(row)

        results.append({
            "question": q,
            "type": q_type,
            "expected_path": exp_path,
            "status": status,
            "error": err_msg,
            "record": record.model_dump() if record else None,
        })

    print("\n" + tabulate(
        table_rows,
        headers=["Question", "Category", "Expected", "Actual", "Clusters", "Entropy", "Latency", "Result"],
        tablefmt="github"
    ) + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "total_tests": len(PREDEFINED_TEST_CASES),
            "results": results,
        }, f, indent=2)

    logger.info("Benchmark report exported to %s", args.output)


if __name__ == "__main__":
    main()
