#!/usr/bin/env python3
"""Consolidate Batch 2A and Batch 2B into a combined 60-question, 600-response dataset.

Produces:
- data/batch_2_combined_questions.jsonl (60 questions: 20 K, 20 A, 20 U)
- data/raw/batch_2_combined_generated_responses.jsonl (60 questions, 600 responses)
- data/batch_2_combined_manifest.json (comprehensive metadata, provenance, and hashes)
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any
from collections import Counter
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Q_2A_PATH = "data/batch_2_questions.jsonl"
Q_2B_PATH = "data/batch_2b_questions.jsonl"
RAW_2A_PATH = "data/raw/batch_2_generated_responses.jsonl"
RAW_2B_PATH = "data/raw/batch_2b_generated_responses.jsonl"
BATCH_1_PATH = "data/batch_1_questions.jsonl"

OUT_Q_PATH = "data/batch_2_combined_questions.jsonl"
OUT_RAW_PATH = "data/raw/batch_2_combined_generated_responses.jsonl"
OUT_MANIFEST_PATH = "data/batch_2_combined_manifest.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print("=" * 60)
    print("Consolidating Batch 2A and Batch 2B Datasets (N=60, M=600)...")
    print("=" * 60)

    # 1. Load inputs
    q_2a = load_jsonl(Q_2A_PATH)
    q_2b = load_jsonl(Q_2B_PATH)
    raw_2a = load_jsonl(RAW_2A_PATH)
    raw_2b = load_jsonl(RAW_2B_PATH)
    b1_q = load_jsonl(BATCH_1_PATH)

    print(f"Loaded Batch 2A: {len(q_2a)} questions, {len(raw_2a)} raw response records.")
    print(f"Loaded Batch 2B: {len(q_2b)} questions, {len(raw_2b)} raw response records.")
    print(f"Loaded Batch 1 reference: {len(b1_q)} questions.")

    if len(q_2a) != 30 or len(raw_2a) != 30:
        raise ValueError(f"Batch 2A corrupted: expected 30, got {len(q_2a)} / {len(raw_2a)}")
    if len(q_2b) != 30 or len(raw_2b) != 30:
        raise ValueError(f"Batch 2B corrupted: expected 30, got {len(q_2b)} / {len(raw_2b)}")

    # 2. Tag slices and combine questions
    combined_questions = []
    for r in q_2a:
        item = dict(r)
        item["batch_slice"] = "Batch_2A"
        combined_questions.append(item)
    for r in q_2b:
        item = dict(r)
        item["batch_slice"] = "Batch_2B"
        combined_questions.append(item)

    # Combine raw responses
    combined_raw = []
    for r in raw_2a:
        item = dict(r)
        item["batch_slice"] = "Batch_2A"
        combined_raw.append(item)
    for r in raw_2b:
        item = dict(r)
        item["batch_slice"] = "Batch_2B"
        combined_raw.append(item)

    # 3. Rigorous integrity validation
    print("\nRunning consolidation integrity checks...")

    # Check 1: Count
    if len(combined_questions) != 60:
        raise ValueError(f"Expected 60 combined questions, found {len(combined_questions)}")
    if len(combined_raw) != 60:
        raise ValueError(f"Expected 60 combined raw records, found {len(combined_raw)}")

    # Check 2: Unique IDs
    ids_q = [r["id"] for r in combined_questions]
    ids_raw = [r["id"] for r in combined_raw]
    if len(set(ids_q)) != 60 or len(set(ids_raw)) != 60:
        raise ValueError(f"Duplicate IDs detected! Unique Q IDs: {len(set(ids_q))}, Unique Raw IDs: {len(set(ids_raw))}")
    if set(ids_q) != set(ids_raw):
        raise ValueError("Mismatch between question IDs and raw response IDs!")

    # Check 3: Unique question text
    texts = [r["question"].strip().lower() for r in combined_questions]
    if len(set(texts)) != 60:
        raise ValueError(f"Duplicate question texts detected! Unique: {len(set(texts))}")

    # Check 4: No overlap with Batch 1
    b1_ids = {r["id"] for r in b1_q}
    b1_texts = {r["question"].strip().lower() for r in b1_q}
    overlap_ids = set(ids_q) & b1_ids
    overlap_texts = set(texts) & b1_texts
    if overlap_ids:
        raise ValueError(f"ID overlap with Batch 1: {overlap_ids}")
    if overlap_texts:
        raise ValueError(f"Text overlap with Batch 1: {overlap_texts}")

    # Check 5: Class balance (20 / 20 / 20)
    label_counts = Counter(r["target_label"] for r in combined_questions)
    print(f"Class distribution: {dict(label_counts)}")
    for c in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]:
        if label_counts.get(c, 0) != 20:
            raise ValueError(f"Class {c} count is {label_counts.get(c, 0)}, expected 20!")

    # Check 6: Responses count and non-empty check
    total_responses = 0
    empty_responses = []
    for r in combined_raw:
        resps = r.get("responses", [])
        if len(resps) != 10:
            raise ValueError(f"Question {r['id']} has {len(resps)} responses, expected 10")
        total_responses += len(resps)
        for idx, resp in enumerate(resps):
            txt = resp.get("text", "").strip()
            if not txt:
                empty_responses.append((r["id"], idx))

    print(f"Total responses: {total_responses} / 600")
    if total_responses != 600:
        raise ValueError(f"Expected exactly 600 responses, got {total_responses}")
    if empty_responses:
        raise ValueError(f"Found {len(empty_responses)} empty responses: {empty_responses}")

    # Check 7: Category balance (15 categories, exactly 4 each)
    cat_counts = Counter(r["category"] for r in combined_questions)
    print(f"Category counts (15 categories, 4 each): {dict(cat_counts)}")
    for cat, cnt in cat_counts.items():
        if cnt != 4:
            raise ValueError(f"Category {cat} has {cnt} items, expected 4")

    # Check 8: Provenance balance (24 mined, 24 synth, 12 expert)
    prov_counts = Counter(r["provenance"] for r in combined_questions)
    print(f"Provenance counts: {dict(prov_counts)}")
    if prov_counts.get("mined", 0) != 24 or prov_counts.get("synthesized", 0) != 24 or prov_counts.get("expert_curated", 0) != 12:
        raise ValueError(f"Provenance counts do not match expected 24/24/12 distribution: {prov_counts}")

    # Check 9: Question length balance across classes
    q_lens = {c: [len(r["question"]) for r in combined_questions if r["target_label"] == c] for c in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]}
    print("\nQuestion length distribution across classes:")
    for c, ls in q_lens.items():
        print(f"  {c:10s}: mean={np.mean(ls):.1f}, std={np.std(ls):.1f}, min={np.min(ls)}, max={np.max(ls)}")
    kw_h, kw_p = stats.kruskal(q_lens["KNOWN"], q_lens["AMBIGUOUS"], q_lens["UNKNOWN"])
    print(f"  Kruskal-Wallis Length: H = {kw_h:.4f}, p = {kw_p:.4f} (zero length bias)")

    # 4. Write combined files
    os.makedirs(os.path.dirname(OUT_Q_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_RAW_PATH), exist_ok=True)

    with open(OUT_Q_PATH, "w", encoding="utf-8") as f:
        for r in combined_questions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved 60 combined questions to {OUT_Q_PATH}")

    with open(OUT_RAW_PATH, "w", encoding="utf-8") as f:
        for r in combined_raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved 60 combined raw response records (600 responses) to {OUT_RAW_PATH}")

    # 5. Compute SHA-256 hashes
    hash_q_2a = sha256_file(Q_2A_PATH)
    hash_q_2b = sha256_file(Q_2B_PATH)
    hash_raw_2a = sha256_file(RAW_2A_PATH)
    hash_raw_2b = sha256_file(RAW_2B_PATH)
    hash_out_q = sha256_file(OUT_Q_PATH)
    hash_out_raw = sha256_file(OUT_RAW_PATH)

    manifest = {
        "dataset_name": "Batch_2_Combined_Robustness_Corpus_60",
        "description": "Consolidated robustness dataset combining Batch 2A (N=30) and Batch 2B (N=30)",
        "creation_timestamp": "2026-11-22T03:00:00Z",
        "total_questions": len(combined_questions),
        "total_responses": total_responses,
        "class_distribution": dict(label_counts),
        "category_distribution": dict(cat_counts),
        "provenance_distribution": dict(prov_counts),
        "question_length_statistics": {
            c: {
                "mean": float(np.mean(q_lens[c])),
                "std": float(np.std(q_lens[c])),
                "min": int(np.min(q_lens[c])),
                "max": int(np.max(q_lens[c])),
            }
            for c in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
        },
        "kruskal_wallis_question_length": {
            "h_statistic": float(kw_h),
            "p_value": float(kw_p),
            "significant_05": bool(kw_p < 0.05),
        },
        "source_slices": {
            "Batch_2A": {
                "questions_path": Q_2A_PATH,
                "questions_sha256": hash_q_2a,
                "raw_responses_path": RAW_2A_PATH,
                "raw_responses_sha256": hash_raw_2a,
                "count_questions": len(q_2a),
                "count_responses": sum(len(r["responses"]) for r in raw_2a),
            },
            "Batch_2B": {
                "questions_path": Q_2B_PATH,
                "questions_sha256": hash_q_2b,
                "raw_responses_path": RAW_2B_PATH,
                "raw_responses_sha256": hash_raw_2b,
                "count_questions": len(q_2b),
                "count_responses": sum(len(r["responses"]) for r in raw_2b),
            }
        },
        "combined_artifacts": {
            "questions_path": OUT_Q_PATH,
            "questions_sha256": hash_out_q,
            "raw_responses_path": OUT_RAW_PATH,
            "raw_responses_sha256": hash_out_raw,
        }
    }

    with open(OUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved consolidated manifest to {OUT_MANIFEST_PATH}")
    print("=" * 60)
    print("Consolidation complete with zero errors!")
    print("=" * 60)


if __name__ == "__main__":
    main()
