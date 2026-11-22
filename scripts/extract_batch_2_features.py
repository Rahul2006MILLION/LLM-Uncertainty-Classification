#!/usr/bin/env python3
"""Feature extraction and unified dataset assembly for Batch 2 (N=30).

Extracts:
- 16 canonical features (9 consistency + 7 linguistic)
- True Single-Response features (r0)
- Single-Response Random features (r_rand from r1..r9)

Outputs:
- data/features/batch_2_features.jsonl
- data/batch_2_unified.jsonl
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.preprocessing.embeddings import EmbeddingEngine

RAW_RESPONSES_PATH = "data/raw/batch_2_generated_responses.jsonl"
QUESTIONS_PATH = "data/batch_2_questions.jsonl"
FEATURES_OUT = "data/features/batch_2_features.jsonl"
UNIFIED_OUT = "data/batch_2_unified.jsonl"


def main():
    print("=" * 60)
    print("Extracting canonical 16 features for Batch 2 (N=30)...")
    print("=" * 60)

    if not os.path.exists(RAW_RESPONSES_PATH):
        raise FileNotFoundError(f"Raw responses not found at {RAW_RESPONSES_PATH}")
    if not os.path.exists(QUESTIONS_PATH):
        raise FileNotFoundError(f"Questions file not found at {QUESTIONS_PATH}")

    # Load raw responses
    raw_records = []
    with open(RAW_RESPONSES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    # Load question metadata
    q_map = {}
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                q_map[rec["id"]] = rec

    print(f"Loaded {len(raw_records)} raw records and {len(q_map)} questions.")
    if len(raw_records) != 30:
        raise ValueError(f"Expected 30 raw records, found {len(raw_records)}")

    engine = EmbeddingEngine()
    clusterer = SemanticClusterer(similarity_threshold=0.80, linkage="average")
    extractor = FeatureExtractor(clusterer=clusterer)

    os.makedirs(os.path.dirname(FEATURES_OUT), exist_ok=True)
    features_records = []
    unified_records = []

    for idx, r in enumerate(raw_records):
        qid = r["id"]
        q_meta = q_map.get(qid, {})
        q_text = r["question"]
        label = r["gold_label"]
        resps = r.get("responses", [])
        
        if len(resps) != 10:
            raise ValueError(f"Record {qid} has {len(resps)} responses, expected 10")

        resp_texts = [resp.get("text", "").strip() for resp in resps]
        
        # 1. Embed responses (truncate for embedding to stay within 512-token context window of all-minilm)
        resp_texts_for_emb = [
            t[:500].rsplit(' ', 1)[0] if len(t) > 500 else t
            for t in resp_texts
        ]
        resp_embs = engine.embed_batch(resp_texts_for_emb)

        # 2. Extract 16 canonical features
        feats = extractor.extract_features(
            question=q_text,
            responses=resp_texts,
            embeddings=resp_embs,
        )

        # Validate canonical features
        for fn in FEATURE_NAMES:
            if fn not in feats:
                raise ValueError(f"Missing canonical feature '{fn}' for {qid}")
            val = feats[fn]
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid feature value '{fn}={val}' for {qid}")

        # 3. Extract True Single-Response features (r0, greedy T=0.0)
        r0_text = resp_texts[0]
        r0_feats = FeatureExtractor.extract_single_response_features(q_text, r0_text)

        # 4. Extract Single-Response Random features (r_rand from r1..r9 with deterministic seed)
        # Seed by integer hash of question ID
        h_val = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
        rand_idx = (h_val % 9) + 1  # picks index in 1..9
        rrand_text = resp_texts[rand_idx]
        rrand_feats = FeatureExtractor.extract_single_response_features(q_text, rrand_text)

        feat_entry = {
            "id": qid,
            "question": q_text,
            "target_label": label,
            "features": feats,
            "r0_features": r0_feats,
            "rrand_features": rrand_feats,
            "rrand_index": rand_idx,
        }
        features_records.append(feat_entry)

        # 5. Assemble unified record
        uni_entry = {
            "id": qid,
            "question": q_text,
            "target_label": label,
            "question_group_id": r.get("question_group_id", q_meta.get("question_group_id")),
            "category": q_meta.get("category", r.get("category", "")),
            "ground_truth": q_meta.get("ground_truth", ""),
            "ambiguity_mechanism": q_meta.get("ambiguity_mechanism", ""),
            "epistemic_barrier": q_meta.get("epistemic_barrier", ""),
            "mapping_rationale": q_meta.get("mapping_rationale", ""),
            "domain": q_meta.get("domain", "general"),
            "provenance": q_meta.get("provenance", "unknown"),
            "source": q_meta.get("source", ""),
            "intended_interpretation": q_meta.get("intended_interpretation"),
            "temporal_anchor": q_meta.get("temporal_anchor"),
            "model": r.get("model", "qwen3:4b"),
            "responses": resps,
            "features": feats,
            "r0_features": r0_feats,
            "rrand_features": rrand_feats,
            "rrand_index": rand_idx,
            "q_elapsed_seconds": r.get("elapsed_seconds", 0.0),
        }
        unified_records.append(uni_entry)

        print(f"[{idx+1:02d}/30] {qid:12s} ({label:9s}) -> Entropy: {feats['semantic_entropy']:.4f}, Clusters: {feats['cluster_count']:.0f}, Negations: {feats['negation_count']:.2f}")

    # Write features JSONL
    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        for entry in features_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(features_records)} feature records to {FEATURES_OUT}")

    # Write unified JSONL
    with open(UNIFIED_OUT, "w", encoding="utf-8") as f:
        for entry in unified_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(unified_records)} unified records to {UNIFIED_OUT}")


if __name__ == "__main__":
    main()
