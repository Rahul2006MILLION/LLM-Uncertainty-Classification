"""Test checkpoint, recovery, and resume behavior for Batch 1 generation pipeline."""

import os
import json
import tempfile
import pytest
import numpy as np

from src.features.feature_extractor import FEATURE_NAMES


def scan_completed_qids(responses_path: str, features_path: str) -> set:
    """Validate and return fully completed question IDs present in both files."""
    if not os.path.exists(responses_path) or not os.path.exists(features_path):
        return set()

    completed_in_resp = set()
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("id")
                resps = rec.get("responses", [])
                if qid and len(resps) == 10:
                    completed_in_resp.add(qid)
            except Exception:
                continue

    completed_in_feat = set()
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("id")
                feats = rec.get("features", {})
                if qid and all(fn in feats and not np.isnan(feats[fn]) for fn in FEATURE_NAMES):
                    completed_in_feat.add(qid)
            except Exception:
                continue

    return completed_in_resp.intersection(completed_in_feat)


def test_clean_checkpoint_scan():
    with tempfile.TemporaryDirectory() as tmpdir:
        resp_p = os.path.join(tmpdir, "resp.jsonl")
        feat_p = os.path.join(tmpdir, "feat.jsonl")

        # 1. Empty files
        assert scan_completed_qids(resp_p, feat_p) == set()

        # 2. Add complete question 1
        dummy_feats = {fn: 0.5 for fn in FEATURE_NAMES}
        dummy_resps = [{"response_id": i, "text": f"Ans {i}"} for i in range(10)]
        with open(resp_p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q1", "responses": dummy_resps}) + "\n")
        with open(feat_p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q1", "features": dummy_feats}) + "\n")

        assert scan_completed_qids(resp_p, feat_p) == {"q1"}

        # 3. Add partial question 2 (only 5 responses in resp file, missing in feat file)
        with open(resp_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q2", "responses": dummy_resps[:5]}) + "\n")

        # q2 should NOT be marked completed
        assert scan_completed_qids(resp_p, feat_p) == {"q1"}

        # 4. Corrupt line in feat file
        with open(feat_p, "a", encoding="utf-8") as f:
            f.write('{"id": "q3", "features": {"broken":\n')

        # q3 should NOT be marked completed, q1 remains completed
        assert scan_completed_qids(resp_p, feat_p) == {"q1"}

        # 5. Question 4 in feat but missing in resp
        with open(feat_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q4", "features": dummy_feats}) + "\n")
        assert scan_completed_qids(resp_p, feat_p) == {"q1"}

        # 6. Complete question 4 in resp as well
        with open(resp_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q4", "responses": dummy_resps}) + "\n")
        assert scan_completed_qids(resp_p, feat_p) == {"q1", "q4"}
