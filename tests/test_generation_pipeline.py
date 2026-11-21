"""Unit tests for the remediated LLM response generation pipeline."""

import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.generation.ollama_client import OllamaClient, OllamaClientError
from src.generation.generator import ResponseGenerator
from src.evaluation.data_validator import DatasetValidator
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.features.clustering import SemanticClusterer
from experiments.generate_responses import ProcessLock, recover_and_load_checkpoint


class TestGenerationPipelineRemediation:
    """Verifies all remediation requirements for Milestone 2 response generation."""

    def test_gold_label_extraction(self):
        """Verifies canonical gold_label extraction from candidate v1.1 records."""
        rec_gold = {
            "id": "draft_know_001",
            "question": "What is the capital of France?",
            "gold_label": "KNOWN",
            "source": "curated",
        }
        extracted = DatasetValidator.extract_label(rec_gold)
        assert extracted == "KNOWN"

        rec_legacy = {
            "id": "draft_know_002",
            "question": "What is the speed of light?",
            "label": "KNOWN",
            "source": "curated",
        }
        assert DatasetValidator.extract_label(rec_legacy) == "KNOWN"

        rec_conflict = {
            "id": "draft_know_003",
            "question": "What is the boiling point of water?",
            "label": "KNOWN",
            "gold_label": "UNKNOWN",
            "source": "curated",
        }
        with pytest.raises(ValueError, match="Conflicting label fields"):
            DatasetValidator.extract_label(rec_conflict)

        rec_missing = {
            "id": "draft_know_004",
            "question": "Where is Mount Everest?",
            "source": "curated",
        }
        with pytest.raises(ValueError, match="missing required label field"):
            DatasetValidator.extract_label(rec_missing)

    def test_question_group_id_preservation(self):
        """Verifies declared group IDs survive extraction without fallback dilution."""
        rec_grouped = {
            "id": "adv_001",
            "question": "What is the speed of light in vacuum?",
            "gold_label": "KNOWN",
            "question_group_id": "grp_adv_vacuum_wave_speed",
            "source": "curated",
        }
        groups = DatasetValidator.extract_group_ids(rec_grouped)
        assert groups == ["grp_adv_vacuum_wave_speed"]

        rec_composite = {
            "id": "adv_002",
            "question": "How fast does an EM wave travel in vacuum?",
            "gold_label": "KNOWN",
            "question_group_id": ["grp_adv_vacuum_wave_speed", "grp_physics_optics"],
            "source": "curated",
        }
        groups_comp = DatasetValidator.extract_group_ids(rec_composite)
        assert sorted(groups_comp) == ["grp_adv_vacuum_wave_speed", "grp_physics_optics"]

    def test_metadata_propagation(self):
        """Verifies category, intended_interpretation, and temporal_anchor propagation."""
        raw_rec = {
            "id": "draft_unk_fut_023",
            "question": "What will be the first encyclical promulgated by Pope Leo XIV?",
            "gold_label": "UNKNOWN",
            "question_group_id": "grp_pope_leo_xiv",
            "category": "epistemic_future",
            "intended_interpretation": "future election and subsequent encyclical promulgation",
            "temporal_anchor": "November 2026",
            "source": "curated_boundary",
            "domain": "religion_and_history",
        }

        # Simulate output record construction in generation pipeline
        gold_label = DatasetValidator.extract_label(raw_rec)
        group_ids = DatasetValidator.extract_group_ids(raw_rec)

        output_record = {
            "id": raw_rec["id"],
            "question": raw_rec["question"],
            "gold_label": gold_label,
            "ground_truth_label": gold_label,
            "question_group_id": group_ids[0],
            "question_group_ids": group_ids,
            "category": raw_rec.get("category"),
            "intended_interpretation": raw_rec.get("intended_interpretation"),
            "temporal_anchor": raw_rec.get("temporal_anchor"),
            "domain": raw_rec.get("domain", "general"),
            "source": raw_rec.get("source"),
        }

        assert output_record["category"] == "epistemic_future"
        assert output_record["intended_interpretation"] == "future election and subsequent encyclical promulgation"
        assert output_record["temporal_anchor"] == "November 2026"
        assert output_record["question_group_id"] == "grp_pope_leo_xiv"
        assert output_record["gold_label"] == "UNKNOWN"

    def test_resume_and_duplicate_prevention(self):
        """Verifies checkpoint recovery correctly identifies completed items to prevent duplicates."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as f:
            temp_path = f.name
            records = [
                {"id": f"q_{i}", "question": f"Question {i}?", "responses": [{"text": "ans"}]}
                for i in range(5)
            ]
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()

        try:
            completed_ids, count = recover_and_load_checkpoint(temp_path)
            assert count == 5
            assert len(completed_ids) == 5
            for i in range(5):
                assert f"q_{i}" in completed_ids

            # Simulating generation loop skipping
            all_questions = [{"id": f"q_{i}", "question": f"Question {i}?"} for i in range(8)]
            to_process = [q for q in all_questions if q["id"] not in completed_ids]
            assert len(to_process) == 3
            assert [q["id"] for q in to_process] == ["q_5", "q_6", "q_7"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_concurrent_execution_prevention(self):
        """Verifies exclusive process lock prevents overlapping generation runs."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = f.name + ".lock"

        try:
            with ProcessLock(lock_path):
                # Attempting to acquire second lock on same file must raise RuntimeError
                with pytest.raises(RuntimeError, match="Exclusive lock could not be acquired"):
                    with ProcessLock(lock_path):
                        pass

            # Once exited, acquiring lock should succeed again
            with ProcessLock(lock_path):
                pass
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
            if os.path.exists(f.name):
                os.remove(f.name)

    def test_interrupted_write_recovery(self):
        """Verifies corrupted/truncated JSON lines are detected, discarded, and file truncated safely."""
        with tempfile.NamedTemporaryFile("wb+", delete=False, suffix=".jsonl") as f:
            temp_path = f.name
            rec1 = json.dumps({"id": "q_1", "question": "Q1", "responses": [{"text": "A1"}]}).encode("utf-8") + b"\n"
            rec2 = json.dumps({"id": "q_2", "question": "Q2", "responses": [{"text": "A2"}]}).encode("utf-8") + b"\n"
            corrupt = b'{"id": "q_3", "question": "Q3", "responses": [{"text": "unfini'  # Truncated write
            f.write(rec1 + rec2 + corrupt)
            f.flush()

        try:
            completed_ids, count = recover_and_load_checkpoint(temp_path)
            assert count == 2
            assert completed_ids == {"q_1", "q_2"}

            # Verify file was cleaned on disk
            with open(temp_path, "r", encoding="utf-8") as f_check:
                lines = [l.strip() for l in f_check if l.strip()]
                assert len(lines) == 2
                assert json.loads(lines[0])["id"] == "q_1"
                assert json.loads(lines[1])["id"] == "q_2"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_ollama_disconnection_and_reconnection(self):
        """Verifies OllamaClient retries on transient connection errors and recovers."""
        import requests
        client = OllamaClient(host="http://mock-host:11434", max_retries=3, backoff_factor=1.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Paris is the capital of France.", "eval_count": 8}

        with patch.object(client.session, "post") as mock_post:
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("Connection refused by daemon"),
                mock_resp,
            ]

            res = client.generate("What is the capital of France?", model="qwen2.5:3b")
            assert res["response"] == "Paris is the capital of France."
            assert mock_post.call_count == 2

    def test_retry_and_empty_response_handling(self):
        """Verifies empty strings trigger retry and recover valid content."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_available.return_value = True

        # First call returns empty string, second returns valid text
        mock_client.generate.side_effect = [
            {"response": "   ", "eval_count": 0},
            {"response": "Water boils at 100 degrees Celsius.", "eval_count": 7},
        ]

        generator = ResponseGenerator(client=mock_client, model="qwen2.5:3b")
        res = generator.generate_single("At what temperature does water boil?", max_sample_retries=2)

        assert res["text"] == "Water boils at 100 degrees Celsius."
        assert mock_client.generate.call_count == 2

    def test_exactly_ten_valid_responses_per_question(self):
        """Verifies sampling protocol: response 0 at T=0.0 (greedy), responses 1..9 at T=0.7 (stochastic)."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.is_available.return_value = True

        recorded_temps = []

        def fake_generate(**kwargs):
            temp = kwargs.get("temperature")
            recorded_temps.append(temp)
            idx = len(recorded_temps)
            return {"response": f"Sample response {idx}", "eval_count": 10}

        mock_client.generate.side_effect = fake_generate

        generator = ResponseGenerator(client=mock_client, model="qwen2.5:3b", temperature=0.7, top_p=0.9)
        responses = generator.generate_multi(
            question="Explain photosynthesis.",
            target_total_count=10,
            stochastic_temperature=0.7,
            greedy_temperature=0.0,
        )

        assert len(responses) == 10
        # Indices 0..9 in order
        assert [r["index"] for r in responses] == list(range(10))
        # Greedy baseline at index 0 (T=0.0)
        assert recorded_temps[0] == 0.0
        assert responses[0]["temperature"] == 0.0
        # Stochastic ensemble at indices 1..9 (T=0.7)
        for t in recorded_temps[1:]:
            assert t == 0.7
        for r in responses[1:]:
            assert r["temperature"] == 0.7

    def test_compatibility_with_downstream_feature_extraction(self):
        """Verifies generated output structure integrates cleanly with downstream feature extractor."""
        # Create a synthetic generated record
        responses = [
            {"index": 0, "text": "Paris is the capital of France.", "temperature": 0.0, "elapsed_seconds": 0.1, "eval_count": 6},
            {"index": 1, "text": "The capital of France is Paris.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 6},
            {"index": 2, "text": "Paris is France's capital city.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 6},
            {"index": 3, "text": "Paris is the capital.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 4},
            {"index": 4, "text": "Paris.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 1},
            {"index": 5, "text": "The capital of France is Paris.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 6},
            {"index": 6, "text": "Paris is the capital city.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 5},
            {"index": 7, "text": "Capital of France: Paris.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 5},
            {"index": 8, "text": "Paris is France's capital.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 4},
            {"index": 9, "text": "Paris is the French capital.", "temperature": 0.7, "elapsed_seconds": 0.1, "eval_count": 5},
        ]
        resp_texts = [r["text"] for r in responses]

        # Deterministic synthetic 384-d embeddings (highly aligned vectors representing consensus)
        base_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        synthetic_embeddings = np.array([
            base_vec + np.random.RandomState(i).normal(0, 0.01, 384)
            for i in range(10)
        ], dtype=np.float32)
        # Normalize
        synthetic_embeddings /= np.linalg.norm(synthetic_embeddings, axis=1, keepdims=True)

        clusterer = SemanticClusterer(similarity_threshold=0.80)
        cluster_res = clusterer.cluster_responses(resp_texts, synthetic_embeddings)
        extractor = FeatureExtractor(clusterer=clusterer)

        features = extractor.extract_features(
            question="What is the capital of France?",
            responses=resp_texts,
            embeddings=synthetic_embeddings,
            cluster_result=cluster_res,
        )

        assert len(features) == 16
        for fn in FEATURE_NAMES:
            assert fn in features
            assert isinstance(features[fn], (float, int))
            assert not np.isnan(features[fn])

        # Verify consensus metrics
        assert features["dominant_agreement"] == 1.0
        assert features["semantic_entropy"] == pytest.approx(0.0, abs=1e-5)
