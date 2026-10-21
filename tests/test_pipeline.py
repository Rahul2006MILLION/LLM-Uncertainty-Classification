"""Unit tests for pipeline components: routing, multi-response count, and error handling."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.classification.router import AdaptiveRouter
from src.generation.generator import ResponseGenerator
from src.generation.ollama_client import OllamaClient, OllamaConnectionError


class TestAdaptiveRouter:
    """Tests routing behavior for fast path and multi-response."""

    def test_arithmetic_routed_to_fast_path(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        decision = router.route("What is 2 + 2?", "4")
        assert decision.path == "FAST_PATH"

    def test_concept_definition_routed_to_fast_path(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        decision = router.route("What is Data Science?", "Data science is an interdisciplinary field.")
        assert decision.path == "FAST_PATH"

    def test_ambiguous_question_routed_to_multi_response(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        decision = router.route("Who is Jordan?", "Jordan is a country.")
        assert decision.path == "MULTI_RESPONSE"

    def test_criterion_ambiguity_routed_to_multi_response(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        decision = router.route("What is the largest city in the world?", "Tokyo by population, New York by area.")
        assert decision.path == "MULTI_RESPONSE"

    def test_unanswerable_question_routed_to_multi_response(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        decision = router.route("What number am I thinking of right now?", "I cannot know your thoughts.")
        assert decision.path == "MULTI_RESPONSE"


class TestResponseGenerator:
    """Tests generator multi-response exact counting and error resilience."""

    def test_exact_ten_responses_generated(self):
        """Must return exactly 10 responses total when target_total_count=10."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value = {
            "response": "Sample answer",
            "eval_count": 5,
            "total_duration_ms": 100,
        }

        gen = ResponseGenerator(client=mock_client)
        responses = gen.generate_multi("What is the capital of France?", target_total_count=10)

        assert len(responses) == 10
        assert all(r["index"] == i for i, r in enumerate(responses))
        # 1 initial + 9 additional generations
        assert mock_client.generate.call_count == 10

    def test_graceful_handling_of_empty_or_failed_generation(self):
        """Network drop during sequential generation must not crash pipeline."""
        mock_client = MagicMock(spec=OllamaClient)
        # Succeeds for first 2, then raises connection error
        mock_client.generate.side_effect = [
            {"response": "Initial", "eval_count": 2},
            {"response": "Second", "eval_count": 2},
            OllamaConnectionError("Network timeout"),
            {"response": "Fourth", "eval_count": 2},
            {"response": "Fifth", "eval_count": 2},
            {"response": "Sixth", "eval_count": 2},
            {"response": "Seventh", "eval_count": 2},
            {"response": "Eighth", "eval_count": 2},
            {"response": "Ninth", "eval_count": 2},
            {"response": "Tenth", "eval_count": 2},
        ]

        gen = ResponseGenerator(client=mock_client)
        responses = gen.generate_multi("Test question", target_total_count=10)

        assert len(responses) == 10
        # The failed generation is recorded safely
        assert responses[2]["text"] == ""
        assert "error" in responses[2]


class TestPipelineRegression:
    """Regression tests for end-to-end pipeline defects."""

    def test_fast_path_does_not_hardcode_known(self, monkeypatch):
        """Regression test for Defect 1: Fast path must invoke classifier, not hardcode 'KNOWN'."""
        from experiments.ask import run_pipeline
        from src.classification.classifier import UncertaintyClassifier

        # Create a trained classifier that predicts 'AMBIGUOUS' for everything
        mock_clf = MagicMock(spec=UncertaintyClassifier)
        mock_clf.is_trained = True
        mock_clf.predict.return_value = {
            "predicted_label": "AMBIGUOUS",
            "confidence": 0.88,
            "status": "PREDICTED",
            "predicted_class_id": 1,
        }

        # Mock generator and embedder to avoid live ollama calls
        mock_gen = MagicMock()
        mock_gen.generate_single.return_value = {
            "text": "4",
            "eval_count": 1,
            "total_duration_ms": 10,
        }
        mock_emb = MagicMock()
        mock_emb.embed_batch.return_value = np.zeros((1, 384), dtype=np.float32)

        config = {
            "routing": {"fast_path_enabled": True},
            "generation": {"total_responses": 10},
            "classifier": {"model_path": "dummy.joblib"},
        }

        record = run_pipeline(
            question="What is 2 + 2?",
            config=config,
            generator=mock_gen,
            classifier=mock_clf,
            embedder=mock_emb,
        )

        assert record.path == "FAST_PATH"
        # The predicted label must come from the classifier, NOT hardcoded 'KNOWN'
        assert record.predicted_label == "AMBIGUOUS"
        assert record.confidence == 0.88
        assert mock_clf.predict.called

    def test_untrained_classifier_reports_not_trained(self):
        """Pipeline must report CLASSIFIER NOT TRAINED when no trained model is loaded."""
        from experiments.ask import run_pipeline
        from src.classification.classifier import UncertaintyClassifier

        mock_clf = UncertaintyClassifier()
        mock_gen = MagicMock()
        mock_gen.generate_single.return_value = {
            "text": "4",
            "eval_count": 1,
            "total_duration_ms": 10,
        }
        mock_emb = MagicMock()
        mock_emb.embed_batch.return_value = np.zeros((1, 384), dtype=np.float32)

        config = {
            "routing": {"fast_path_enabled": True},
            "generation": {"total_responses": 10},
            "classifier": {"model_path": "dummy.joblib"},
        }

        record = run_pipeline(
            question="What is 2 + 2?",
            config=config,
            generator=mock_gen,
            classifier=mock_clf,
            embedder=mock_emb,
        )

        assert record.classifier_status == "CLASSIFIER NOT TRAINED"
        assert record.predicted_label is None
        assert record.confidence is None
