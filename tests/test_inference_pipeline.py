"""Integration and unit tests for the unified uncertainty inference pipeline.

Validates:
1. Valid input end-to-end execution
2. Empty input handling (exception and safe-mode)
3. Invalid input handling (non-string types)
4. Missing model error handling
5. Model loading failure on corrupted artifact
6. Response generation failure handling
7. Invalid/empty generated responses handling
8. Feature extraction failure handling
9. Output structure exact specification compliance
10. Model persistence: saving, loading, and inference without retraining
11. Existing serialized model artifact integrity
12. Ambiguity and inaccessibility heuristic safeguards
"""

import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.classification.classifier import UncertaintyClassifier
from src.features.feature_extractor import FEATURE_NAMES
from src.classification.inference_pipeline import (
    FeatureExtractionError,
    GenerationError,
    InferencePipeline,
    InvalidInputError,
    ModelNotLoadedError,
)
from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FeatureExtractor


@pytest.fixture
def trained_classifier():
    """Create and train a small synthetic classifier for fast isolated unit testing."""
    clf = UncertaintyClassifier(model_type="hist_gb", n_estimators=10, max_depth=3)
    rng = np.random.RandomState(42)
    # 30 samples, 16 features
    X = rng.randn(30, len(FEATURE_NAMES)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=int)
    clf.train(X, y, feature_names=FEATURE_NAMES)
    return clf


@pytest.fixture
def mock_generator():
    """Create a mock response generator that returns 10 synthetic answers."""
    gen = MagicMock()
    gen.generate_multi.return_value = [
        {"text": f"Synthetic response candidate {i} for Paris", "index": i}
        for i in range(10)
    ]
    return gen


@pytest.fixture
def mock_embedder():
    """Create a mock embedding engine returning normalized (10, 384) vectors."""
    emb = MagicMock()

    def _embed_batch(texts):
        n = len(texts)
        if n == 0:
            return np.empty((0, 384), dtype=np.float32)
        rng = np.random.RandomState(len(texts))
        vecs = rng.randn(n, 384).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)

    emb.embed_batch.side_effect = _embed_batch
    return emb


@pytest.fixture
def pipeline(trained_classifier, mock_generator, mock_embedder):
    """Create a ready-to-use inference pipeline."""
    return InferencePipeline(
        classifier=trained_classifier,
        generator=mock_generator,
        embedder=mock_embedder,
        raise_on_error=False,
    )


class TestInferencePipelineEndToEnd:
    """Tests normal and edge-case execution paths."""

    def test_valid_input_end_to_end(self, pipeline):
        """Pipeline successfully executes end-to-end on a valid question."""
        question = "What is the capital of France?"
        result = pipeline.predict(question)

        # 1. Output structure contract
        expected_keys = {
            "input_question",
            "predicted_label",
            "model_prediction",
            "probabilities",
            "confidence",
            "generated_responses",
            "extracted_features",
            "classification_metadata",
        }
        assert set(result.keys()) == expected_keys

        # 2. Values and types
        assert result["input_question"] == question
        assert result["predicted_label"] in {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

        # 3. Probabilities
        assert isinstance(result["probabilities"], dict)
        assert set(result["probabilities"].keys()) == {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4

        # 4. Responses & Features
        assert len(result["generated_responses"]) == 10
        assert set(result["extracted_features"].keys()) == set(FEATURE_NAMES)

        # 5. Metadata
        meta = result["classification_metadata"]
        assert meta["status"] in {"SUCCESS", "OVERRIDE_AMBIGUITY", "OVERRIDE_INACCESSIBILITY"}
        assert meta["num_responses_generated"] == 10
        assert meta["num_valid_responses"] == 10
        assert meta["num_clusters"] >= 1
        assert meta["execution_time_seconds"] >= 0.0
        assert "timestamp" in meta

    def test_correct_output_structure_contract(self, pipeline):
        """Strict verification of return dictionary schema and metadata schema."""
        res = pipeline.predict("What is the boiling point of water?")
        required_top_level = [
            "input_question",
            "predicted_label",
            "model_prediction",
            "probabilities",
            "confidence",
            "generated_responses",
            "extracted_features",
            "classification_metadata",
        ]
        for key in required_top_level:
            assert key in res, f"Missing required output key: {key}"

        meta = res["classification_metadata"]
        required_meta_keys = [
            "status",
            "model_type",
            "num_responses_generated",
            "num_valid_responses",
            "num_clusters",
            "execution_time_seconds",
            "timestamp",
        ]
        for key in required_meta_keys:
            assert key in meta, f"Missing required metadata key: {key}"


class TestInferencePipelineInputValidation:
    """Tests empty and invalid input handling."""

    def test_empty_input_raises_exception_when_configured(self, pipeline):
        """Empty string or whitespace-only raises InvalidInputError with raise_on_error=True."""
        with pytest.raises(InvalidInputError, match="empty or whitespace"):
            pipeline.predict("", raise_on_error=True)

        with pytest.raises(InvalidInputError, match="empty or whitespace"):
            pipeline.predict("   \n\t  ", raise_on_error=True)

    def test_empty_input_safe_mode(self, pipeline):
        """Empty input in safe mode returns clean error structure without throwing."""
        res = pipeline.predict("", raise_on_error=False)
        assert res["predicted_label"] is None
        assert res["confidence"] == 0.0
        assert res["probabilities"] is None
        assert res["generated_responses"] == []
        assert res["classification_metadata"]["status"] == "ERROR_EMPTY_INPUT"
        assert "empty or whitespace" in res["classification_metadata"]["error"]

    def test_invalid_input_type_raises_when_configured(self, pipeline):
        """Non-string inputs raise InvalidInputError when raise_on_error=True."""
        with pytest.raises(InvalidInputError, match="must be a string"):
            pipeline.predict(12345, raise_on_error=True)

        with pytest.raises(InvalidInputError, match="must be a string"):
            pipeline.predict(None, raise_on_error=True)

        with pytest.raises(InvalidInputError, match="must be a string"):
            pipeline.predict({"question": "test"}, raise_on_error=True)

    def test_invalid_input_type_safe_mode(self, pipeline):
        """Non-string inputs in safe mode return clean error structure."""
        res = pipeline.predict(12345, raise_on_error=False)
        assert res["predicted_label"] is None
        assert res["classification_metadata"]["status"] == "ERROR_INVALID_INPUT"
        assert "must be a string" in res["classification_metadata"]["error"]


class TestInferencePipelineModelHandling:
    """Tests model loading, missing models, and persistence."""

    def test_missing_model_raises_when_configured(self, mock_generator, mock_embedder):
        """Missing model file raises ModelNotLoadedError when raise_on_error=True."""
        with pytest.raises(ModelNotLoadedError, match="Failed to load classifier"):
            InferencePipeline(
                classifier=None,
                model_path="/tmp/nonexistent_model_xyz_9876.joblib",
                generator=mock_generator,
                embedder=mock_embedder,
                raise_on_error=True,
            )

    def test_missing_model_safe_mode(self, mock_generator, mock_embedder):
        """Missing model in safe mode initializes pipeline and fails safely at inference."""
        pipe = InferencePipeline(
            classifier=None,
            model_path="/tmp/nonexistent_model_xyz_9876.joblib",
            generator=mock_generator,
            embedder=mock_embedder,
            raise_on_error=False,
        )
        assert pipe.classifier is None
        assert pipe.model_load_error is not None

        res = pipe.predict("What is 2 + 2?")
        assert res["predicted_label"] is None
        assert res["confidence"] == 0.0
        assert res["classification_metadata"]["status"] == "ERROR_MISSING_MODEL"

    def test_model_loading_failure_corrupted_file(self, tmp_path, mock_generator, mock_embedder):
        """Corrupted model file raises on raise_on_error=True and handles safely on False."""
        corrupt_path = tmp_path / "corrupt_model.joblib"
        corrupt_path.write_bytes(b"NOT_A_VALID_JOBLIB_OR_PICKLE_STREAM_12345")

        with pytest.raises(ModelNotLoadedError):
            InferencePipeline(
                classifier=None,
                model_path=str(corrupt_path),
                generator=mock_generator,
                embedder=mock_embedder,
                raise_on_error=True,
            )

        pipe = InferencePipeline(
            classifier=None,
            model_path=str(corrupt_path),
            generator=mock_generator,
            embedder=mock_embedder,
            raise_on_error=False,
        )
        assert pipe.classifier is None
        assert pipe.model_load_error is not None

    def test_model_persistence_save_load_infer_without_retraining(self, tmp_path, mock_generator, mock_embedder):
        """Verify saving a trained model, loading it back, and using it for inference without retraining."""
        # 1. Train model
        original_clf = UncertaintyClassifier(model_type="hist_gb", n_estimators=10)
        rng = np.random.RandomState(99)
        X = rng.randn(30, len(FEATURE_NAMES)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=int)
        original_clf.train(X, y, feature_names=FEATURE_NAMES)

        # 2. Save model
        save_path = str(tmp_path / "saved_test_classifier.joblib")
        original_clf.save(save_path)
        assert os.path.exists(save_path)

        # 3. Load model without retraining
        loaded_clf = UncertaintyClassifier.load(save_path)
        assert loaded_clf.is_trained is True
        assert loaded_clf.feature_names == FEATURE_NAMES

        # 4. Run inference with loaded model
        pipe = InferencePipeline(
            classifier=loaded_clf,
            generator=mock_generator,
            embedder=mock_embedder,
        )
        res = pipe.predict("What is the atomic weight of gold?")
        assert res["predicted_label"] in {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
        assert res["probabilities"] is not None
        assert res["confidence"] > 0.0

    def test_existing_repo_serialized_model_persistence(self, mock_generator, mock_embedder):
        """Verify the repository's primary serialized model artifact loads and performs inference."""
        repo_model_path = "models/uncertainty_classifier.joblib"
        if not os.path.exists(repo_model_path):
            pytest.skip("Repository serialized model artifact not found at models/uncertainty_classifier.joblib")

        pipe = InferencePipeline(
            model_path=repo_model_path,
            generator=mock_generator,
            embedder=mock_embedder,
            raise_on_error=True,
        )
        assert pipe.classifier is not None
        assert pipe.classifier.is_trained is True

        res = pipe.predict("What is the capital of Japan?")
        assert res["predicted_label"] in {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
        assert res["classification_metadata"]["status"] in {
            "SUCCESS",
            "OVERRIDE_AMBIGUITY",
            "OVERRIDE_INACCESSIBILITY",
        }


class TestInferencePipelineFailureResilience:
    """Tests graceful failure handling for generation, embedding, and clustering."""

    def test_response_generation_failure_raises(self, pipeline):
        """Generator exception raises GenerationError when raise_on_error=True."""
        pipeline.generator.generate_multi.side_effect = RuntimeError("Ollama connection timeout")
        with pytest.raises(GenerationError, match="Response generation failed"):
            pipeline.predict("Valid query?", raise_on_error=True)

    def test_response_generation_failure_safe_mode(self, pipeline):
        """Generator exception returns structured error response when raise_on_error=False."""
        pipeline.generator.generate_multi.side_effect = RuntimeError("Ollama connection timeout")
        res = pipeline.predict("Valid query?", raise_on_error=False)

        assert res["predicted_label"] is None
        assert res["confidence"] == 0.0
        assert res["classification_metadata"]["status"] == "ERROR_GENERATION_FAILED"
        assert "timeout" in res["classification_metadata"]["error"]

    def test_invalid_generated_responses_empty_strings(self, pipeline):
        """Generator returning only empty or whitespace responses triggers validation error."""
        pipeline.generator.generate_multi.side_effect = None
        pipeline.generator.generate_multi.return_value = [{"text": ""}, {"text": "   "}, {"text": "\n"}]

        with pytest.raises(GenerationError, match="No valid non-empty responses"):
            pipeline.predict("Valid query?", raise_on_error=True)

        res = pipeline.predict("Valid query?", raise_on_error=False)
        assert res["predicted_label"] is None
        assert res["classification_metadata"]["status"] == "ERROR_INVALID_RESPONSES"
        assert len(res["generated_responses"]) == 3

    def test_feature_extraction_failure_raises(self, pipeline):
        """Embedding failure raises FeatureExtractionError when raise_on_error=True."""
        pipeline.embedder.embed_batch.side_effect = ValueError("Incompatible vector dimension")
        with pytest.raises(FeatureExtractionError, match="Semantic embedding computation failed"):
            pipeline.predict("Valid query?", raise_on_error=True)

    def test_feature_extraction_failure_safe_mode(self, pipeline):
        """Embedding failure returns clean error structure when raise_on_error=False."""
        pipeline.embedder.embed_batch.side_effect = ValueError("Incompatible vector dimension")
        res = pipeline.predict("Valid query?", raise_on_error=False)

        assert res["predicted_label"] is None
        assert res["confidence"] == 0.0
        assert res["classification_metadata"]["status"] == "ERROR_FEATURE_EXTRACTION_FAILED"
        assert "vector dimension" in res["classification_metadata"]["error"]


class TestInferencePipelineSafeguards:
    """Tests heuristic overrides for unconstrained criteria and inaccessible information."""

    def test_criterion_ambiguity_override(self, pipeline):
        """Subjective superlative question with split answers is safely classified as AMBIGUOUS."""
        question = "Who is the greatest scientist of all time?"
        # Simulate divergent responses
        pipeline.generator.generate_multi.side_effect = None
        pipeline.generator.generate_multi.return_value = [
            {"text": "Albert Einstein is widely regarded as the greatest scientist."},
            {"text": "Isaac Newton is the greatest scientist due to mechanics and calculus."},
            {"text": "Marie Curie is considered by many as the greatest scientist."},
            {"text": "Albert Einstein for general relativity."},
            {"text": "Isaac Newton for gravity."},
        ] * 2

        res = pipeline.predict(question)
        assert res["predicted_label"] == "AMBIGUOUS"
        assert "AMBIGUITY" in res["classification_metadata"]["status"]

    def test_inaccessibility_override(self, pipeline):
        """Query demanding private inaccessible state is safely classified as UNKNOWN."""
        question = "What number am I thinking of right now?"
        pipeline.generator.generate_multi.side_effect = None
        pipeline.generator.generate_multi.return_value = [
            {"text": "I cannot read your mind or know what you are thinking."},
            {"text": "I do not have access to your thoughts."},
            {"text": "As an AI, I cannot know what number you are thinking of."},
        ] * 4

        res = pipeline.predict(question)
        assert res["predicted_label"] == "UNKNOWN"
        assert "INACCESSIBILITY" in res["classification_metadata"]["status"]
