"""Production-grade unified inference pipeline for LLM uncertainty classification.

This module provides a clean programmatic interface that coordinates:
Question -> Response Generation -> Response Validation -> Semantic Embedding ->
Clustering & Consistency -> Feature Vector Extraction -> Classifier Inference ->
Rule/Heuristic Safeguards -> Standardized Output (KNOWN / AMBIGUOUS / UNKNOWN).
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
from src.classification.classifier import ID_TO_LABEL, LABEL_TO_ID, UncertaintyClassifier
from src.classification.inaccessibility_assessor import InaccessibilityAssessor
from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FEATURE_NAMES, FeatureExtractor
from src.generation.generator import ResponseGenerator
from src.generation.ollama_client import OllamaClient
from src.preprocessing.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class InferencePipelineError(Exception):
    """Base exception for inference pipeline execution failures."""
    pass


class InvalidInputError(InferencePipelineError, ValueError):
    """Raised when the input question is invalid or empty."""
    pass


class ModelNotLoadedError(InferencePipelineError, RuntimeError):
    """Raised when the classifier model is missing or not trained."""
    pass


class GenerationError(InferencePipelineError, RuntimeError):
    """Raised when response generation fails."""
    pass


class FeatureExtractionError(InferencePipelineError, RuntimeError):
    """Raised when embedding or feature extraction fails."""
    pass


class InferencePipeline:
    """Unified multi-response consistency and uncertainty classification pipeline.

    Executes the standard research and production workflow:
    1. Validate input question
    2. Generate stochastic responses (default: 10 responses)
    3. Validate generated responses (ensure valid non-empty texts)
    4. Compute semantic embeddings
    5. Perform agglomerative clustering and consistency analysis
    6. Extract the canonical 16-feature vector
    7. Predict uncertainty class (KNOWN, AMBIGUOUS, UNKNOWN) via trained model
    8. Apply rule/heuristic safeguards (Criterion Ambiguity & Inaccessibility)
    9. Return comprehensive structured output dictionary
    """

    def __init__(
        self,
        classifier: Optional[UncertaintyClassifier] = None,
        generator: Optional[ResponseGenerator] = None,
        embedder: Optional[EmbeddingEngine] = None,
        clusterer: Optional[SemanticClusterer] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        model_path: Optional[str] = "models/uncertainty_classifier.joblib",
        num_responses: int = 10,
        similarity_threshold: float = 0.80,
        linkage: str = "average",
        apply_heuristics: bool = True,
        raise_on_error: bool = False,
    ):
        """Initialize the inference pipeline with modular components.

        Args:
            classifier: Pre-trained UncertaintyClassifier instance.
            generator: ResponseGenerator instance for multi-response sampling.
            embedder: EmbeddingEngine instance for response embeddings.
            clusterer: SemanticClusterer instance for grouping responses.
            feature_extractor: FeatureExtractor instance for canonical 16 features.
            model_path: Filepath to load serialized classifier if classifier is None.
            num_responses: Target number of stochastic responses to sample.
            similarity_threshold: Cosine similarity threshold for clustering.
            linkage: Linkage criterion for agglomerative clustering.
            apply_heuristics: Whether to apply ambiguity & inaccessibility overrides.
            raise_on_error: If True, raises exceptions on errors; if False, returns
                structured error responses with error metadata.
        """
        self.num_responses = num_responses
        self.similarity_threshold = similarity_threshold
        self.linkage = linkage
        self.apply_heuristics = apply_heuristics
        self.raise_on_error = raise_on_error
        self.model_load_error: Optional[str] = None

        # 1. Classifier setup
        self.classifier = classifier
        if self.classifier is None and model_path is not None:
            self._try_load_model(model_path)

        # 2. Sub-components setup
        self.generator = generator
        self.embedder = embedder
        self.clusterer = clusterer or SemanticClusterer(
            similarity_threshold=self.similarity_threshold,
            linkage=self.linkage,
        )
        self.feature_extractor = feature_extractor or FeatureExtractor(clusterer=self.clusterer)

    def _try_load_model(self, model_path: str) -> None:
        """Attempt to load classifier model from disk."""
        try:
            self.classifier = UncertaintyClassifier.load(model_path)
            self.model_load_error = None
        except Exception as e:
            self.classifier = None
            self.model_load_error = str(e)
            if self.raise_on_error:
                raise ModelNotLoadedError(f"Failed to load classifier from {model_path}: {e}") from e

    def load_model(self, model_path: str) -> UncertaintyClassifier:
        """Explicitly load a trained classifier from disk.

        Args:
            model_path: Path to serialized model (.joblib or .json).

        Returns:
            The loaded UncertaintyClassifier instance.
        """
        try:
            self.classifier = UncertaintyClassifier.load(model_path)
            self.model_load_error = None
            return self.classifier
        except Exception as e:
            self.classifier = None
            self.model_load_error = str(e)
            raise ModelNotLoadedError(f"Failed to load model from {model_path}: {e}") from e

    def _build_error_response(
        self,
        question: Any,
        status: str,
        error_message: str,
        responses: Optional[List[str]] = None,
        features: Optional[Dict[str, float]] = None,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Construct standard error output dictionary with all required keys."""
        elapsed = round(time.time() - start_time, 4) if start_time else 0.0
        return {
            "input_question": str(question) if question is not None else "",
            "predicted_label": None,
            "model_prediction": None,
            "probabilities": None,
            "confidence": 0.0,
            "generated_responses": responses or [],
            "extracted_features": features or {},
            "classification_metadata": {
                "status": status,
                "error": error_message,
                "model_type": getattr(self.classifier, "model_type", None),
                "num_responses_generated": len(responses or []),
                "num_valid_responses": 0,
                "num_clusters": 0,
                "execution_time_seconds": elapsed,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        }

    def predict(
        self,
        question: str,
        num_responses: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end uncertainty classification on a single input question.

        Args:
            question: The natural language question to classify.
            num_responses: Optional override for number of responses to generate.
            raise_on_error: Optional override for exception behavior.

        Returns:
            Dictionary containing:
                - input_question (str)
                - predicted_label (str or None)
                - model_prediction (dict or None)
                - probabilities (dict or None)
                - confidence (float)
                - generated_responses (list of str)
                - extracted_features (dict of str -> float)
                - classification_metadata (dict)
        """
        start_time = time.time()
        effective_raise = self.raise_on_error if raise_on_error is None else raise_on_error
        target_count = num_responses if num_responses is not None else self.num_responses

        # -------------------------------------------------------------
        # Step 1: Input Validation
        # -------------------------------------------------------------
        if not isinstance(question, str):
            err_msg = f"Input question must be a string, got {type(question).__name__}"
            if effective_raise:
                raise InvalidInputError(err_msg)
            return self._build_error_response(
                question=question,
                status="ERROR_INVALID_INPUT",
                error_message=err_msg,
                start_time=start_time,
            )

        cleaned_question = question.strip()
        if not cleaned_question:
            err_msg = "Input question cannot be empty or whitespace."
            if effective_raise:
                raise InvalidInputError(err_msg)
            return self._build_error_response(
                question=question,
                status="ERROR_EMPTY_INPUT",
                error_message=err_msg,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 2: Model Readiness Validation
        # -------------------------------------------------------------
        if self.classifier is None or not getattr(self.classifier, "is_trained", False):
            err_msg = (
                f"Classifier model is missing or not trained. "
                f"Load error: {self.model_load_error or 'No model loaded.'}"
            )
            if effective_raise:
                raise ModelNotLoadedError(err_msg)
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_MISSING_MODEL",
                error_message=err_msg,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 3: Response Generation
        # -------------------------------------------------------------
        generator = self.generator
        if generator is None:
            try:
                client = OllamaClient()
                generator = ResponseGenerator(client=client)
            except Exception as e:
                err_msg = f"Failed to initialize response generator: {e}"
                if effective_raise:
                    raise GenerationError(err_msg) from e
                return self._build_error_response(
                    question=cleaned_question,
                    status="ERROR_GENERATION_FAILED",
                    error_message=err_msg,
                    start_time=start_time,
                )

        try:
            raw_responses = generator.generate_multi(
                question=cleaned_question,
                target_total_count=target_count,
            )
        except Exception as e:
            err_msg = f"Response generation failed: {e}"
            if effective_raise:
                raise GenerationError(err_msg) from e
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_GENERATION_FAILED",
                error_message=err_msg,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 4: Response Validation
        # -------------------------------------------------------------
        if not isinstance(raw_responses, list):
            err_msg = f"Generator returned non-list output: {type(raw_responses).__name__}"
            if effective_raise:
                raise GenerationError(err_msg)
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_GENERATION_FAILED",
                error_message=err_msg,
                start_time=start_time,
            )

        response_texts: List[str] = []
        for r in raw_responses:
            if isinstance(r, str):
                response_texts.append(r)
            elif isinstance(r, dict):
                response_texts.append(str(r.get("text", "")))
            else:
                response_texts.append(str(r))

        valid_texts = [t for t in response_texts if t.strip()]
        if not valid_texts:
            err_msg = "No valid non-empty responses generated."
            if effective_raise:
                raise GenerationError(err_msg)
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_INVALID_RESPONSES",
                error_message=err_msg,
                responses=response_texts,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 5: Semantic Embedding
        # -------------------------------------------------------------
        embedder = self.embedder
        if embedder is None:
            try:
                embedder = EmbeddingEngine()
            except Exception as e:
                err_msg = f"Failed to initialize embedding engine: {e}"
                if effective_raise:
                    raise FeatureExtractionError(err_msg) from e
                return self._build_error_response(
                    question=cleaned_question,
                    status="ERROR_FEATURE_EXTRACTION_FAILED",
                    error_message=err_msg,
                    responses=response_texts,
                    start_time=start_time,
                )

        try:
            embeddings = embedder.embed_batch(valid_texts)
            if not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != len(valid_texts):
                raise ValueError(
                    f"Embedding dimension mismatch: expected {len(valid_texts)} rows, "
                    f"got {getattr(embeddings, 'shape', None)}"
                )
        except Exception as e:
            err_msg = f"Semantic embedding computation failed: {e}"
            if effective_raise:
                raise FeatureExtractionError(err_msg) from e
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_FEATURE_EXTRACTION_FAILED",
                error_message=err_msg,
                responses=response_texts,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 6: Clustering & Feature Extraction
        # -------------------------------------------------------------
        try:
            cluster_result = self.clusterer.cluster_responses(valid_texts, embeddings)
            extracted_features = self.feature_extractor.extract_features(
                question=cleaned_question,
                responses=valid_texts,
                embeddings=embeddings,
                cluster_result=cluster_result,
            )
        except Exception as e:
            err_msg = f"Clustering or feature extraction failed: {e}"
            if effective_raise:
                raise FeatureExtractionError(err_msg) from e
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_FEATURE_EXTRACTION_FAILED",
                error_message=err_msg,
                responses=response_texts,
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # Step 7: Classifier Inference
        # -------------------------------------------------------------
        try:
            pred_res = self.classifier.predict(extracted_features)
            model_label = pred_res.get("predicted_label")
            probabilities = pred_res.get("probabilities")
            confidence = float(pred_res.get("confidence", 0.0))
        except Exception as e:
            err_msg = f"Classifier inference failed: {e}"
            if effective_raise:
                raise InferencePipelineError(err_msg) from e
            return self._build_error_response(
                question=cleaned_question,
                status="ERROR_INFERENCE_FAILED",
                error_message=err_msg,
                responses=response_texts,
                features=extracted_features,
                start_time=start_time,
            )

        final_label = model_label
        final_confidence = confidence
        override_reason: Optional[str] = None
        pipeline_status = "SUCCESS"

        # -------------------------------------------------------------
        # Step 8: Criterion Ambiguity & Inaccessibility Safeguards
        # -------------------------------------------------------------
        if self.apply_heuristics:
            try:
                # Ambiguity Safeguard: unconstrained subjective superlatives
                ambig_check = CriterionAmbiguityAssessor.assess(
                    question=cleaned_question,
                    responses=valid_texts,
                    features=extracted_features,
                )
                if ambig_check.is_ambiguous and final_label != "AMBIGUOUS":
                    final_label = "AMBIGUOUS"
                    final_confidence = float(ambig_check.confidence)
                    override_reason = (
                        ambig_check.evidence[0] if ambig_check.evidence else "Criterion ambiguity"
                    )
                    pipeline_status = f"OVERRIDE_AMBIGUITY ({override_reason})"

                # Inaccessibility Safeguard: private thoughts/future unknowable queries
                inacc_check = InaccessibilityAssessor.assess(
                    question=cleaned_question,
                    responses=valid_texts,
                    features=extracted_features,
                )
                if inacc_check.is_inaccessible:
                    final_label = "UNKNOWN"
                    final_confidence = float(inacc_check.confidence)
                    override_reason = (
                        inacc_check.evidence[0] if inacc_check.evidence else "Epistemic inaccessibility"
                    )
                    pipeline_status = f"OVERRIDE_INACCESSIBILITY ({override_reason})"
            except Exception as e:
                logger.warning("Heuristic safeguard evaluation warning: %s", e)

        # -------------------------------------------------------------
        # Step 9: Assemble Standard Output Dictionary
        # -------------------------------------------------------------
        elapsed_seconds = round(time.time() - start_time, 4)
        metadata = {
            "status": pipeline_status,
            "model_type": getattr(self.classifier, "model_type", "unknown"),
            "num_responses_generated": len(response_texts),
            "num_valid_responses": len(valid_texts),
            "num_clusters": int(cluster_result.get("cluster_count", len(cluster_result.get("cluster_counts", [])))),
            "cluster_counts": cluster_result.get("cluster_counts", []),
            "heuristic_override": override_reason,
            "execution_time_seconds": elapsed_seconds,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        return {
            "input_question": cleaned_question,
            "predicted_label": final_label,
            "model_prediction": pred_res,
            "probabilities": probabilities,
            "confidence": final_confidence,
            "generated_responses": response_texts,
            "extracted_features": extracted_features,
            "classification_metadata": metadata,
        }
