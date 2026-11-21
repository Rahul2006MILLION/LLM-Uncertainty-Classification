"""Schema definitions for questions, annotations, and experiment records."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class QuestionRecord(BaseModel):
    """Canonical schema for uncertainty benchmark questions."""

    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="The query string")
    label: Literal["KNOWN", "AMBIGUOUS", "UNKNOWN"] = Field(
        ..., description="Ground truth uncertainty class"
    )
    domain: str = Field(default="general", description="Subject area (e.g., geography, math, science, history)")
    subcategory: Optional[str] = Field(
        default=None, description="Detailed category (e.g. concept_definition, factual, entity_polysemy, criterion_ambiguity, false_premise)"
    )
    source: str = Field(..., description="Dataset or literature origin (e.g., ambignq, natural_questions, curated_unanswerable)")
    license: Optional[str] = Field(default="CC BY-SA 3.0", description="Data license")
    ground_truth_answers: Optional[List[str]] = Field(
        default=None, description="Known valid reference answers, if applicable"
    )
    interpretations: Optional[List[str]] = Field(
        default=None, description="Distinct plausible interpretations if ambiguous"
    )
    annotation_type: Literal["human_verified", "gold_benchmark", "provisional"] = Field(
        default="human_verified", description="Provenance of ground-truth label"
    )
    annotation_rationale: Optional[str] = Field(
        default=None, description="Scientific explanation of why this label applies"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extension metadata")


class ExperimentRecord(BaseModel):
    """Schema for individual QA pipeline execution results and metadata."""

    timestamp: str
    question: str
    path: Literal["FAST_PATH", "MULTI_RESPONSE"]
    final_answer: str
    response_count: int
    cluster_count: int
    dominant_cluster_agreement: float
    second_cluster_agreement: float
    agreement_margin: float
    semantic_entropy: float
    normalized_entropy: float
    mean_pairwise_similarity: float
    elapsed_seconds: float
    classifier_status: str
    predicted_label: Optional[str] = None
    confidence: Optional[float] = Field(
        default=None,
        description="Policy routing or classifier confidence score; represents decision certainty, NOT ground-truth factual verification."
    )
    features: Dict[str, float] = Field(default_factory=dict)
    responses: List[Dict[str, Any]] = Field(default_factory=list)
    clusters: Dict[str, Any] = Field(default_factory=dict)
    model: str
    embedding_model: str
    original_question: Optional[str] = None
    clarified_question: Optional[str] = None
    is_clarified: bool = False
    clarification_rounds: int = 0
    clarification_history: List[Dict[str, Any]] = Field(default_factory=list)
    clarification_decision: Optional[str] = None
    interaction_status: Optional[str] = Field(
        default=None,
        description="Outcome of HITL clarification interaction (NOT_REQUIRED, PENDING, RESOLVED, USER_DECLINED, MAX_ROUNDS_REACHED)",
    )
    ambiguity_type: Optional[str] = None
    clarification_reason: Optional[str] = None
