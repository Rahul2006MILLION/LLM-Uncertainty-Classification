"""Human-in-the-Loop (HITL) Uncertainty Resolution Engine.

Coordinates the 3-state uncertainty decision mechanism:
  1. ANSWERABLE: Question is sufficiently contextualized to proceed.
  2. CLARIFICATION_REQUIRED: Potentially answerable, but missing user context/criteria.
  3. ABSTAIN / INSUFFICIENT_INFORMATION: Inherently unresolvable, private, or unknowable.

Integrates Gate 1 (early route) and Gate 2 (post-consistency) with safeguards
against infinite loops, credential solicitation, and prompt injection.
"""

import re
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from src.classification.inaccessibility_assessor import InaccessibilityAssessor
from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
from src.classification.clarity_assessor import QuestionClarityAssessor
from src.classification.entity_ambiguity import EntityAmbiguityAssessor
from src.classification.sensitive_request import is_conceptual_question
from src.classification.hitl_prompts import (
    sanitize_user_input,
    is_user_declining,
    contains_sensitive_credentials,
    generate_fallback_clarification,
    CURATED_CLARIFICATIONS,
)


class ClarificationState(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    ABSTAIN = "ABSTAIN"
    UNRESOLVED = "UNRESOLVED"


class AmbiguityType(str, Enum):
    ENTITY_POLYSEMY = "ENTITY_POLYSEMY"
    SUBJECTIVE_CRITERIA = "SUBJECTIVE_CRITERIA"
    UNDERSPECIFIED_CONTEXT = "UNDERSPECIFIED_CONTEXT"
    MISSING_REFERENT = "MISSING_REFERENT"
    MISSING_COMPARISON = "MISSING_COMPARISON"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    MULTIPLE_INTERPRETATIONS = "MULTIPLE_INTERPRETATIONS"
    INACCESSIBLE_PRIVATE = "INACCESSIBLE_PRIVATE"
    INHERENTLY_UNKNOWABLE = "INHERENTLY_UNKNOWABLE"
    FALSE_PREMISE = "FALSE_PREMISE"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    NONE = "NONE"


def _clarity_ambiguity_type(clarity) -> "AmbiguityType":
    """Map a clarity assessment to the HITL ambiguity taxonomy.

    Polysemous named entities keep the established ENTITY_POLYSEMY type whichever
    component (clarity assessor or polysemy check) detected them first.
    """
    if any(str(e).startswith("polysemic_term:") for e in (clarity.evidence or [])):
        return AmbiguityType.ENTITY_POLYSEMY
    if clarity.ambiguity_type in AmbiguityType._member_map_:
        return AmbiguityType[clarity.ambiguity_type]
    return AmbiguityType.UNDERSPECIFIED_CONTEXT


@dataclass
class ClarificationDecision:
    """Decision produced by HITL assessment gates.
    
    IMPORTANT EPISTEMIC DISTINCTION:
    - 'confidence' represents POLICY ASSESSMENT CONFIDENCE (the system's confidence in its
      routing/classification decision based on observed linguistic signals, inaccessibility
      markers, or cluster dispersion). It DOES NOT represent factual truth, ground-truth
      correctness, or epistemic certainty of generated answers.
    - An 'ANSWERABLE' decision indicates that the query is structurally and contextually
      sufficient to produce an answer without user interrogation under the active decision policy;
      it does NOT imply the answer has been factually verified against ground truth.
    - 'response_consistency_score' (when applicable) quantifies empirical sample agreement
      (dominant cluster agreement ratio), which measures aleatoric generation stability, not
      veracity.
    """
    decision: ClarificationState
    ambiguity_type: AmbiguityType
    reason: str
    clarification_prompt: Optional[str] = None
    options: List[str] = field(default_factory=list)
    why_it_matters: str = ""
    can_user_resolve: bool = False
    confidence: float = 1.0
    response_consistency_score: Optional[float] = None
    interaction_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    original_question: str
    current_question: str
    round_count: int = 0
    max_rounds: int = 2
    is_clarified: bool = False
    user_declined: bool = False
    interaction_status: str = "NOT_REQUIRED"
    detected_ambiguity_type: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def can_clarify(self) -> bool:
        """Check whether another clarification round is permitted."""
        return self.round_count < self.max_rounds and not self.user_declined

    def add_turn(
        self,
        clarification_prompt: str,
        user_response: str,
        consolidated_question: str,
        ambiguity_type: str = "",
    ) -> None:
        """Record an executed clarification turn."""
        self.round_count += 1
        self.is_clarified = True
        self.interaction_status = "RESOLVED"
        if ambiguity_type:
            self.detected_ambiguity_type = ambiguity_type
        self.current_question = consolidated_question
        self.history.append({
            "round": self.round_count,
            "clarification_prompt": clarification_prompt,
            "user_response": user_response,
            "consolidated_question": consolidated_question,
            "ambiguity_type": ambiguity_type or self.detected_ambiguity_type,
        })

    def decline_clarification(self, reason: str = "User declined", ambiguity_type: str = "") -> None:
        """Record that user declined further clarification."""
        self.user_declined = True
        self.interaction_status = "USER_DECLINED"
        if ambiguity_type:
            self.detected_ambiguity_type = ambiguity_type
        self.history.append({
            "round": self.round_count,
            "action": "declined",
            "reason": reason,
            "ambiguity_type": self.detected_ambiguity_type,
        })


class HITLEngine:
    """Evaluates question answerability and orchestrates human clarification."""

    # Patterns for inherently unknowable future events (exact non-deterministic / chaotic outcomes)
    FUTURE_CONTINGENCY_PATTERNS = [
        r"\b(stock|shares|nasdaq|dow|s&p|crypto|bitcoin|ethereum)\b.*\b(exact\s+)?(price|close|worth|reach)\b.*\b(20[3-9]\d|tomorrow|next year)\b",
        r"\bwho\s+will\s+win\s+(the\s+)?(fifa\s+)?(world\s*cup|super\s*bowl|election|presidency|oscar|tournament)\s+(in\s+)?20[3-9]\d\b",
        r"\bwhat\s+will\s+(be\s+)?(the\s+)?(exact\s+|final\s+)*(score|scoreboard|lottery\s+numbers|winning\s+numbers)\b",
        r"\b(winning\s+lottery\s+numbers|winning\s+numbers\s+in\s+the\s+next)\b",
    ]

    # Probabilistic and projection qualifiers that should NOT be blanket-rejected as inherently unknowable
    PROBABILISTIC_QUALIFIERS = [
        r"\b(projection|projections|forecast|forecasts|estimate|estimates|outlook|probability|probabilities|consensus|expected\s+range|analyst\s+target|analyst\s+expectation)\b",
    ]

    # Patterns for presupposition failures and category errors
    PRESUPPOSITION_FAILURE_PATTERNS = [
        r"\b(who|what)\s+is\s+(the\s+)?(name\s+of\s+(the\s+)?)?(current\s+)?(king|queen|emperor|monarch)\s+of\s+(the\s+)?(united states|usa|france|germany|italy)\b",
        r"\b(who|what)\s+was\s+(the\s+)?(first\s+)?female\s+president\s+of\s+(the\s+)?(united states|usa)\b",
        r"\bhow\s+many\s+corners\s+does\s+a\s+(circle|sphere|oval)\s+have\b",
        r"\bwhat\s+is\s+the\s+boiling\s+point\s+of\s+(a\s+)?(pure\s+)?vacuum\b",
        r"\bwhy\s+do\s+airplanes\s+fly\s+by\s+flapping\b",
        r"\b((chemical\s+)?formula|composition)\s+of\s+dry\s+water\b",
    ]

    # Specific polysemic terms that have distinct, common human referents
    KNOWN_POLYSEMIC_TERMS = [
        r"\b(georgia)\b",
        r"\b(jordan)\b",
        r"\b(jaguar)\b",
        r"\b(mercury)\b",
        r"\b(bridge)\b",
        r"\b(columbia|victoria|alexandria|springfield)\b",
    ]

    CONTEXT_DEPENDENCE_MARKERS = (
        r"\b(it depends on|depends on (the|which|what|whether|your|how)|depending on (the|which|what|whether|your)|"
        r"(can|could|may|might) refer to|refers? to (several|multiple|different|both|either)|"
        r"(several|multiple|different) (meanings|interpretations|senses|referents|possible answers)|"
        r"varies (by|depending on|between) (country|state|jurisdiction|region|context)|"
        r"(if you mean|if you are referring to|are you referring to|do you mean))\b"
    )

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds

    def evaluate_early(
        self,
        question: str,
        initial_response: Optional[str] = None,
        router_triggers: Optional[List[str]] = None,
        conversation_state: Optional[ConversationState] = None,
    ) -> ClarificationDecision:
        """Gate 1: Evaluate question answerability prior to multi-response generation.
        
        Identifies whether a question has clear polysemy or missing criteria that the
        user can resolve, while preventing requests for private or unknowable information.
        """
        q_clean = question.strip()
        q_lower = q_clean.lower()
        resp_lower = (initial_response or "").strip().lower()

        # Check round ceiling
        if conversation_state and not conversation_state.can_clarify():
            ambig_type = (
                AmbiguityType(conversation_state.detected_ambiguity_type)
                if conversation_state.detected_ambiguity_type in AmbiguityType._value2member_map_
                else AmbiguityType.UNDERSPECIFIED_CONTEXT
            )
            if conversation_state.round_count >= conversation_state.max_rounds:
                return ClarificationDecision(
                    decision=ClarificationState.UNRESOLVED,
                    ambiguity_type=ambig_type,
                    reason="Maximum clarification rounds reached without ambiguity resolution.",
                    can_user_resolve=False,
                    confidence=0.85,
                    interaction_status="MAX_ROUNDS_REACHED",
                    metadata={"clarification_exhausted": True, "round_count": conversation_state.round_count, "interaction_status": "MAX_ROUNDS_REACHED"},
                )
            elif conversation_state.user_declined:
                return ClarificationDecision(
                    decision=ClarificationState.UNRESOLVED,
                    ambiguity_type=ambig_type,
                    reason="User declined further clarification; ambiguity remains unresolved.",
                    can_user_resolve=False,
                    confidence=0.85,
                    interaction_status="USER_DECLINED",
                    metadata={"user_declined": True, "interaction_status": "USER_DECLINED"},
                )

        # Check if question has already been contextualized via previous clarification
        if re.search(r"\[context:\s*[^\]]+\]", q_clean, flags=re.IGNORECASE):
            return ClarificationDecision(
                decision=ClarificationState.ANSWERABLE,
                ambiguity_type=AmbiguityType.NONE,
                reason="Question includes explicit user-provided context block. Proceeding to evaluation.",
                can_user_resolve=False,
            )

        # 1. HARD SAFEGUARD: Inaccessibility Assessor (private credentials, mental states, safe secrets)
        inacc_assess = InaccessibilityAssessor.assess(question=q_clean, responses=initial_response)
        if inacc_assess.is_inaccessible:
            return ClarificationDecision(
                decision=ClarificationState.ABSTAIN,
                ambiguity_type=AmbiguityType.INACCESSIBLE_PRIVATE,
                reason=f"Epistemically inaccessible private information ({inacc_assess.reason}). User clarification cannot resolve public answerability.",
                can_user_resolve=False,
                confidence=inacc_assess.confidence,
                metadata={"inaccessibility_evidence": inacc_assess.evidence},
            )

        # 2. HARD SAFEGUARD: Future contingencies & chaotic events
        # Probabilistic queries (projections, forecasts, expected ranges) are permitted to receive qualified answers.
        is_probabilistic = any(re.search(pat, q_lower) for pat in self.PROBABILISTIC_QUALIFIERS)
        if not is_probabilistic:
            for pat in self.FUTURE_CONTINGENCY_PATTERNS:
                if re.search(pat, q_lower):
                    return ClarificationDecision(
                        decision=ClarificationState.ABSTAIN,
                        ambiguity_type=AmbiguityType.INHERENTLY_UNKNOWABLE,
                        reason="Query seeks exact future contingent or chaotic events (e.g. specific stock close, lottery numbers) that are fundamentally unknowable. User cannot provide factual certainty.",
                        can_user_resolve=False,
                        confidence=0.95,
                    )

        # 3. HARD SAFEGUARD: Category errors & presupposition failures
        for pat in self.PRESUPPOSITION_FAILURE_PATTERNS:
            if re.search(pat, q_lower):
                return ClarificationDecision(
                    decision=ClarificationState.ABSTAIN,
                    ambiguity_type=AmbiguityType.FALSE_PREMISE,
                    reason="Query contains a false existential premise or category error (e.g., monarch of a republic, corners of a circle, boiling point of a vacuum). No affirmative factual answer exists.",
                    can_user_resolve=False,
                    confidence=0.95,
                )

        # 3.5. QUESTION CLARITY ASSESSMENT: Missing referents, missing comparison, or missing context
        clarity = QuestionClarityAssessor.assess(q_clean)
        if clarity.status == "AMBIGUOUS" and clarity.clarification_required:
            ambig_enum = _clarity_ambiguity_type(clarity)
            if conversation_state:
                conversation_state.detected_ambiguity_type = ambig_enum.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason=clarity.reason)
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=ambig_enum,
                reason=clarity.reason,
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=clarity.confidence,
            )

        # 4. RESOLVABLE AMBIGUITY: Lexical Polysemy & Name Collisions.
        #    Context-aware: "Michael Jordan", "the planet Mercury" or "capital of Jordan"
        #    already select a sense and must not trigger clarification.
        entity = EntityAmbiguityAssessor.assess(q_clean)
        if entity.is_ambiguous:
            pat = entity.term
            if conversation_state:
                conversation_state.detected_ambiguity_type = AmbiguityType.ENTITY_POLYSEMY.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason=f"Polysemic entity matched: {pat}")
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=AmbiguityType.ENTITY_POLYSEMY,
                reason=f"Question contains polysemic entity or name collision '{pat}'. User can specify intended referent.",
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=0.90,
            )

        # 5. RESOLVABLE AMBIGUITY: Subjective Superlatives without Objective Criteria
        has_superlative = any(re.search(pat, q_lower) for pat in CriterionAmbiguityAssessor.CRITERION_SUPERLATIVES)
        has_constraint = any(re.search(pat, q_lower) for pat in CriterionAmbiguityAssessor.OBJECTIVE_CRITERIA_CONSTRAINTS)
        if has_superlative and not has_constraint:
            if conversation_state:
                conversation_state.detected_ambiguity_type = AmbiguityType.SUBJECTIVE_CRITERIA.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason="Subjective superlative without criteria")
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=AmbiguityType.SUBJECTIVE_CRITERIA,
                reason="Question seeks superlative judgment without explicit evaluation criteria. User can specify constraints or use case.",
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=0.88,
            )

        # 6. Default: Question is answerable or warrants standard multi-response analysis
        return ClarificationDecision(
            decision=ClarificationState.ANSWERABLE,
            ambiguity_type=AmbiguityType.NONE,
            reason="Question appears sufficiently contextualized for direct processing.",
            can_user_resolve=False,
            confidence=0.80,
        )

    def evaluate_post_consistency(
        self,
        question: str,
        responses: List[str],
        features: Dict[str, float],
        cluster_result: Dict[str, Any],
        classifier_label: Optional[str] = None,
        inaccessibility_assessment: Optional[Any] = None,
        ambiguity_assessment: Optional[Any] = None,
        conversation_state: Optional[ConversationState] = None,
    ) -> ClarificationDecision:
        """Gate 2: Evaluate ambiguity after multi-response generation and feature extraction.
        
        Distinguishes legitimate distinct interpretations from stylistic refusal dispersion
        and hallucinated factual contradictions.
        """
        q_clean = question.strip()
        q_lower = q_clean.lower()

        # Check round ceiling
        metadata = {
            "classifier_label": classifier_label,
            "cluster_count": int(features.get("cluster_count", 1)),
            "dominant_agreement": float(features.get("dominant_agreement", 1.0)),
            "second_agreement": float(features.get("second_agreement", 0.0)),
            "semantic_entropy": float(features.get("semantic_entropy", 0.0)),
        }

        if conversation_state and not conversation_state.can_clarify():
            ambig_type = (
                AmbiguityType(conversation_state.detected_ambiguity_type)
                if conversation_state.detected_ambiguity_type in AmbiguityType._value2member_map_
                else AmbiguityType.UNDERSPECIFIED_CONTEXT
            )
            if conversation_state.round_count >= conversation_state.max_rounds:
                return ClarificationDecision(
                    decision=ClarificationState.UNRESOLVED,
                    ambiguity_type=ambig_type,
                    reason="Maximum clarification rounds reached without ambiguity resolution.",
                    can_user_resolve=False,
                    confidence=0.85,
                    interaction_status="MAX_ROUNDS_REACHED",
                    metadata={**metadata, "clarification_exhausted": True, "interaction_status": "MAX_ROUNDS_REACHED"},
                )
            elif conversation_state.user_declined:
                return ClarificationDecision(
                    decision=ClarificationState.UNRESOLVED,
                    ambiguity_type=ambig_type,
                    reason="User declined further clarification; ambiguity remains unresolved.",
                    can_user_resolve=False,
                    confidence=0.85,
                    interaction_status="USER_DECLINED",
                    metadata={**metadata, "user_declined": True, "interaction_status": "USER_DECLINED"},
                )

        # Priority 1: Inaccessibility check
        if inaccessibility_assessment and getattr(inaccessibility_assessment, "is_inaccessible", False):
            return ClarificationDecision(
                decision=ClarificationState.ABSTAIN,
                ambiguity_type=AmbiguityType.INACCESSIBLE_PRIVATE,
                reason=getattr(inaccessibility_assessment, "reason", "Inaccessible private information"),
                can_user_resolve=False,
                confidence=getattr(inaccessibility_assessment, "confidence", 0.90),
                metadata=metadata,
            )

        # Priority 2: Refusal variation across responses (stylistic refusal dispersion)
        refusal_count = 0
        for r in responses:
            r_low = r.lower()
            if any(re.search(pat, r_low) for pat in InaccessibilityAssessor.REFUSAL_INACCESSIBILITY_MARKERS):
                refusal_count += 1

        # Refusals that only reflect missing user context ("I don't know your location" for
        # "What is the speed limit?") are resolvable by clarification, not unknowable.
        clarity_pre = QuestionClarityAssessor.assess(q_clean)
        user_resolvable = (
            clarity_pre.status == "AMBIGUOUS"
            and clarity_pre.clarification_required
            and not (conversation_state and conversation_state.is_clarified)
        )
        # Explanations and procedures ("why should you never share your password") often
        # contain words such as "personal" or "confidential" without refusing anything.
        conceptual = is_conceptual_question(q_clean)
        if refusal_count >= max(2, int(0.4 * len(responses))) and not user_resolvable and not conceptual:
            return ClarificationDecision(
                decision=ClarificationState.ABSTAIN,
                ambiguity_type=AmbiguityType.INHERENTLY_UNKNOWABLE,
                reason="Multiple response clusters reflect stylistic variations of epistemic refusal, not competing interpretations.",
                can_user_resolve=False,
                confidence=0.88,
                metadata={**metadata, "refusal_count": refusal_count},
            )

        # Priority 3: Criterion ambiguity check (subjective superlatives lacking criteria)
        # If the question was already clarified with user context/criteria, do not re-prompt for criteria
        if ambiguity_assessment and getattr(ambiguity_assessment, "is_ambiguous", False) and not (conversation_state and conversation_state.is_clarified):
            if conversation_state:
                conversation_state.detected_ambiguity_type = AmbiguityType.SUBJECTIVE_CRITERIA.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason=getattr(ambiguity_assessment, "reason", ""))
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=AmbiguityType.SUBJECTIVE_CRITERIA,
                reason="Responses confirm question depends on unstated evaluation criteria or perspective.",
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=getattr(ambiguity_assessment, "confidence", 0.85),
                metadata=metadata,
            )

        # Priority 4: Empirical cluster divergence analysis
        k = int(features.get("cluster_count", 1))
        p2 = float(features.get("second_agreement", 0.0))

        # Classifier label or entropy ALONE must NEVER independently trigger clarification
        # Clarification requires BOTH multi-cluster divergence (k >= 2, p2 >= 0.25)
        # AND verifiable evidence of user-resolvable ambiguity (polysemy, context dependency, or criteria)
        if k >= 2 and p2 >= 0.25:
            context_match = re.search(r"\[context:\s*([^\]]+)\]", q_clean, flags=re.IGNORECASE)
            context_text = context_match.group(1).lower() if context_match else ""
            is_already_clarified = bool(conversation_state and conversation_state.is_clarified)

            # Check whether user-supplied context already resolved the polysemic entity
            is_polysemy_resolved = False
            if context_text:
                disambiguating_markers = [
                    r"\b(country|state|nation|republic|city|river|island|animal|car|vehicle|company|corporation|game|card|bridge|caucasus|atlanta|tbilisi)\b",
                ]
                if any(re.search(pat, context_text) for pat in disambiguating_markers):
                    is_polysemy_resolved = True

            has_polysemy = (not is_polysemy_resolved) and EntityAmbiguityAssessor.assess(q_clean).is_ambiguous
            # Explicit statements that the answer depends on the reading of the question.
            # Bare common words ("or", "area", "definition") appear in ordinary factual
            # answers and are not evidence of user-resolvable ambiguity.
            has_context_dependence = (not is_already_clarified) and any(
                re.search(self.CONTEXT_DEPENDENCE_MARKERS, r.lower()) for r in responses
            )
            has_superlative = (not is_already_clarified) and any(re.search(pat, q_lower) for pat in CriterionAmbiguityAssessor.CRITERION_SUPERLATIVES)

            # A question the clarity assessor already found user-resolvably ambiguous
            # (missing context / referent / comparison) explains the divergence: the
            # samples disagree because the question is underspecified, not because the
            # model contradicts itself on a factual question.
            if has_polysemy or has_context_dependence or has_superlative or user_resolvable:
                if has_polysemy:
                    ambig_type = AmbiguityType.ENTITY_POLYSEMY
                elif user_resolvable and not (has_context_dependence or has_superlative):
                    ambig_type = _clarity_ambiguity_type(clarity_pre)
                else:
                    ambig_type = AmbiguityType.UNDERSPECIFIED_CONTEXT
                if conversation_state:
                    conversation_state.detected_ambiguity_type = ambig_type.value
                    conversation_state.interaction_status = "PENDING"
                q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason="Multiple plausible interpretations found")
                return ClarificationDecision(
                    decision=ClarificationState.CLARIFICATION_REQUIRED,
                    ambiguity_type=ambig_type,
                    reason="Multi-response consistency analysis revealed distinct semantic clusters with contextual or lexical dependence. User clarification can isolate intended sense.",
                    clarification_prompt=q_prompt,
                    options=options,
                    why_it_matters=why,
                    can_user_resolve=True,
                    confidence=0.85,
                    metadata=metadata,
                )
            else:
                # Factual contradiction / model unreliability:
                # Multiple clusters on a factual query without lexical or contextual ambiguity markers.
                # Do NOT prompt the user with a generic clarification.
                return ClarificationDecision(
                    decision=ClarificationState.MODEL_DISAGREEMENT,
                    ambiguity_type=AmbiguityType.MODEL_DISAGREEMENT,
                    reason="Response divergence reflects factual inconsistency or model unreliability on a factual query, not user-resolvable ambiguity.",
                    can_user_resolve=False,
                    confidence=0.75,
                    metadata={**metadata, "model_disagreement": True},
                )

        # Priority 5: Inherent question ambiguity from clarity assessment
        clarity = QuestionClarityAssessor.assess(q_clean)
        if clarity.status == "AMBIGUOUS" and clarity.clarification_required and not (conversation_state and conversation_state.is_clarified):
            ambig_enum = _clarity_ambiguity_type(clarity)
            if conversation_state:
                conversation_state.detected_ambiguity_type = ambig_enum.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason=clarity.reason)
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=ambig_enum,
                reason=clarity.reason,
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=clarity.confidence,
                metadata=metadata,
            )

        # Priority 6: Consistency with classifier label
        # If classifier predicted AMBIGUOUS and question clarity is AMBIGUOUS on an unclarified query, trigger clarification
        if classifier_label == "AMBIGUOUS" and clarity.status == "AMBIGUOUS" and not (conversation_state and conversation_state.is_clarified):
            ambig_enum = (
                AmbiguityType[clarity.ambiguity_type]
                if clarity.ambiguity_type and clarity.ambiguity_type in AmbiguityType._member_map_
                else AmbiguityType.UNDERSPECIFIED_CONTEXT
            )
            if conversation_state:
                conversation_state.detected_ambiguity_type = ambig_enum.value
                conversation_state.interaction_status = "PENDING"
            q_prompt, why, options = generate_fallback_clarification(q_clean, ambiguity_reason="Query flagged as ambiguous by classifier")
            return ClarificationDecision(
                decision=ClarificationState.CLARIFICATION_REQUIRED,
                ambiguity_type=ambig_enum,
                reason="Classifier predicted ambiguity and query requires context resolution.",
                clarification_prompt=q_prompt,
                options=options,
                why_it_matters=why,
                can_user_resolve=True,
                confidence=0.80,
                metadata=metadata,
            )

        # Default: Confirmed answerable or single dominant cluster
        return ClarificationDecision(
            decision=ClarificationState.ANSWERABLE,
            ambiguity_type=AmbiguityType.NONE,
            reason="Response consistency indicates single dominant semantic cluster; answerable under current policy without factual verification.",
            can_user_resolve=False,
            confidence=0.85,
            response_consistency_score=float(features.get("dominant_agreement", 1.0)),
            metadata=metadata,
        )

    def apply_clarification(
        self,
        conversation_state: ConversationState,
        user_input: str,
        decision: Optional[ClarificationDecision] = None,
    ) -> Tuple[ConversationState, str, bool]:
        """Apply user clarification to create a consolidated query without losing original wording.
        
        Returns:
            (updated_state, consolidated_question, is_successful)
        """
        raw_text = (user_input or "").strip()

        # Capture ambiguity type from decision if present
        ambig_val = decision.ambiguity_type.value if decision and decision.ambiguity_type != AmbiguityType.NONE else (conversation_state.detected_ambiguity_type or "")
        if ambig_val:
            conversation_state.detected_ambiguity_type = ambig_val

        # 1. Check if user declined
        if not raw_text or is_user_declining(raw_text):
            conversation_state.decline_clarification(reason="User declined or skipped clarification.", ambiguity_type=conversation_state.detected_ambiguity_type or "")
            return conversation_state, conversation_state.original_question, False

        # 2. Check for sensitive credentials accidentally provided
        if contains_sensitive_credentials(raw_text):
            conversation_state.decline_clarification(reason="Sensitive credential pattern detected; aborted for security.", ambiguity_type=conversation_state.detected_ambiguity_type or "")
            return conversation_state, conversation_state.original_question, False

        # 3. Sanitize user input (prevents prompt injection and instruction tags)
        cleaned = sanitize_user_input(raw_text)

        # 4. Map numeric option selection if user provided option index
        if decision and decision.options and cleaned.isdigit():
            idx = int(cleaned) - 1
            if 0 <= idx < len(decision.options):
                chosen_opt = decision.options[idx]
                if "skip" in chosen_opt.lower() or "other" in chosen_opt.lower():
                    conversation_state.decline_clarification(reason="User selected Skip option.", ambiguity_type=conversation_state.detected_ambiguity_type or "")
                    return conversation_state, conversation_state.original_question, False
                cleaned = chosen_opt

        # 5. Build consolidated query preserving original wording as data
        consolidated = f"{conversation_state.original_question} [Context: {cleaned}]"

        # 6. Update state
        prompt_text = decision.clarification_prompt if decision else ""
        conversation_state.add_turn(
            clarification_prompt=prompt_text,
            user_response=cleaned,
            consolidated_question=consolidated,
            ambiguity_type=ambig_val,
        )

        return conversation_state, consolidated, True
