"""Ambiguity assessment decoupling response consensus from objective truth."""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class AmbiguityAssessment:
    is_ambiguous: bool
    reason: str
    confidence: float  # Rule-based heuristic score (not an empirical calibrated probability)
    confidence_type: str = "rule_based_heuristic"
    evidence: List[str] = field(default_factory=list)


class CriterionAmbiguityAssessor:
    """Evaluates question formulation and response evidence to assess criterion ambiguity.
    
    Principles:
    1. Response Agreement != Objective Answerability:
       High response consistency or low semantic entropy across LLM samples indicates
       that the model has a common prior or cultural default. It does NOT establish
       that a single determinate truth exists when the question lacks evaluation criteria.
    2. Negative Formulations in Subjective Questions != UNKNOWN:
       Phrases such as 'There is no single best...' reflect polysemic/subjective criteria
       (AMBIGUOUS), not unknowable private states or future contingencies (UNKNOWN).
    """

    CRITERION_SUPERLATIVES = [
        r"\b(most\s+(successful|important|influential|significant|popular|valuable|beautiful|powerful|useful))\b",
        r"\b(greatest|finest|worst|favorite|best)\b",
    ]

    SUBJECTIVITY_RESPONSE_MARKERS = [
        r"\bsubjective\b",
        r"\bno single (best|greatest|most|definitive|universally accepted)\b",
        r"\bdepends on (the\s+)?(criteria|task|context|definition|perspective)\b",
        r"\bno universally accepted\b",
        r"\bvaries (greatly\s+)?based on\b",
        r"\bmatter of (opinion|perspective)\b",
        r"\bdifferent criteria\b",
    ]

    # Explicit objective constraints that specify criteria, rendering a superlative answerable
    OBJECTIVE_CRITERIA_CONSTRAINTS = [
        r"\bby (population|area|land area|gdp|revenue|net worth|height|weight|speed|market cap)\b",
        r"\bin (19\d\d|20\d\d)\b",
        r"\baccording to (the guinness|forbes|imf|world bank)\b",
    ]

    @classmethod
    def assess(
        cls,
        question: str,
        responses: List[str],
        features: Optional[Dict[str, float]] = None,
    ) -> AmbiguityAssessment:
        q_clean = question.strip().lower()
        evidence = []

        # Check for objective criteria constraints in question
        has_objective_constraint = any(
            re.search(pat, q_clean) for pat in cls.OBJECTIVE_CRITERIA_CONSTRAINTS
        )

        # Signal 1: Question asks for subjective superlative without objective constraint
        has_superlative = any(re.search(pat, q_clean) for pat in cls.CRITERION_SUPERLATIVES)
        if has_superlative and not has_objective_constraint:
            evidence.append("question_subjective_superlative_without_criteria")

        # Signal 2: Response text indicates subjectivity or criterion dependence
        resp_triggers = 0
        for r in responses:
            r_lower = r.lower()
            if any(re.search(pat, r_lower) for pat in cls.SUBJECTIVITY_RESPONSE_MARKERS):
                resp_triggers += 1

        if resp_triggers > 0:
            evidence.append(f"responses_mention_subjectivity_or_criteria({resp_triggers}/{len(responses)})")

        # Decision rule:
        # If question has subjective superlative without criteria AND (responses mention subjectivity OR high agreement on opinion):
        if "question_subjective_superlative_without_criteria" in evidence:
            return AmbiguityAssessment(
                is_ambiguous=True,
                reason=(
                    "Question asks for a subjective superlative or value judgment without "
                    "specifying evaluation criteria. Response consensus reflects shared prior "
                    "or cultural default, not a single determinate ground truth."
                ),
                confidence=0.90 if resp_triggers > 0 else 0.82,
                evidence=evidence,
            )

        if resp_triggers >= max(2, int(0.3 * len(responses))):
            return AmbiguityAssessment(
                is_ambiguous=True,
                reason=(
                    "Multiple responses explicitly identify the question as subjective, context-dependent, "
                    "or lacking a single consensus answer."
                ),
                confidence=0.85,
                evidence=evidence,
            )

        return AmbiguityAssessment(
            is_ambiguous=False,
            reason="No criterion ambiguity or subjective value judgments detected.",
            confidence=0.0,
            evidence=evidence,
        )
