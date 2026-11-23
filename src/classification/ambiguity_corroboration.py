"""Routing-layer check: is an AMBIGUOUS classification corroborated enough to ask the user?

This module never changes the classifier's label. It only decides whether an
AMBIGUOUS label should turn into a human clarification request.

Background (docs/REMEDIATION_AUDIT.md): the trained classifier labels many unanimous,
single-cluster answers to plain explanatory questions ("What is two-factor
authentication?") as AMBIGUOUS, and `experiments/ask.py` used to turn every such label
into `CLARIFICATION_REQUIRED`. The method's own premise is that ambiguity shows up as
competing interpretations, so a clarification request is made only when at least one
independent signal corroborates the label:

1. the question itself is ambiguous (QuestionClarityAssessor: missing referent,
   missing comparison or context, unresolved polysemous entity, subjective criteria);
2. criterion ambiguity (CriterionAmbiguityAssessor);
3. a competing semantic cluster: second-cluster agreement >= the configured
   `output_selection.explanation_threshold` (0.25 by default);
4. the responses themselves say the answer depends on the reading of the question
   ("can refer to", "it depends on which…") in at least 30% of samples.

Without corroboration the label is still reported as AMBIGUOUS. The user simply isn't
interrupted with a clarification question that has nothing to resolve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
from src.classification.clarity_assessor import QuestionClarityAssessor

RESPONSE_AMBIGUITY_MARKERS = [
    r"\b(can|could|may|might) refer to\b",
    r"\bit depends\b",
    r"\bdepend(s|ing) on (the|which|what|whether|your|how|context|criteria|perspective)\b",
    r"\b(several|multiple|different) (meanings|interpretations|senses|possible answers)\b",
    r"\bif you (mean|are referring to)\b",
    r"\bno single (answer|best|definitive)\b",
    r"\bsubjective\b",
]
RESPONSE_MARKER_SHARE = 0.3


@dataclass
class AmbiguityCorroboration:
    corroborated: bool
    reason: str
    evidence: List[str] = field(default_factory=list)


def response_ambiguity_share(responses: List[str]) -> float:
    if not responses:
        return 0.0
    hits = sum(1 for r in responses if any(re.search(p, (r or "").lower()) for p in RESPONSE_AMBIGUITY_MARKERS))
    return hits / len(responses)


def assess_ambiguity_corroboration(
    question: str,
    responses: List[str],
    features: Dict[str, float],
    explanation_threshold: float = 0.25,
    is_clarified: bool = False,
) -> AmbiguityCorroboration:
    evidence: List[str] = []

    clarity = QuestionClarityAssessor.assess(question)
    if clarity.status == "AMBIGUOUS" and clarity.clarification_required and not is_clarified:
        evidence.append(f"question_clarity:{clarity.ambiguity_type}")

    criterion = CriterionAmbiguityAssessor.assess(question=question, responses=responses, features=features)
    if criterion.is_ambiguous and not is_clarified:
        evidence.append("criterion_ambiguity")

    p2 = float(features.get("second_agreement", 0.0) or 0.0)
    if p2 >= explanation_threshold:
        evidence.append(f"competing_cluster:{p2:.2f}")

    share = response_ambiguity_share(responses)
    if share >= RESPONSE_MARKER_SHARE:
        evidence.append(f"responses_mention_alternatives:{share:.2f}")

    if evidence:
        return AmbiguityCorroboration(True, "AMBIGUOUS classification corroborated: " + ", ".join(evidence), evidence)
    return AmbiguityCorroboration(
        False,
        "Classifier predicted AMBIGUOUS, but no independent signal supports asking the user: the question is "
        f"clear, samples show no competing interpretation (second cluster {p2:.0%} < {explanation_threshold:.0%}) "
        f"and {share:.0%} of responses mention alternative readings.",
        [],
    )
