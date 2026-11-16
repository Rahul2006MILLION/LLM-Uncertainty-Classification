"""Adaptive routing logic to send questions to fast path or multi-response analysis."""

import re
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Question patterns that inherently contain polysemy, multiple criteria, or unanswerability
AMBIGUITY_QUESTION_PATTERNS = [
    # Polysemic entities with well-known multiple referents (documented regression cases & geographic polysemy)
    r"\b(georgia|jordan|washington|columbia|alexandria|victoria|mercury)\b",
    # Underspecified single-name queries, e.g. "Who is Jordan?", "Who is Smith?"
    r"^(who|what|where)\s+is\s+[A-Za-z]+(\s+[A-Za-z]+)?\??$",
    # Criterion ambiguity & subjective superlatives (requiring unstated criteria, e.g. success, greatness, impact, language utility)
    r"\b(largest|biggest|tallest|fastest|best|oldest|richest|greatest|finest|worst|favorite|most\s+[a-z]+)\b",
    # Private mental state or indeterminate future contingencies
    r"\b(thinking|mind|guess|secret|future|tomorrow|lottery|predict)\b",
]

# Patterns in initial response that indicate the model itself sees multiple interpretations or uncertainty
INITIAL_RESPONSE_AMBIGUITY_TRIGGERS = [
    r"\bit depends\b",
    r"\bcan refer to\b",
    r"\bcould refer to\b",
    r"\bthere are multiple\b",
    r"\bmultiple (meanings|definitions|cities|people|countries)\b",
    r"\beither\b.*\bor\b",
    r"\bdepending on (the|whether)\b",
    r"\bby population\b.*\bby area\b",
    r"\bstate of georgia\b.*\bcountry of georgia\b",
    r"\bi cannot determine\b",
    r"\bi cannot know\b",
    r"\bunable to know\b",
    r"\bno way to know\b",
    r"\bas an ai,?\s*i cannot\b",
    r"\bno single\b",
    r"\bsubjective\b",
    r"\bdepends on (the\s+)?(criteria|task|context|definition|perspective)\b",
    r"\bno universally accepted\b",
    r"\bvaries (greatly\s+)?based on\b",
    r"\bmatter of (opinion|perspective)\b",
    r"\bdifferent criteria\b",
]

# High-confidence fast-path triggers: strictly deterministic arithmetic and established concept definitions
FAST_PATH_QUESTION_PATTERNS = [
    r"^\s*what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$",              # e.g., "What is 2 + 2?"
    r"^\s*\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$",                          # e.g., "2 + 2"
    r"^\s*what\s+is\s+data\s+science\??\s*$",                          # Concept definition
    r"^\s*define\s+[a-z\s]+\??\s*$",                                  # Concept definition
]


@dataclass
class RoutingDecision:
    path: str  # "FAST_PATH" or "MULTI_RESPONSE"
    reason: str
    confidence: float
    triggers: list


class AdaptiveRouter:
    """Decides whether a question can be answered via fast path or requires multi-response."""

    def __init__(self, fast_path_enabled: bool = True):
        self.fast_path_enabled = fast_path_enabled

    def route(self, question: str, initial_response: str) -> RoutingDecision:
        """Evaluate question and initial single response to decide routing path."""
        q_clean = question.strip()
        resp_clean = initial_response.strip().lower()
        triggers = []

        if not self.fast_path_enabled:
            return RoutingDecision(
                path="MULTI_RESPONSE",
                reason="Fast path is globally disabled in configuration.",
                confidence=1.0,
                triggers=["fast_path_disabled"],
            )

        # 1. Check for explicit fast path triggers (arithmetic, simple definitional questions)
        for pattern in FAST_PATH_QUESTION_PATTERNS:
            if re.search(pattern, q_clean, flags=re.IGNORECASE):
                # Ensure initial response does not show unexpected hesitation
                if not any(re.search(trigger, resp_clean) for trigger in INITIAL_RESPONSE_AMBIGUITY_TRIGGERS):
                    return RoutingDecision(
                        path="FAST_PATH",
                        reason=f"Matched high-certainty query pattern '{pattern}'.",
                        confidence=0.95,
                        triggers=["fast_pattern_match"],
                    )

        # 2. Check if the initial response signaled ambiguity or inability to answer
        for trigger in INITIAL_RESPONSE_AMBIGUITY_TRIGGERS:
            if re.search(trigger, resp_clean):
                triggers.append(f"response_trigger:{trigger}")

        # 3. Check if the question text signals inherent ambiguity or unanswerable private state
        for pattern in AMBIGUITY_QUESTION_PATTERNS:
            if re.search(pattern, q_clean, flags=re.IGNORECASE):
                triggers.append(f"question_trigger:{pattern}")

        # 4. If any ambiguity/uncertainty triggers fired, route to MULTI_RESPONSE
        if triggers:
            return RoutingDecision(
                path="MULTI_RESPONSE",
                reason=f"Detected ambiguity or uncertainty indicators: {', '.join(triggers[:3])}",
                confidence=0.85,
                triggers=triggers,
            )

        # 5. Length and complexity heuristic:
        # If response is concise (< 200 chars) and free of hedging/subjectivity, can qualify as fast path
        has_hedging = bool(re.search(r"\b(maybe|perhaps|possibly|might|unclear|subjective|depends)\b", resp_clean))
        is_subjective_query = any(re.search(p, q_clean, flags=re.IGNORECASE) for p in AMBIGUITY_QUESTION_PATTERNS)
        if len(resp_clean) < 200 and not has_hedging and not is_subjective_query and len(q_clean.split()) <= 8:
            return RoutingDecision(
                path="FAST_PATH",
                reason="Concise, direct single response with zero hedging on short query.",
                confidence=0.80,
                triggers=["concise_unhedged"],
            )

        # Default fallback: route to multi-response for thorough uncertainty evaluation
        return RoutingDecision(
            path="MULTI_RESPONSE",
            reason="Query complexity or response breadth warrants multi-response consistency analysis.",
            confidence=0.75,
            triggers=["default_multi_response"],
        )
