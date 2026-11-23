"""Fast-path answerability and uncertainty assessment for single-response routes."""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.classification.clarity_assessor import QuestionClarityAssessor
from src.classification.inaccessibility_assessor import InaccessibilityAssessor


@dataclass
class FastPathAssessment:
    is_assessed: bool
    predicted_label: Optional[str]  # "KNOWN", "AMBIGUOUS", "UNKNOWN", or None if uncertain
    confidence: float
    confidence_type: str
    status: str
    reason: str
    evidence: List[str] = field(default_factory=list)


class FastPathAnswerabilityAssessor:
    """Explicit answerability assessment for fast-path questions.

    Principles:
    1. A fast route describes an execution strategy (generating 1 response instead of 10),
       NOT an automatic label of 'KNOWN'.
    2. Single-response features must NOT be passed to a multi-response classifier because
       distributional assumptions (cluster_ratio, response_len_std, distinct_2gram_ratio)
       diverge dramatically between N=1 and N=10 samples.
    3. Answerability Evaluation:
       - Refusal / Inability to answer -> UNKNOWN
       - Detected polysemy / criterion ambiguity -> AMBIGUOUS
       - Deterministic arithmetic -> KNOWN
       - Established concept definition -> KNOWN
       - Direct, unhedged factual answer to a concise query -> KNOWN
       - Hedged, conflicting, or uncertain single response -> UNSUPPORTED / REROUTE
    """

    # Explicit refusal and inability to answer triggers
    REFUSAL_TRIGGERS = [
        r"\bi cannot (determine|answer|know|predict|verify)\b",
        r"\bunable to (determine|answer|know|predict|verify)\b",
        r"\b(can|could|is|was|are|were)?\s*cannot be (known|determined|answered|verified|found)\b",
        r"\bno (way|method|information|record|data|source) to know\b",
        r"\bthere is no (record|information|evidence)\b",
        r"\bnever recorded\b",
        r"\bunknown\b",
        r"\bunanswerable\b",
        r"\bas an ai,?\s*i cannot\b",
        r"\bnot possible to (know|determine)\b",
        # Access / disclosure refusals ("I don't have access to your passwords")
        r"\bi (do not|don't|cannot|can't|am unable to|am not able to) (have )?(access|provide|share|retrieve|see|disclose|reveal)\b",
        r"\b(do not|don't) have (access to|the ability to|any way to|information about) (your|personal|private|that)\b",
        r"\b(i'm|i am) (sorry|afraid),? (but )?(i|as an ai)\b",
    ]

    # Explicit ambiguity and polysemy triggers
    AMBIGUITY_TRIGGERS = [
        r"\bit depends\b",
        r"\bcan refer to\b",
        r"\bcould refer to\b",
        r"\bthere are multiple\b",
        r"\bmultiple (meanings|definitions|interpretations)\b",
        r"\bno single (answer|definition|consensus|best|greatest)\b",
        r"\bsubjective\b",
        r"\bmatter of (opinion|perspective)\b",
        r"\bvaries (greatly\s+)?based on\b",
        r"\bdepends on (the\s+)?(criteria|task|context|definition)\b",
    ]

    # Patterns for strictly deterministic closed-domain queries
    ARITHMETIC_QUERY_PATTERN = r"^\s*(what\s+is\s+)?\d+\s*[\+\-\*\/]\s*\d+\s*\??\s*$"
    ARITHMETIC_RESPONSE_PATTERN = r"\b\d+(\.\d+)?\b"

    # Patterns for established concept definitions
    DEFINITION_QUERY_PATTERNS = [
        # "what is my/your/their X" is a request for a specific value, not a concept definition
        r"^\s*what\s+is\s+(?!(the|my|your|our|his|her|their|its)\s+)(a\s+|an\s+)?[a-z\s]+\??\s*$",
        r"^\s*define\s+[a-z\s]+\??\s*$",
        r"^\s*what\s+does\s+[a-z\s]+\s+mean\??\s*$",
    ]

    # Hedging and epistemic doubt markers that prevent certification as KNOWN
    HEDGING_PATTERNS = [
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bmight\b",
        r"\bcould be\b",
        r"\bunclear\b",
        r"\bnot certain\b",
        r"\bnot entirely sure\b",
        r"\bdisputed\b",
        r"\bcontroversial\b",
    ]

    @classmethod
    def assess(
        cls,
        question: str,
        response: str,
    ) -> FastPathAssessment:
        q_clean = question.strip()
        resp_clean = response.strip()
        q_lower = q_clean.lower()
        resp_lower = resp_clean.lower()

        # -1. Safeguards take precedence over any fast-path certification. A request for
        #     private or otherwise inaccessible information is UNKNOWN even when the single
        #     greedy response sounds confident.
        inacc = InaccessibilityAssessor.assess(question=q_clean, responses=[resp_clean] if resp_clean else None)
        if inacc.is_inaccessible:
            refusal = next((p for p in cls.REFUSAL_TRIGGERS if re.search(p, resp_lower)), None)
            return FastPathAssessment(
                is_assessed=True,
                predicted_label="UNKNOWN",
                confidence=inacc.confidence,
                confidence_type="rule_based_heuristic",
                # Keep the established REFUSAL status when the response itself refuses;
                # INACCESSIBLE marks the case where the safeguard caught a confident answer.
                status="ASSESSED_FAST_PATH_REFUSAL" if refusal else "ASSESSED_FAST_PATH_INACCESSIBLE",
                reason=inacc.reason,
                evidence=inacc.evidence + ([f"refusal:{refusal}"] if refusal else []),
            )

        # 0. Check Question Clarity (missing referent, missing context, polysemy)
        clarity = QuestionClarityAssessor.assess(q_clean)
        if clarity.status == "AMBIGUOUS" and clarity.clarification_required:
            return FastPathAssessment(
                is_assessed=True,
                predicted_label="AMBIGUOUS",
                confidence=clarity.confidence,
                confidence_type="rule_based_heuristic",
                status="ASSESSED_FAST_PATH_AMBIGUITY",
                reason=f"Question clarity assessment flagged {clarity.ambiguity_type}: {clarity.reason}",
                evidence=clarity.evidence,
            )

        # 1. Check for explicit inability/refusal in response -> UNKNOWN
        for pat in cls.REFUSAL_TRIGGERS:
            if re.search(pat, resp_lower):
                return FastPathAssessment(
                    is_assessed=True,
                    predicted_label="UNKNOWN",
                    confidence=0.90,
                    confidence_type="rule_based_heuristic",
                    status="ASSESSED_FAST_PATH_REFUSAL",
                    reason=f"Single response explicitly indicates inability to answer: '{pat}'.",
                    evidence=[f"refusal:{pat}"],
                )

        # 2. Check for explicit ambiguity/polysemy in response -> AMBIGUOUS
        for pat in cls.AMBIGUITY_TRIGGERS:
            if re.search(pat, resp_lower):
                return FastPathAssessment(
                    is_assessed=True,
                    predicted_label="AMBIGUOUS",
                    confidence=0.90,
                    confidence_type="rule_based_heuristic",
                    status="ASSESSED_FAST_PATH_AMBIGUITY",
                    reason=f"Single response indicates multiple interpretations or subjectivity: '{pat}'.",
                    evidence=[f"ambiguity:{pat}"],
                )

        # 3. Check for hedging or epistemic doubt -> UNCERTAIN (cannot certify from single response)
        hedging_found = [pat for pat in cls.HEDGING_PATTERNS if re.search(pat, resp_lower)]
        if hedging_found:
            return FastPathAssessment(
                is_assessed=False,
                predicted_label=None,
                confidence=0.50,
                confidence_type="rule_based_heuristic",
                status="FAST_PATH_UNCERTAIN_HEDGED",
                reason=f"Response contains epistemic hedging ({', '.join(hedging_found)}). Cannot certify certainty from a single sample.",
                evidence=hedging_found,
            )

        # 4. Check for deterministic arithmetic query
        if re.search(cls.ARITHMETIC_QUERY_PATTERN, q_lower):
            if re.search(cls.ARITHMETIC_RESPONSE_PATTERN, resp_lower):
                return FastPathAssessment(
                    is_assessed=True,
                    predicted_label="KNOWN",
                    confidence=0.95,
                    confidence_type="rule_based_heuristic",
                    status="ASSESSED_FAST_PATH_DETERMINISTIC_ARITHMETIC",
                    reason="Deterministic arithmetic calculation verified with direct numeric answer.",
                    evidence=["arithmetic_query_and_response_match"],
                )

        # 5. Check for concept definition query
        is_def_query = any(re.search(pat, q_lower) for pat in cls.DEFINITION_QUERY_PATTERNS)
        if is_def_query:
            # Must be substantive (> 20 chars) and free of hedging/negation
            if len(resp_clean) >= 20:
                return FastPathAssessment(
                    is_assessed=True,
                    predicted_label="KNOWN",
                    confidence=0.90,
                    confidence_type="rule_based_heuristic",
                    status="ASSESSED_FAST_PATH_CONCEPT_DEFINITION",
                    reason="Definitional query answered directly with substantive definition.",
                    evidence=["definition_query_and_substantive_response"],
                )

        # 6. Check for direct concise factual query
        # Short question (<= 8 words) with concise answer (< 200 chars) and zero hedging
        if len(q_clean.split()) <= 8 and len(resp_clean) < 200 and len(resp_clean) >= 2:
            return FastPathAssessment(
                is_assessed=True,
                predicted_label="KNOWN",
                confidence=0.85,
                confidence_type="rule_based_heuristic",
                status="ASSESSED_FAST_PATH_DIRECT_FACT",
                reason="Concise direct query answered with unhedged factual response.",
                evidence=["concise_unhedged_factual_match"],
            )

        # 7. Fallback: Query complexity or response breadth cannot be certified from N=1
        return FastPathAssessment(
            is_assessed=False,
            predicted_label=None,
            confidence=0.50,
            confidence_type="rule_based_heuristic",
            status="FAST_PATH_UNCERTAIN_REROUTE",
            reason="Single response cannot be conclusively certified as KNOWN without consistency verification.",
            evidence=["unclassified_fast_path"],
        )
