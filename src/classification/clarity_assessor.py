"""Question clarity assessment layer distinguishing question clarity from epistemic answerability.

Principles:
1. Question clarity is orthogonal to answerability:
   - "What is the capital of France?" -> UNAMBIGUOUS (Clarity) | KNOWN (Answerability)
   - "What will the exact population of Bengaluru be on Jan 1, 2040?" -> UNAMBIGUOUS (Clarity) | UNKNOWN (Answerability)
   - "Is it better?" -> AMBIGUOUS (Clarity) | AMBIGUOUS (Answerability)
   - "Who is the greatest scientist of all time?" -> AMBIGUOUS (Clarity) | AMBIGUOUS (Answerability)
2. Prevents missing-context/missing-referent queries from entering direct FAST_PATH execution.
3. Does not over-classify broad but answerable questions (e.g. questions with use-case contexts like 'for machine learning').
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ClarityAssessment:
    status: str  # "UNAMBIGUOUS" or "AMBIGUOUS"
    confidence: float
    ambiguity_type: Optional[str] = None  # "MISSING_REFERENT", "MISSING_COMPARISON", "SUBJECTIVE_CRITERIA", "MISSING_CONTEXT", "MULTIPLE_INTERPRETATIONS"
    clarification_required: bool = False
    reason: str = ""
    evidence: List[str] = field(default_factory=list)


class QuestionClarityAssessor:
    """Evaluates question formulation to diagnose structural and semantic clarity."""

    # 1. Unanchored deictic pronouns & missing referents
    MISSING_REFERENT_PATTERNS = [
        # "Which one is better?", "Which one should I buy?"
        r"^\s*(which|what)\s+one\b",
        # "Is it better?", "Is that good?", "Is this right?"
        r"^\s*(is|was|will|are|were|do|does|did|should|can|could|would)\s+(it|that|this)\s+(better|good|bad|worth\s+it|ok|okay|safe|true|correct|right|wrong|effective|fast|fine|working)\??\s*$",
        # "Should I choose it?", "Should I pick this?", "Should I buy that?"
        r"^\s*(should|can|could|would|shall)\s+(i|we|one)\s+(choose|pick|buy|take|get|use|select|prefer)\s+(it|this|that|one)\??\s*$",
        # "Is it better?", "Which one is better?"
        r"^\s*(which\s+one\s+is\s+(better|best|worse|worst|preferable|faster|cheaper|good|bad))\??\s*$",
        # "What about it?", "How about that?"
        r"^\s*(what|how)\s+about\s+(it|this|that|one)\??\s*$",
        # Short pronoun-only queries: "Do you like it?", "Is that all?"
        r"^\s*do\s+you\s+(like|recommend|prefer|want)\s+(it|this|that)\??\s*$",
    ]

    # 2. Missing comparison baseline (comparative adjectives lacking reference point or baseline)
    MISSING_COMPARISON_PATTERNS = [
        r"^\s*(which|what)\s+(is|are)\s+(better|worse|faster|slower|cheaper|stronger|easier|harder)\??\s*$",
        r"^\s*(is|are)\s+[a-zA-Z0-9_\-\.\s]+\s+(better|worse|faster|slower|cheaper)\??\s*$",
    ]

    # 3. Superlatives lacking objective criteria or use-case constraints
    SUBJECTIVE_CRITERIA_PATTERNS = [
        r"\b(who|what)\s+is\s+(the\s+)?(greatest|finest|best|worst|favorite|most\s+successful|most\s+important)\s+[a-zA-Z\s]+(\s+of\s+all\s+time)?\??$",
        r"\b(who|which)\s+was\s+(the\s+)?(greatest|finest|best|worst|favorite|most\s+successful|most\s+important)\b",
    ]

    # 4. Objective constraints or explicit contexts that disambiguate criteria
    OBJECTIVE_CRITERIA_CONSTRAINTS = [
        r"\bby\s+(population|area|land area|gdp|revenue|net worth|height|weight|speed|market cap|volume)\b",
        r"\bin\s+(19\d\d|20\d\d)\b",
        r"\baccording to\s+(guinness|forbes|imf|world bank|iso|who|un)\b",
        # Use-case or domain contexts that specify purpose
        r"\bfor\s+(machine\s+learning|data\s+science|beginners|gaming|programming|web\s+development|college|office|travel|work|business|general\s+use|speed|production)\b",
        r"\b(budget|under\s+\d+|inr|\$|usd|eur)\b",
    ]

    # 5. Missing geographic or situational context
    MISSING_CONTEXT_PATTERNS = [
        r"^\s*what\s+time\s+does\s+(the\s+)?(bank|post\s+office|store|supermarket|mall|pharmacy)\s+close\??\s*$",
        r"^\s*what\s+is\s+(the\s+)?(sales\s+tax\s+rate|speed\s+limit|minimum\s+wage|legal\s+voting\s+age|drinking\s+age)\??\s*$",
        r"^\s*who\s+won\s+(the\s+)?championship\s+game\s+last\s+night\??\s*$",
        r"^\s*what\s+is\s+the\s+temperature\s+right\s+now\??\s*$",
    ]

    # 6. Polysemic terms with multiple common referents
    POLYSEMIC_TERMS = [
        r"\b(georgia|jordan|mercury|columbia|springfield|alexandria|victoria)\b",
    ]

    @classmethod
    def assess(cls, question: str) -> ClarityAssessment:
        q_clean = question.strip()
        q_lower = q_clean.lower()

        # Check for explicit context block (e.g. from previous clarification round)
        if re.search(r"\[context:\s*[^\]]+\]", q_clean, flags=re.IGNORECASE):
            return ClarityAssessment(
                status="UNAMBIGUOUS",
                confidence=0.95,
                ambiguity_type=None,
                clarification_required=False,
                reason="Question includes explicit user-provided context block.",
                evidence=["explicit_context_block"],
            )

        # 1. Missing Referent: Deictic pronouns and unanchored referents
        for pat in cls.MISSING_REFERENT_PATTERNS:
            if re.search(pat, q_lower):
                return ClarityAssessment(
                    status="AMBIGUOUS",
                    confidence=0.95,
                    ambiguity_type="MISSING_REFERENT",
                    clarification_required=True,
                    reason="Question uses unanchored demonstrative or deictic pronoun ('it', 'that', 'this', 'which one') without an antecedent.",
                    evidence=[f"missing_referent:{pat}"],
                )

        # 2. Missing Comparison: Comparatives lacking baseline or options
        for pat in cls.MISSING_COMPARISON_PATTERNS:
            if re.search(pat, q_lower):
                # Ensure no explicit 'than' or 'between' or 'or' is provided
                if not re.search(r"\b(than|between|versus|vs|or)\b", q_lower):
                    return ClarityAssessment(
                        status="AMBIGUOUS",
                        confidence=0.90,
                        ambiguity_type="MISSING_COMPARISON",
                        clarification_required=True,
                        reason="Question contains a comparative adjective without specifying comparison options or baseline.",
                        evidence=[f"missing_comparison:{pat}"],
                    )

        # 3. Missing Context: Situational/jurisdiction queries lacking anchor
        for pat in cls.MISSING_CONTEXT_PATTERNS:
            if re.search(pat, q_lower):
                return ClarityAssessment(
                    status="AMBIGUOUS",
                    confidence=0.90,
                    ambiguity_type="MISSING_CONTEXT",
                    clarification_required=True,
                    reason="Question assumes unstated local, geographic, or temporal context.",
                    evidence=[f"missing_context:{pat}"],
                )

        # 4. Multiple Interpretations: Polysemic terms
        # Check if question mentions polysemic terms without disambiguating context
        for pat in cls.POLYSEMIC_TERMS:
            if re.search(pat, q_lower):
                # Ensure question doesn't already disambiguate (e.g. "country of Georgia" or "capital of Georgia")
                disambiguated = re.search(
                    r"\b(state\s+of|country\s+of|republic\s+of|element|planet|god|river|island|capital\s+of|city\s+of|river\s+in)\b",
                    q_lower
                )
                if not disambiguated:
                    return ClarityAssessment(
                        status="AMBIGUOUS",
                        confidence=0.88,
                        ambiguity_type="MULTIPLE_INTERPRETATIONS",
                        clarification_required=True,
                        reason=f"Question contains polysemic entity or name with multiple common interpretations: '{pat}'.",
                        evidence=[f"polysemic_term:{pat}"],
                    )

        # 5. Subjective Criteria: Superlatives without criteria or use-case constraints
        has_constraint = any(re.search(pat, q_lower) for pat in cls.OBJECTIVE_CRITERIA_CONSTRAINTS)
        if not has_constraint:
            for pat in cls.SUBJECTIVE_CRITERIA_PATTERNS:
                if re.search(pat, q_lower):
                    return ClarityAssessment(
                        status="AMBIGUOUS",
                        confidence=0.88,
                        ambiguity_type="SUBJECTIVE_CRITERIA",
                        clarification_required=True,
                        reason="Question asks for a subjective superlative without explicit evaluation criteria or domain constraints.",
                        evidence=[f"subjective_criteria:{pat}"],
                    )

        # Default: Question has sufficient structural and semantic clarity
        return ClarityAssessment(
            status="UNAMBIGUOUS",
            confidence=0.95,
            ambiguity_type=None,
            clarification_required=False,
            reason="Question specifies unambiguous referents, domain, and entities.",
            evidence=["clear_grammatical_formulation"],
        )
