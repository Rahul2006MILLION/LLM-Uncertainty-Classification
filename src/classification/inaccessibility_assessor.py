"""Inaccessibility assessment for private, unrecorded, or epistemically inaccessible queries."""

import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from src.classification.sensitive_request import (
    SensitiveRequestDetector,
    is_conceptual_question,
    normalize_text,
)


@dataclass
class InaccessibilityAssessment:
    is_inaccessible: bool
    predicted_label: Optional[str]  # "UNKNOWN" if inaccessible, else None
    confidence: float  # Rule-based heuristic confidence
    confidence_type: str = "rule_based_heuristic"
    status: str = "NOT_INACCESSIBLE"
    reason: str = ""
    evidence: List[str] = field(default_factory=list)


class InaccessibilityAssessor:
    """Evaluates question formulation and generated response evidence to assess inaccessibility.

    Principles:
    1. Epistemic Inaccessibility != Polysemic Ambiguity:
       An unanswerable private credential or unrecorded personal state has no accessible
       truth in any public training corpus. When an LLM samples multiple refusals with
       slight stylistic variance (e.g., 'As an AI, I cannot...' vs. 'The password is unknown...'),
       the resulting clustering and agreement margin mimic the statistical profile of
       ambiguity (AMBIGUOUS), leading to misclassification.
    2. Refusal Phrases Alone != Inaccessible:
       A refusal phrase alone (e.g., 'I cannot determine a single answer...') must NOT
       force UNKNOWN; legitimate ambiguous questions (e.g. 'What is the capital of Georgia?')
       frequently receive refusal or clarification responses.
       Therefore, inaccessibility classification strictly requires explicit question-level
       evidence of inaccessibility combined with response-level refusal evidence.
    """

    # Category 1: Private credentials, passphrases, and security tokens
    PRIVATE_CREDENTIAL_PATTERNS = [
        # Passwords / passphrases in physical safes, vaults, locks, personal accounts
        r"\b(secret\s+)?(passwords?|passphrases?|passcodes?|pins?|pin\s+codes?|combinations?)\b.*\b(safe|vault|lock|phone|router|wifi|network|account)\b",
        r"\b(safe|vault|lock|phone|router|wifi|network|account)\b.*\b(secret\s+)?(passwords?|passphrases?|passcodes?|pins?|pin\s+codes?|combinations?)\b",
        # Wi-Fi / Router credentials
        r"\b(wi-?fi|wpa2?|router|network)\s+(wpa2?\s+)?(passwords?|passphrases?|passcodes?|keys?|secrets?)\b",
        # Generic secret credentials
        r"\b(private\s+keys?|secret\s+keys?|access\s+tokens?|api\s+keys?|encryption\s+keys?)\b",
        r"\bwhat\s+is\s+(the\s+)?(secret\s+)?(passwords?|passphrases?|passcodes?|pin)\b",
    ]

    # Category 2: Private, unrecorded mental states and thoughts
    PRIVATE_MENTAL_STATE_PATTERNS = [
        r"\b(what|tell\s+me)\b.*\b(was\s+i|am\s+i|i\s+was|i\s+am)\s+(thinking|dreaming|feeling|wondering|imagining|contemplating)\b",
        r"\b(what\s+did\s+i\s+(think|dream|feel)\s+about)\b",
        r"\b(what\s+(is|was)\s+(in\s+)?my\s+(mind|head|thoughts?|dreams?))\b",
        r"\b(what\s+(number|word|letter|color|symbol|name)\s+am\s+i\s+(thinking|contemplating))\b",
        r"\b(read\s+my\s+mind|guess\s+what\s+i('m|\s+am)\s+thinking)\b",
        # "Guess the number I have in my head", "what colour is in my mind right now"
        r"\b(in|on)\s+my\s+(mind|head)\b",
        r"\bguess\s+(the|which|what)\s+[a-z]+\s+i\s+(have|am|'m|picked|chose|chosen)\b",
    ]

    # Subset of the mental-state / future-action patterns that unambiguously ask for a
    # value only the asker holds (no general-psychology reading is possible).
    STRONG_PRIVATE_STATE_PATTERNS = [
        r"\bwhat\s+(number|word|letter|colou?r|symbol|name|card|animal|shape)\s+(am\s+i|i'm|i\s+am|was\s+i)\s+(thinking|contemplating|imagining|picturing)\b",
        r"\b(read\s+my\s+mind|guess\s+what\s+i('m|\s+am)\s+thinking)\b",
        r"\bguess\s+(the|which|what)\s+[a-z]+\s+i\s+(have|am|'m|picked|chose|chosen)\b",
        r"\b(in|on)\s+my\s+(mind|head)\s+(right\s+)?now\b",
        r"\bwhat\s+(am\s+i|will\s+i|i'm)\s+(going\s+to\s+|gonna\s+)?(have|eat|cook|order|wear|buy)\b.*\b(tomorrow|tonight|later|next)\b",
    ]

    # Category 2b: The asker's own future private choices ("What am I going to eat tomorrow?")
    PRIVATE_FUTURE_ACTION_PATTERNS = [
        r"\bwhat\s+(am\s+i|will\s+i|i'm|i\s+am|are\s+we|will\s+we)\s+(going\s+to\s+|gonna\s+)?"
        r"(have|eat|cook|order|wear|buy|do|say|choose|pick|watch|name)\b"
        r".*\b(tomorrow|tonight|later|next|today|this\s+(evening|weekend|week))\b",
    ]

    # Category 3: Private physical secrets, unindexed notes, notebooks, and future private actions
    PRIVATE_PHYSICAL_SECRET_PATTERNS = [
        r"\b(code|word|note|writing|text|phrase|message)\b.*\b(written|hidden|placed|stored|locked|kept)\b.*\b(safe|vault|notebook|diary|journal|pocket|drawer|box)\b",
        r"\b(written|hidden|placed|stored|locked|kept)\b.*\b(safe|vault|notebook|diary|journal|pocket|drawer|box)\b.*\b(code|word|note|writing|text|phrase|message)\b",
        r"\bwhat\s+(secret\s+)?(word|note|message|number|phrase)\s+will\s+i\s+(write|put|type|say|think|jot)\b",
        r"\bwhat\s+will\s+i\s+(write\s+down|record|jot\s+down)\s+in\s+my\s+(notebook|diary|journal|notes?)\b",
        r"\b(in|inside)\s+my\s+(locked\s+)?safe\b",
    ]

    # Category 4: Genuinely unavailable personal facts and unrecorded historical minutiae
    UNRECORDED_HISTORICAL_MINUTIAE_PATTERNS = [
        r"\bwhat\s+did\s+[a-z\s]+\s+of\s+[a-z\s]+\s+(eat|have|drink)\s+for\s+(breakfast|lunch|dinner|supper)\b",
        r"\bwhat\s+did\s+[a-z\s]+\s+(eat|have|drink)\s+for\s+(breakfast|lunch|dinner|supper)\s+on\s+[a-z]+\s+\d{1,2}(,\s*\d{4})?\b",
        r"\bwhat\s+(exact\s+)?words\s+did\s+[a-z\s]+\s+whisper\s+(as|when|before)\b",
        r"\bwhat\s+(was|were)\s+the\s+unrecorded\s+(words|thoughts|actions)\b",
    ]

    # Response-level refusal and epistemic barrier indicators
    REFUSAL_INACCESSIBILITY_MARKERS = [
        r"\bi (cannot|can't|do not|don't|am unable to)\s+(access|know|see|determine|read|predict|tell|verify|provide|guess)\b",
        r"\b(do not|don't) have access to (your|the|any)\b",
        r"\b(cannot|can't)\s+(read|access|know)\s+(your\s+)?(mind|thoughts?|safe|router|network|notebook|diary)\b",
        r"\b(is|are|was|were)\s*(not known|unknown|unrecorded|inaccessible|confidential|private)\b",
        r"\bno (way|method|information|record|data|source) to know\b",
        r"\bthere is no (record|information|evidence)\b",
        r"\bnever recorded\b",
        r"\bas an ai,?\s*i (cannot|don't|do not)\b",
        r"\bpersonal (and confidential|security|information|matter|secret)\b",
        r"\bnot possible to (know|determine)\b",
        r"\bwithout (being provided|you telling|access)\b",
    ]

    # Explicit entity/criterion ambiguity signals that counter-indicate inaccessibility
    AMBIGUITY_COUNTER_SIGNALS = [
        r"\b(capital\s+of\s+georgia|who\s+is\s+jordan|springfield)\b",
        r"\b(best\s+programming\s+language|most\s+successful\s+person|greatest\s+country)\b",
    ]

    @classmethod
    def assess(
        cls,
        question: str,
        responses: Optional[Union[List[str], str]] = None,
        features: Optional[Dict[str, float]] = None,
    ) -> InaccessibilityAssessment:
        """Assess whether a query seeks inaccessible private or unrecorded information.

        Requires:
        1. Explicit question-level evidence matching a supported inaccessibility class.
        2. Where responses are available, confirmation that responses indicate unanswerability
           or access barriers (avoiding false overrides on subjective or polysemous queries).
        """
        q_clean = question.strip()
        # Normalised text: undoes light obfuscation and collapses line breaks so that
        # multi-line or spaced-out input cannot slip between pattern keywords.
        q_lower = normalize_text(q_clean)
        evidence: List[str] = []

        # 0. Strongest evidence: a request for a specific person's private data
        #    ("what is MY bank account number"). The model has no access to such data,
        #    so the question is UNKNOWN whatever the responses say; no refusal
        #    confirmation is required.
        private = SensitiveRequestDetector.assess(q_clean)
        if private.is_private_request:
            return InaccessibilityAssessment(
                is_inaccessible=True,
                predicted_label="UNKNOWN",
                confidence=0.95,
                confidence_type="rule_based_heuristic",
                status="ASSESSED_INACCESSIBLE_RULE_HEURISTIC (private_personal_data)",
                reason=(
                    f"{private.reason} Personal records and secrets are not available to the model, "
                    "so any generated value would be fabricated."
                ),
                evidence=private.evidence,
            )

        # Conceptual / procedural questions about sensitive topics ("what is an API key",
        # "how do I reset a router password") are answerable public knowledge.
        conceptual = is_conceptual_question(q_clean)

        # Disqualify if question matches known ambiguity counter-signals without private keywords
        for counter_pat in cls.AMBIGUITY_COUNTER_SIGNALS:
            if re.search(counter_pat, q_lower):
                # Ensure no explicit private credential trigger is present
                if not any(re.search(p, q_lower) for p in cls.PRIVATE_CREDENTIAL_PATTERNS):
                    return InaccessibilityAssessment(
                        is_inaccessible=False,
                        predicted_label=None,
                        confidence=0.0,
                        status="NOT_INACCESSIBLE",
                        reason=f"Question matches known ambiguity signature '{counter_pat}'.",
                        evidence=[f"counter_ambiguity:{counter_pat}"],
                    )

        # 1. Question-level evidence checking across supported inaccessibility classes
        inacc_category = None

        # Credential keywords without an owner (e.g. "the password for the router at X") are
        # weaker evidence: they still require refusal confirmation from the responses, and
        # never apply to conceptual or procedural questions.
        if not conceptual:
            for pat in cls.PRIVATE_CREDENTIAL_PATTERNS:
                if re.search(pat, q_lower):
                    inacc_category = "private_credential"
                    evidence.append(f"question_private_credential:{pat}")
                    break

        if not inacc_category:
            for pat in cls.PRIVATE_MENTAL_STATE_PATTERNS:
                if re.search(pat, q_lower):
                    inacc_category = "private_mental_state"
                    evidence.append(f"question_private_mental_state:{pat}")
                    break

        if not inacc_category:
            for pat in cls.PRIVATE_FUTURE_ACTION_PATTERNS:
                if re.search(pat, q_lower):
                    inacc_category = "private_future_action"
                    evidence.append(f"question_private_future_action:{pat}")
                    break

        if not inacc_category:
            for pat in cls.PRIVATE_PHYSICAL_SECRET_PATTERNS:
                if re.search(pat, q_lower):
                    inacc_category = "private_physical_secret"
                    evidence.append(f"question_private_physical_secret:{pat}")
                    break

        if not inacc_category:
            for pat in cls.UNRECORDED_HISTORICAL_MINUTIAE_PATTERNS:
                if re.search(pat, q_lower):
                    inacc_category = "unrecorded_historical_minutiae"
                    evidence.append(f"question_unrecorded_historical_minutiae:{pat}")
                    break

        # If no question-level inaccessibility pattern matched, do NOT force UNKNOWN
        if not inacc_category:
            return InaccessibilityAssessment(
                is_inaccessible=False,
                predicted_label=None,
                confidence=0.0,
                status="NOT_INACCESSIBLE",
                reason="Question does not match supported inaccessible private or unrecorded information patterns.",
                evidence=evidence,
            )

        # 2. Response-level evidence checking (where available)
        response_list: List[str] = []
        if isinstance(responses, str):
            response_list = [responses]
        elif isinstance(responses, (list, tuple)):
            response_list = [str(r) for r in responses if r]

        refusal_count = 0
        if response_list:
            for r in response_list:
                r_lower = r.lower()
                if any(re.search(pat, r_lower) for pat in cls.REFUSAL_INACCESSIBILITY_MARKERS):
                    refusal_count += 1
            evidence.append(f"responses_indicate_inaccessibility({refusal_count}/{len(response_list)})")

            # If responses are available, verify that at least one response (or >= 20%) displays refusal
            # This protects against cases where a question might have a factual answer available
            # Keyword-only credential evidence (no owner) is weak: require at least one
            # refusal even for a single fast-path response.
            weak_evidence = inacc_category == "private_credential"
            # Explicit requests to identify a value only the asker holds ("what number am I
            # thinking of", "read my mind") cannot be answered by anyone else: a confident
            # response is fabricated, so no refusal confirmation is required.
            strong_evidence = inacc_category in ("private_mental_state", "private_future_action") and any(
                re.search(p, q_lower) for p in cls.STRONG_PRIVATE_STATE_PATTERNS
            )
            if refusal_count == 0 and not strong_evidence and (len(response_list) >= 3 or weak_evidence):
                return InaccessibilityAssessment(
                    is_inaccessible=False,
                    predicted_label=None,
                    confidence=0.0,
                    status="RESPONSES_LACK_REFUSAL",
                    reason="Question matches inaccessibility pattern, but responses do not reflect refusal or access barrier.",
                    evidence=evidence,
                )

        confidence = 0.92 if (response_list and refusal_count >= max(1, int(0.3 * len(response_list)))) else 0.85
        return InaccessibilityAssessment(
            is_inaccessible=True,
            predicted_label="UNKNOWN",
            confidence=confidence,
            confidence_type="rule_based_heuristic",
            status=f"ASSESSED_INACCESSIBLE_RULE_HEURISTIC ({inacc_category})",
            reason=(
                f"Question queries {inacc_category.replace('_', ' ')} absent from any accessible corpus. "
                "Response variations reflect stylistic paraphrasing of refusals, not legitimate competing interpretations."
            ),
            evidence=evidence,
        )

