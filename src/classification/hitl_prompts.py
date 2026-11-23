"""Targeted clarification prompt generation and interaction templates for HITL.

Principles:
1. One specific clarification question at a time (<= 25 words).
2. Brief explanation of why the context matters.
3. Suggested answer choices where applicable with an explicit skip option.
4. Open-ended response support.
5. Strict protection against credential solicitation and prompt injection.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


# Regex patterns to detect sensitive credentials accidentally provided by the user
SENSITIVE_CREDENTIAL_PATTERNS = [
    r"\b(password|passwd|pin|secret|api[_-]?key|token|auth[_-]?token|private[_-]?key)\s*[:=]\s*\S+",
    r"\b(pin|passcode|secret\s*code|pass\s*code)\s*[:=]?\s*\d{4,8}\b",  # Labeled PIN or passcode
    r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9_\-\.]+)\b",
    # Natural-language disclosure: "the password is hunter22", "my pin was 4821".
    # The value must look like a secret (contains a digit or symbol) so that
    # "the password is strong" is not flagged.
    r"\b(password|passwd|passcode|passphrase|pass|pwd|pw|pin|api\s*key|secret(\s*key)?|token|cvv|cvc)\s*(is|was|=|:)\s*['\"]?(?=[^\s'\"]*[\d@#$%^&*!?])[^\s'\"]{4,}",
    # Provider key families (prefix + long random body), not single prefixes:
    #   OpenAI / Anthropic / OpenRouter  sk-…, sk-proj-…, sk-ant-…, sk-or-…
    #   Stripe                            sk_live_…, sk_test_…, rk_live_…, pk_live_…
    #   Slack                             xoxb-…, xoxp-…, xoxa-…, xoxr-…, xoxs-…
    #   Google API                        AIza + 35 chars
    #   GitHub                            ghp_/gho_/ghu_/ghs_/ghr_…, github_pat_…
    #   AWS access key id                 AKIA/ASIA + 16 chars
    #   JSON Web Token                    eyJ….eyJ….…
    r"\bsk-(proj-|live-|test-|ant-|or-)?[a-z0-9_\-]{16,}",
    r"\b(sk|rk|pk)_(live|test)_[a-z0-9]{16,}",
    r"\bxox[abprs]-[a-z0-9\-]{10,}",
    r"\baiza[0-9a-z_\-]{35}\b",
    r"\b(gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,})\b",
    r"\b(akia|asia)[0-9a-z]{16}\b",
    r"\beyj[a-z0-9_\-]{8,}\.eyj[a-z0-9_\-]{8,}\.[a-z0-9_\-]{8,}",
    # PEM private key blocks
    r"-----begin [a-z ]*private key-----",
]

# Payment card numbers: 13–19 digits, optionally grouped, validated with the Luhn checksum
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")

# Decline/skip intent patterns in user clarification response
DECLINE_PATTERNS = [
    r"^(skip|cancel|decline|none|neither|no|pass|n/a|stop|exit|quit|never\s*mind|just\s*answer|show\s*both)$",
    r"^(i\s*don'?t\s*(care|know)|doesn'?t\s*matter|either\s*is\s*fine|any|whatever)$",
    # Polite / conversational refusals, matched after punctuation is stripped
    r"^(no|nah|nope)(\s+(thanks|thank\s+you|thx|need|way))?$",
    r"^(skip|ignore|forget)\s+(it|this|that|the\s+question|clarification)$",
    r"^(not\s+now|forget\s+it|it\s+doesn'?t\s+matter|it\s+does\s+not\s+matter|no\s+preference|any\s+of\s+them|either\s+one)$",
    r"^(i'?d\s+rather\s+not|rather\s+not|i\s+don'?t\s+want\s+to\s+(say|clarify))$",
    r"^(please\s+)?(just\s+)?(answer|show|give\s+me)(\s+(it|the\s+answer|the\s+result|the\s+analysis|anyway))?(\s+please)?$",
    r"^(don'?t|do\s+not)\s+(ask|clarify)$",
    # "Just answer the original question", "go with my original question"
    r"^(please\s+)?(just\s+)?(answer|use|go\s+with|keep)\s+(the|my)\s+(original\s+|first\s+)?question(\s+(as\s+is|please))?$",
    r"^(i\s+)?(don'?t|do\s+not)\s+want\s+to\s+(clarify|explain|answer\s+that|specify)$",
]

# Curated clarification templates for recognized polysemic and subjective categories
CURATED_CLARIFICATIONS: Dict[str, Dict[str, Any]] = {
    "georgia": {
        "question": "Are you asking about Georgia the US State or Georgia the country in the Caucasus?",
        "options": ["US State of Georgia", "Country of Georgia", "Other / Skip"],
        "why_it_matters": "The capital, history, and government differ entirely between the state and the country.",
    },
    "jordan": {
        "question": "Are you asking about Jordan the country, Michael Jordan, or another person named Jordan?",
        "options": ["Country of Jordan", "Michael Jordan (Athlete)", "Michael B. Jordan (Actor)", "Other / Skip"],
        "why_it_matters": "Jordan can refer to a sovereign Middle Eastern nation or prominent public figures.",
    },
    "jaguar": {
        "question": "Are you asking about Jaguar the feline animal, the automobile brand, or the operating system?",
        "options": ["Animal (Big Cat)", "Automobile Manufacturer", "Software / OS", "Other / Skip"],
        "why_it_matters": "Physical properties, top speed, and history vary depending on the referent.",
    },
    "mercury": {
        "question": "Are you asking about Mercury the planet, the chemical element, or the Roman god?",
        "options": ["Planet Mercury", "Chemical Element (Hg)", "Roman God", "Other / Skip"],
        "why_it_matters": "Scientific and mythological descriptions are completely distinct.",
    },
    "bridge": {
        "question": "Are you asking about Bridge the card game, the civil engineering structure, or a dental bridge?",
        "options": ["Card Game", "Civil Infrastructure", "Dental Appliance", "Other / Skip"],
        "why_it_matters": "The definition and rules depend entirely on the physical or ludic context.",
    },
    "programming_language": {
        "question": "Which application domain or use case do you have in mind for the programming language?",
        "options": ["Machine Learning / Data Science", "Web Development", "Systems / High Performance", "Mobile Apps", "General Purpose"],
        "why_it_matters": "Language suitability depends directly on performance, ecosystem, and task requirements.",
    },
    "largest_city": {
        "question": "Do you mean largest by population, land area, or metropolitan area?",
        "options": ["Population (City Proper)", "Metropolitan Population", "Land Area", "Other / Skip"],
        "why_it_matters": "Tokyo leads by metropolitan population, Chongqing by administrative area, and others by land area.",
    },
    "model": {
        "question": "What is your primary optimization target or task for selecting a model?",
        "options": ["Low Latency / Edge", "Reasoning & Accuracy", "Code Generation", "Cost Efficiency"],
        "why_it_matters": "Model selection balances accuracy, parameter size, latency, and operational cost.",
    },
    "laptop": {
        "question": "What will you primarily use the laptop for—gaming, programming, productivity, or general use? Also, do you have a budget?",
        "options": ["Programming / Data Science", "Gaming / High Performance", "College / Productivity", "Budget / Value-focused"],
        "why_it_matters": "Hardware specifications (CPU, GPU, RAM, battery) and pricing depend directly on your intended use case.",
    },
    "phone": {
        "question": "What OS ecosystem (iOS vs. Android) and primary feature (camera, battery, budget) do you prefer?",
        "options": ["iOS / iPhone", "Android Flagship", "Camera Priority", "Budget / Value Priority"],
        "why_it_matters": "Operating system ecosystem and hardware priorities determine the optimal smartphone recommendation.",
    },
    "car": {
        "question": "What vehicle category, budget, and primary requirement (fuel efficiency, safety, cargo, performance) are you looking for?",
        "options": ["Sedan / Commuter", "SUV / Family", "Electric / Hybrid", "Performance / Luxury"],
        "why_it_matters": "Vehicle requirements and price brackets differ substantially across vehicle classes.",
    },
}


# Instruction-override phrasing commonly used for prompt injection
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|system|preceding)?\s*(instructions?|rules|prompts?|directives?|guidelines)\b[^.?!]*[.?!]?",
    r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you)\b[^.?!]*[.?!]?",
    r"\b(reveal|print|show|repeat|output)\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+instructions|initial\s+prompt)\b[^.?!]*[.?!]?",
    r"\b(classify|label|mark)\s+(this|it|the\s+question)\s+as\s+(known|ambiguous|unknown)\b[^.?!]*[.?!]?",
]


def sanitize_user_input(text: str) -> str:
    """Sanitize user clarification input so it is treated as data, not instructions.

    * Removes whole instruction blocks (``[INST]…[/INST]``, ``<<SYS>>…<</SYS>>``,
      ``<system>…</system>``), not just their tags.
    * Drops leading role labels ("SYSTEM:", "assistant:") and instruction-override
      phrases ("ignore all previous instructions…").
    * Collapses whitespace and caps the length at 200 characters.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Remove complete instruction blocks, including their contents
    cleaned = re.sub(r"\[(INST|SYS|SYSTEM)\].*?\[/\1\]", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<<\s*SYS\s*>>.*?<<\s*/\s*SYS\s*>>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<(system|user|assistant|tool_call)>.*?</\1>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Remove any remaining stray tags
    cleaned = re.sub(r"\[/?(?:INST|SYS|SYSTEM|USER|ASSISTANT)\]|<</?\s*SYS\s*>>|</?(system|user|assistant|tool_call)>", " ", cleaned, flags=re.IGNORECASE)
    # Role labels at the start of the text or of a line
    cleaned = re.sub(r"(^|\n)\s*(system|assistant|developer|user)\s*:\s*", " ", cleaned, flags=re.IGNORECASE)
    for pat in INJECTION_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:")
    # Limit length to 200 characters to prevent buffer bloating
    if len(cleaned) > 200:
        cleaned = cleaned[:200].strip()
    return cleaned


def contains_prompt_injection(text: str) -> bool:
    """True when the text contains instruction-override or role-manipulation phrasing."""
    t = text or ""
    if re.search(r"\[(INST|SYS)\]|<<\s*SYS\s*>>|<(system|assistant)>", t, flags=re.IGNORECASE):
        return True
    return any(re.search(p, t, flags=re.IGNORECASE) for p in INJECTION_PATTERNS)


def is_user_declining(user_input: str) -> bool:
    """Determine whether the user is explicitly declining or skipping clarification.

    Matching ignores case, surrounding punctuation and repeated whitespace, so
    "Skip.", "No thanks!" and "just answer please" are recognised.
    """
    cleaned = re.sub(r"[^\w\s'/]", " ", (user_input or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return any(re.search(pat, cleaned) for pat in DECLINE_PATTERNS)


def _luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _looks_like_random_secret(token: str) -> bool:
    """Generic detector for unknown key formats: a long, high-entropy token.

    Requires >= 24 characters from the base64/hex/url-safe alphabet, both letters and
    digits, and Shannon entropy >= 3.5 bits/char. URLs, e-mail addresses, file paths and
    dotted names are excluded because their tokens contain ':', '@', '/' or '.'.
    """
    import math

    tok = token.strip("'\"()[]{}<>,;")
    if len(tok) < 24 or not re.fullmatch(r"[A-Za-z0-9_\-+=]+", tok):
        return False
    if not (re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok)):
        return False
    counts = {c: tok.count(c) for c in set(tok)}
    entropy = -sum(n / len(tok) * math.log2(n / len(tok)) for n in counts.values())
    return entropy >= 3.5


def contains_sensitive_credentials(text: str) -> bool:
    """Check if text appears to contain sensitive credentials, passwords, tokens or card numbers.

    Text is normalised first, so spaced-out or hyphenated obfuscation
    ("p a s s w o r d: abc123") is still recognised.
    """
    from src.classification.sensitive_request import normalize_text  # local import avoids a cycle

    raw_lower = (text or "").lower()
    norm = normalize_text(text or "")
    for t_lower in {raw_lower, norm}:
        if any(re.search(pat, t_lower) for pat in SENSITIVE_CREDENTIAL_PATTERNS):
            return True
    if any(_looks_like_random_secret(tok) for tok in (text or "").split()):
        return True
    for m in _CARD_CANDIDATE_RE.finditer(raw_lower):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return True
    return False


def format_clarification_prompt(
    question: str,
    why_it_matters: str,
    options: Optional[List[str]] = None,
    round_num: int = 1,
    max_rounds: int = 2,
) -> str:
    """Format clean terminal prompt for human clarification."""
    lines = [
        f"\n[CLARIFICATION NEEDED - Round {round_num}/{max_rounds}]",
        f"  Clarification Question: {question}",
        f"  Why this matters:       {why_it_matters}",
    ]
    if options:
        lines.append("  Suggested Options:")
        for idx, opt in enumerate(options, 1):
            lines.append(f"    [{idx}] {opt}")
        lines.append("    [0] Skip / Just answer with all interpretations")
    lines.append("  Your Response (enter option number or free text):")
    return "\n".join(lines)


def generate_fallback_clarification(
    question: str,
    ambiguity_reason: str,
    ambiguity_type: str = "UNDERSPECIFIED_CONTEXT",
) -> Tuple[str, str, List[str]]:
    """Generate concise, deterministic clarification prompt when no curated template matches.
    
    Guarantees:
    - Question is <= 25 words.
    - Explicit explanation of missing context.
    - Standard actionable options.
    """
    q_lower = question.lower()

    if "programming language" in q_lower or "best language" in q_lower:
        t = CURATED_CLARIFICATIONS["programming_language"]
        return t["question"], t["why_it_matters"], t["options"]

    if "largest city" in q_lower or "biggest city" in q_lower:
        t = CURATED_CLARIFICATIONS["largest_city"]
        return t["question"], t["why_it_matters"], t["options"]

    for key, template in CURATED_CLARIFICATIONS.items():
        if re.search(r"\b" + key + r"\b", q_lower):
            return template["question"], template["why_it_matters"], template["options"]

    # Subjective superlatives (e.g., "What is the best automobile to buy?", "What is the greatest movie?")
    if any(w in q_lower for w in ["best", "greatest", "most important", "finest", "favorite"]):
        q_text = "What specific criteria or application should be prioritized for this evaluation?"
        why = "Without defined criteria, multiple valid perspectives and rankings exist."
        options = ["Performance / Quality", "Value / Cost", "Historical Impact", "Popularity", "Skip"]
        return q_text, why, options

    # Generic polysemy / underspecified
    q_text = "Which specific meaning, context, or timeframe would you like evaluated?"
    why = "The query admits multiple distinct valid interpretations or historical contexts."
    options = ["Most common modern meaning", "Technical definition", "Historical context", "Skip"]
    return q_text, why, options
