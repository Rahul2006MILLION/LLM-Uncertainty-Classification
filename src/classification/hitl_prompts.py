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
]

# Decline/skip intent patterns in user clarification response
DECLINE_PATTERNS = [
    r"^(skip|cancel|decline|none|neither|no|pass|n/a|stop|exit|quit|never\s*mind|just\s*answer|show\s*both)$",
    r"^(i\s*don'?t\s*(care|know)|doesn'?t\s*matter|either\s*is\s*fine|any|whatever)$",
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


def sanitize_user_input(text: str) -> str:
    """Sanitize user clarification input to prevent prompt injection and remove system delimiters."""
    if not text:
        return ""
    # Strip dangerous characters, control characters, and markdown command injections
    cleaned = text.strip()
    # Remove system/inst instruction tags if present
    cleaned = re.sub(r"\[/?(?:INST|SYS|SYSTEM|USER|ASSISTANT)\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<(?:system|user|assistant|tool_call)>.*?</(?:system|user|assistant|tool_call)>", "", cleaned, flags=re.IGNORECASE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Limit length to 200 characters to prevent buffer bloating
    if len(cleaned) > 200:
        cleaned = cleaned[:200].strip()
    return cleaned


def is_user_declining(user_input: str) -> bool:
    """Determine whether the user is explicitly declining or skipping clarification."""
    cleaned = user_input.strip().lower()
    return any(re.search(pat, cleaned) for pat in DECLINE_PATTERNS)


def contains_sensitive_credentials(text: str) -> bool:
    """Check if text appears to contain sensitive credentials, passwords, or tokens."""
    t_lower = text.lower()
    return any(re.search(pat, t_lower) for pat in SENSITIVE_CREDENTIAL_PATTERNS)


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
