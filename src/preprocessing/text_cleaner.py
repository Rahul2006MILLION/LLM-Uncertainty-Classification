"""Text cleaning and normalization that preserves negations, numbers, and key semantics."""

import re
from typing import List, Tuple

# Leading conversational filler patterns commonly produced by LLMs
BOILERPLATE_PREFIXES = [
    r"^sure,?\s*(here is (the answer|information|what you need):?|i can help with that:?)?",
    r"^certainly!?:?",
    r"^as an ai language model,?:?",
    r"^the answer is:?",
    r"^based on the (information|data),?:?",
    r"^to answer your question,?:?",
    r"^in short,?:?",
]

# Negation words that must NEVER be removed
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hardly", "scarcely", "barely"
}


def strip_boilerplate(text: str) -> str:
    """Remove conversational preamble while keeping the substantive answer."""
    cleaned = text.strip()
    for pattern in BOILERPLATE_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def clean_text_for_embedding(text: str) -> str:
    """Normalize text for embedding while preserving critical semantic tokens.
    
    Preserves:
    - Negations ('not', 'no', 'never', 'cannot')
    - Proper names and casing structure
    - Numbers and dates
    - Contrasts ('however', 'but')
    """
    if not text or not text.strip():
        return ""

    cleaned = strip_boilerplate(text)
    # Collapse multiple whitespaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Truncate to safe context length (~800 characters / ~150 words) to prevent embedding context overflow
    if len(cleaned) > 800:
        cleaned = cleaned[:800].rsplit(" ", 1)[0]
    return cleaned


def tokenize_preserving_negation(text: str) -> List[str]:
    """Tokenize into words preserving contractions and numbers."""
    if not text:
        return []
    # Matches words with apostrophes (e.g. don't) and numbers (e.g. 2, 2.5, 2024)
    tokens = re.findall(r"\b[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?\b", text.lower())
    return tokens
