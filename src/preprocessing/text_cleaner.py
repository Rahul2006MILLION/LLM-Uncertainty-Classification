"""Text cleaning and normalization module preserving negations, numbers, and key semantics.

Provides both modern pipeline cleaners (preserving negations and syntactic boundaries)
and backward-compatible normalization utilities for answer extraction and comparison.
"""

import re
from typing import List

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


# ============================================================================
# Backward-Compatible Legacy Normalization Utilities
# ============================================================================


def clean_text(text: str) -> str:
    """Normalize whitespace and superficial formatting without altering semantics.

    Args:
        text: Raw input string from the user or model generation.

    Returns:
        Cleaned, normalized string retaining all semantic distinctions.
    """
    if not text:
        return ""

    # Replace carriage returns with standard newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown code block fences if present (preserving inner text)
    text = re.sub(r"```[a-zA-Z0-9_]*\n?", "", text)

    # Normalize bullet points and list markers
    text = re.sub(r"^\s*[\*\-\•]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[\.\)]\s+", "1. ", text, flags=re.MULTILINE)

    # Normalize excessive blank lines while preserving paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize horizontal whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)

    # Strip empty leading/trailing newlines
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def normalize_for_comparison(text: str) -> str:
    """Prepare text for lexical comparison while retaining negations and numbers.

    Note: This is used for lexical statistics (token counts, overlap),
    NOT for sentence embeddings which operate best on natural cased text.
    """
    if not text:
        return ""

    text = clean_text(text).lower()
    # Retain alphanumeric characters, hyphens, and whitespace
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = [t for t in text.split() if t]
    return " ".join(tokens)


def tokenize_words(text: str) -> List[str]:
    """Tokenize into lowercase word tokens for lexical diversity/overlap metrics."""
    normalized = normalize_for_comparison(text)
    return normalized.split() if normalized else []
