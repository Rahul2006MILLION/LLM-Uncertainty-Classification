"""Text preprocessing module for semantic analysis.

Normalizes whitespace, markdown formatting, and superficial variations
while strictly preserving semantic distinctions such as negations ('not', 'no', 'never'),
proper names, numbers, dates, and contradictory claims.
"""

import re
from typing import List


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
