"""Candidate answer extractor and semantic final answer selection.

Selects the concise final answer using semantic clusters and medoid representatives
instead of naive exact-text voting. Detects ambiguity and uncertainty splits across
clusters to communicate uncertainty transparently.
"""

import re
from typing import Any, Dict, List, Optional
from src.preprocessing.text_cleaner import clean_text


def extract_concise_answer(text: str) -> str:
    """Extract a concise summary answer from a full response text.

    Preserves essential statements while trimming verbose preamble and subsequent paragraphs.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Split into paragraphs and pick the first substantial block
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    first_block = paragraphs[0] if paragraphs else cleaned

    # Pick the first non-empty line
    lines = [line.strip() for line in first_block.split("\n") if line.strip()]
    first_line = lines[0] if lines else first_block

    # Iteratively remove preamble phrases
    pattern = r"^(the answer is|in short|simply put|to answer your question|based on the available information|sure|certainly)[,:]?\s*"
    while True:
        new_line = re.sub(pattern, "", first_line, flags=re.IGNORECASE).strip()
        if new_line == first_line:
            break
        first_line = new_line

    return first_line.strip()


def select_semantic_final_answer(
    clusters: List[Any],
    majority_agreement: float,
    second_agreement: float,
    question: str = "",
) -> Dict[str, Any]:
    """Select final answer based on semantic clusters rather than exact-text voting.

    Args:
        clusters: List of SemanticCluster objects sorted by size descending.
        majority_agreement: Proportion of responses in majority cluster.
        second_agreement: Proportion of responses in second cluster.
        question: Original question string.

    Returns:
        Dict with keys:
            - 'final_answer': Final answer candidate or disambiguation statement.
            - 'is_split_decision': True if answers materially disagree.
            - 'explanation': Brief note explaining how the answer was selected.
    """
    if not clusters:
        return {
            "final_answer": "No answer could be determined from the responses.",
            "is_split_decision": True,
            "explanation": "No valid response clusters found.",
        }

    top_cluster = clusters[0]
    rep_text = top_cluster.representative_text
    concise_top = extract_concise_answer(rep_text)

    # Check if the responses are materially split across multiple interpretations
    # E.g., majority agreement is below 60%, or the second cluster is large (>= 30%)
    if len(clusters) > 1 and (majority_agreement < 0.60 or second_agreement >= 0.30):
        second_cluster = clusters[1]
        concise_second = extract_concise_answer(second_cluster.representative_text)

        # Build an informative clarification output
        split_msg = (
            f"The question has multiple distinct interpretations or conflicting answers:\n"
            f"  • Interpretation 1 ({top_cluster.size}/{sum(c.size for c in clusters)} responses): {concise_top}\n"
            f"  • Interpretation 2 ({second_cluster.size}/{sum(c.size for c in clusters)} responses): {concise_second}\n"
            f"Please clarify your specific intent."
        )
        return {
            "final_answer": split_msg,
            "is_split_decision": True,
            "explanation": (
                f"Material disagreement detected: majority cluster has {majority_agreement:.1%} agreement, "
                f"second cluster has {second_agreement:.1%} agreement."
            ),
        }

    # If majority cluster is dominant, use its representative medoid
    explanation = (
        f"Selected from dominant semantic cluster ({top_cluster.size}/{sum(c.size for c in clusters)} responses, "
        f"{majority_agreement:.1%} agreement)."
    )

    return {
        "final_answer": concise_top,
        "is_split_decision": False,
        "explanation": explanation,
    }
