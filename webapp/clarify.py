"""Clarification helpers for the web adapter (isolated from the research pipeline).

The research pipeline decides *whether* clarification is required (its HITL gates
set ``clarification_decision == "CLARIFICATION_REQUIRED"``). This module only
helps the web UI *ask* for it:

* ``suggest_interpretations`` asks the configured local Ollama model for distinct,
  question-specific readings, so the UI can offer one-click options.
* ``merge_clarification`` asks the same model to fold the user's clarification into
  a single self-contained question.

Neither function classifies anything or touches features, clustering or models.
Every model output is validated, and if the model fails we fall back to the
pipeline's own HITL options and its ``"<question> [Context: ...]"`` convention.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.classification.hitl_prompts import sanitize_user_input
from src.generation.ollama_client import OllamaClient

MAX_SUGGESTIONS = 4

_SUGGEST_SYSTEM = (
    "You help disambiguate questions. You never answer the question itself. "
    "You reply with JSON only, no prose and no markdown."
)

_SUGGEST_PROMPT = """A question was flagged as ambiguous ({ambiguity}). List the {n} most likely DISTINCT, real things the asker could mean.
For each interpretation give:
- "label": the specific, well-known name of what they might mean (max 6 words)
- "detail": a short factual description that tells it apart (max 12 words)
- "question": the original question rewritten to be unambiguous, naturally naming that specific thing

Example
Question: "What is Mercury?"
{{"interpretations": [
 {{"label": "Mercury, the planet", "detail": "Smallest planet, closest to the Sun", "question": "What is the planet Mercury?"}},
 {{"label": "Mercury, the chemical element", "detail": "Liquid metal with symbol Hg", "question": "What is the chemical element mercury?"}},
 {{"label": "Freddie Mercury", "detail": "Lead singer of the band Queen", "question": "Who was Freddie Mercury?"}}
]}}

Now do the same.
Question: "{question}"
{hints}
Reply with JSON only."""

_MERGE_SYSTEM = (
    "You rewrite a question so that it includes the asker's clarification. "
    "You never answer the question. Reply with the rewritten question only, on one line."
)

_MERGE_PROMPT = """Original question: "{question}"
Asker's clarification: "{clarification}"
Rewrite the original question as one natural, self-contained question that includes the clarification.
Example: "Who is Jordan?" + "Michael Jordan, the basketball player" -> "Who is Michael Jordan, the basketball player?"
Rewritten question:"""


def _client(cfg: Dict[str, Any]) -> tuple[OllamaClient, str]:
    oc = cfg.get("ollama", {})
    client = OllamaClient(
        host=oc.get("host", "http://localhost:11434"),
        request_timeout=float(oc.get("request_timeout", 45.0)),
        max_retries=1,
    )
    return client, oc.get("generation_model", "qwen2.5:3b")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull JSON out of a model reply (tolerates code fences and a bare list)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start < 0 or end <= start:
            continue
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return {"interpretations": data}
        if isinstance(data, dict):
            return data
    return None


def _clean(value: Any, limit: int) -> str:
    return sanitize_user_input(str(value or ""))[:limit].strip().strip('"').strip()


def _valid_question(q: str, original: str) -> bool:
    """A usable rewrite: one line, changed, and not a meta question back to the user."""
    return (
        8 <= len(q) <= 240
        and "\n" not in q
        and q.lower() != original.strip().lower()
        and not re.match(r"^(are you|do you|did you|would you|is it|which (one|meaning))\b", q, re.IGNORECASE)
    )


def suggest_interpretations(
    question: str,
    cfg: Dict[str, Any],
    ambiguity_type: Optional[str] = None,
    pipeline_prompt: Optional[str] = None,
    sample_answers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return {"items": [...], "source": "model" | "none", "error": str | None}."""
    hints = []
    if pipeline_prompt:
        hints.append(f'The pipeline\'s own clarification question was: "{pipeline_prompt}"')
    for a in (sample_answers or [])[:2]:
        hints.append(f'One sampled answer was: "{a[:220]}"')
    prompt = _SUGGEST_PROMPT.format(
        question=sanitize_user_input(question),  # treat the question as data, not instructions
        ambiguity=(ambiguity_type or "multiple interpretations").replace("_", " ").lower(),
        hints="\n".join(hints),
        n=MAX_SUGGESTIONS - 1,
    )
    try:
        client, model = _client(cfg)
        raw = client.generate(prompt, model=model, system=_SUGGEST_SYSTEM, temperature=0.2, seed=7,
                              options={"num_predict": 400})["response"]
    except Exception as e:  # pragma: no cover - depends on Ollama
        return {"items": [], "source": "none", "error": f"Suggestion model unavailable: {e}"}

    data = _extract_json(raw) or {}
    items: List[Dict[str, str]] = []
    seen = set()
    for it in data.get("interpretations", []) if isinstance(data.get("interpretations"), list) else []:
        if not isinstance(it, dict):
            continue
        label = _clean(it.get("label"), 60)
        detail = _clean(it.get("detail"), 120)
        q = _clean(it.get("question"), 240)
        # The UI always offers its own "Something else" option.
        if not label or label.lower() in seen or re.match(r"^(an)?other\b|^something else|^none\b", label, re.IGNORECASE):
            continue
        seen.add(label.lower())
        items.append({"label": label, "detail": detail, "question": q if _valid_question(q, question) else ""})
        if len(items) >= MAX_SUGGESTIONS - 1:
            break
    if not items:
        return {"items": [], "source": "none", "error": "The model did not return usable interpretations."}
    return {"items": items, "source": "model", "model": model, "error": None}


def merge_clarification(question: str, clarification: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    """Fold a clarification into the question. Returns {"question", "method"}."""
    fallback = f"{question.strip()} [Context: {clarification}]"  # pipeline's HITL convention
    try:
        client, model = _client(cfg)
        raw = client.generate(
            _MERGE_PROMPT.format(question=sanitize_user_input(question), clarification=sanitize_user_input(clarification)),
            model=model, system=_MERGE_SYSTEM, temperature=0.0, seed=7, options={"num_predict": 80},
        )["response"]
    except Exception:
        return {"question": fallback, "method": "context_block"}
    line = _clean(raw.strip().splitlines()[0] if raw.strip() else "", 240)
    line = re.sub(r"^(rewritten question\s*:\s*)", "", line, flags=re.IGNORECASE).strip().strip('"')
    if not _valid_question(line, question) or not line.endswith("?"):
        return {"question": fallback, "method": "context_block"}
    return {"question": line, "method": "model_rewrite"}
