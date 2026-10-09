"""Ollama LLM generation module.

Handles sequential generations, context window verification, and clean,
actionable error handling when Ollama is unavailable.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

from configs.config import DEFAULT_MODEL_CONFIG


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not found in Ollama."""
    pass


@dataclass
class GenerationResult:
    question: str
    model: str
    generation_index: int
    response: str
    temperature: float
    top_p: float
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    total_duration_sec: Optional[float] = None


def check_ollama_server(base_url: str = DEFAULT_MODEL_CONFIG.ollama_base_url) -> bool:
    """Check if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def verify_runtime_context(
    model: str = DEFAULT_MODEL_CONFIG.model_name,
    base_url: str = DEFAULT_MODEL_CONFIG.ollama_base_url,
) -> int:
    """Query Ollama API to verify the effective runtime context window configured for the model.

    Returns:
        The context window size in tokens (e.g., 1096).

    Raises:
        OllamaConnectionError: If Ollama is unreachable.
        OllamaModelNotFoundError: If the model is not found.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"name": model},
            timeout=10,
        )
    except requests.RequestException as e:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama server at {base_url}. "
            "Please ensure Ollama is running (`ollama serve`)."
        ) from e

    if resp.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{model}' is not installed in Ollama. "
            f"Available models can be checked with `ollama list`."
        )

    if resp.status_code != 200:
        raise OllamaError(
            f"Failed to inspect model '{model}': HTTP {resp.status_code} {resp.text}"
        )

    data = resp.json()
    parameters_text = data.get("parameters", "")
    context_size = DEFAULT_MODEL_CONFIG.expected_context_window

    for line in parameters_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "num_ctx":
            try:
                context_size = int(parts[1])
            except ValueError:
                pass

    return context_size


def generate_response(
    question: str,
    model: str = DEFAULT_MODEL_CONFIG.model_name,
    temperature: float = DEFAULT_MODEL_CONFIG.temperature,
    top_p: float = DEFAULT_MODEL_CONFIG.top_p,
    num_ctx: Optional[int] = None,
    base_url: str = DEFAULT_MODEL_CONFIG.ollama_base_url,
    timeout: int = DEFAULT_MODEL_CONFIG.timeout_seconds,
) -> str:
    """Generate a single response from Ollama with error handling and context configuration.

    Args:
        question: Prompt / question string.
        model: Target Ollama model name.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        num_ctx: Context window length (if None, model default is used).
        base_url: Ollama base URL.
        timeout: Request timeout in seconds.

    Returns:
        The generated response string.

    Raises:
        OllamaConnectionError: If Ollama is unreachable.
        OllamaModelNotFoundError: If the model is not found.
        OllamaError: For other Ollama API errors.
    """
    options: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = {
        "model": model,
        "prompt": question,
        "stream": False,
        "options": options,
    }

    url = f"{base_url}/api/generate"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise OllamaConnectionError(
            f"Cannot connect to Ollama server at {base_url}. "
            "Please ensure Ollama is running (`ollama serve`)."
        ) from e
    except requests.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    if resp.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{model}' not found in Ollama. Run `ollama pull {model}`."
        )

    if resp.status_code != 200:
        raise OllamaError(
            f"Ollama API returned HTTP {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    response_text = data.get("response", "").strip()
    return response_text


def generate_multiple_responses(
    question: str,
    model: str = DEFAULT_MODEL_CONFIG.model_name,
    num_generations: int = 10,
    temperature: float = DEFAULT_MODEL_CONFIG.temperature,
    top_p: float = DEFAULT_MODEL_CONFIG.top_p,
    num_ctx: Optional[int] = None,
    base_url: str = DEFAULT_MODEL_CONFIG.ollama_base_url,
    progress_callback: Optional[Any] = None,
) -> List[GenerationResult]:
    """Generate multiple independent responses sequentially.

    Sequential generation avoids creating multiple simultaneous model instances,
    respecting memory limits on Apple Silicon / local machines.
    """
    results: List[GenerationResult] = []

    for i in range(num_generations):
        if progress_callback:
            progress_callback(i + 1, num_generations)

        answer = generate_response(
            question=question,
            model=model,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx,
            base_url=base_url,
        )

        results.append(
            GenerationResult(
                question=question,
                model=model,
                generation_index=i + 1,
                response=answer,
                temperature=temperature,
                top_p=top_p,
            )
        )

    return results