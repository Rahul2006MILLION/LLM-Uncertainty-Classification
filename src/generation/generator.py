"""Response generator that coordinates single and multi-response LLM sampling."""

import time
import logging
from typing import List, Dict, Any, Optional
from src.generation.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a factual, concise question answering assistant. "
    "Provide a direct, accurate answer to the user's question without unnecessary filler."
)


class ResponseGenerator:
    """Manages stochastic generation of responses for uncertainty estimation."""

    def __init__(
        self,
        client: OllamaClient,
        model: str = "qwen2.5:3b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

    def generate_single(
        self,
        question: str,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        max_sample_retries: int = 2,
    ) -> Dict[str, Any]:
        """Generate a single response with automatic retry for empty outputs."""
        temp = self.temperature if temperature is None else temperature

        for attempt in range(max_sample_retries + 1):
            start_time = time.time()
            result = self.client.generate(
                prompt=question,
                model=self.model,
                system=self.system_prompt,
                temperature=temp,
                top_p=self.top_p,
                seed=seed,
            )
            elapsed = time.time() - start_time
            text = result.get("response", "").strip()

            # Check for empty response and retry if retries remain
            if not text:
                if attempt < max_sample_retries:
                    logger.warning(
                        "Ollama returned empty response for query '%s' (attempt %d/%d). Retrying...",
                        question[:40], attempt + 1, max_sample_retries + 1
                    )
                    time.sleep(0.5)
                    continue
                else:
                    logger.error(
                        "Ollama returned empty response for query '%s' after %d attempts.",
                        question[:40], max_sample_retries + 1
                    )

            return {
                "index": 0,
                "text": text,
                "model": self.model,
                "temperature": temp,
                "elapsed_seconds": elapsed,
                "eval_count": result.get("eval_count", 0),
            }

    def generate_multi(
        self,
        question: str,
        initial_response: Optional[Dict[str, Any]] = None,
        target_total_count: int = 10,
        stochastic_temperature: Optional[float] = None,
        greedy_temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate exactly target_total_count responses.

        Preserves canonical sampling methodology:
        - Response 0: Deterministic greedy generation at temperature=0.0 (or uses initial_response).
        - Responses 1..N-1: Stochastic sampling at temperature=0.7, top_p=0.9.
        """
        responses: List[Dict[str, Any]] = []
        stoch_temp = self.temperature if stochastic_temperature is None else stochastic_temperature

        # 1. Deterministic / initial sample at index 0 (T=0.0)
        if initial_response is not None:
            init_copy = dict(initial_response)
            init_copy["index"] = 0
            responses.append(init_copy)
        else:
            try:
                first = self.generate_single(question, temperature=greedy_temperature)
                first["index"] = 0
                responses.append(first)
            except Exception as e:
                logger.error("Failed to generate initial greedy response: %s", e)
                responses.append({
                    "index": 0,
                    "text": "",
                    "model": self.model,
                    "temperature": greedy_temperature,
                    "elapsed_seconds": 0.0,
                    "eval_count": 0,
                    "error": str(e),
                })

        # 2. Sequential stochastic samples at indices 1..(target_total_count - 1) (T=0.7, top_p=0.9)
        needed = target_total_count - len(responses)
        for i in range(needed):
            idx = len(responses)
            try:
                res = self.generate_single(question, temperature=stoch_temp)
                res["index"] = idx
                responses.append(res)
            except Exception as e:
                logger.error("Error generating stochastic response %d: %s", idx, e)
                responses.append({
                    "index": idx,
                    "text": "",
                    "model": self.model,
                    "temperature": stoch_temp,
                    "elapsed_seconds": 0.0,
                    "eval_count": 0,
                    "error": str(e),
                })

        return responses
