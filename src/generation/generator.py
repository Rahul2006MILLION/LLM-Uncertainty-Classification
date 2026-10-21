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
    ) -> Dict[str, Any]:
        """Generate a single initial response."""
        temp = self.temperature if temperature is None else temperature
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
        return {
            "index": 0,
            "text": result["response"],
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
    ) -> List[Dict[str, Any]]:
        """Generate exactly target_total_count responses.
        
        If initial_response is provided, uses it as index 0 and generates
        (target_total_count - 1) additional sequential stochastic responses.
        """
        responses: List[Dict[str, Any]] = []

        if initial_response is not None:
            init_copy = dict(initial_response)
            init_copy["index"] = 0
            responses.append(init_copy)
        else:
            first = self.generate_single(question)
            responses.append(first)

        needed = target_total_count - len(responses)
        for i in range(needed):
            idx = len(responses)
            try:
                res = self.generate_single(question)
                res["index"] = idx
                responses.append(res)
            except Exception as e:
                logger.error("Error generating response %d: %s", idx, e)
                # Fallback empty string marked so clustering handles cleanly
                responses.append({
                    "index": idx,
                    "text": "",
                    "model": self.model,
                    "temperature": self.temperature,
                    "elapsed_seconds": 0.0,
                    "eval_count": 0,
                    "error": str(e),
                })

        return responses
