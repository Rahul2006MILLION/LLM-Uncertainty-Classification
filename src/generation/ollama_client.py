"""Robust HTTP client for interacting with a local Ollama instance."""

import time
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client interactions."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama cannot be reached."""
    pass


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error or invalid payload."""
    pass


class OllamaClient:
    """Client for local Ollama API endpoints."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        request_timeout: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.host = host.rstrip("/")
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if the Ollama daemon is reachable and responding."""
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return names of available local models."""
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name") for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.warning("Failed to list Ollama models: %s", e)
            return []

    def get_version(self) -> str:
        """Return Ollama daemon version string."""
        try:
            resp = self.session.get(f"{self.host}/api/version", timeout=3.0)
            if resp.status_code == 200:
                return resp.json().get("version", "unknown")
            return "unknown"
        except Exception as e:
            logger.warning("Failed to get Ollama version: %s", e)
            return "unknown"

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Return model metadata including digest and details."""
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    m_name = m.get("name", "")
                    if m_name == model or m_name.split(":")[0] == model.split(":")[0]:
                        return {
                            "name": m.get("name"),
                            "model": m.get("model"),
                            "digest": m.get("digest"),
                            "size": m.get("size"),
                            "modified_at": m.get("modified_at"),
                            "details": m.get("details", {}),
                        }
            return {}
        except Exception as e:
            logger.warning("Failed to get model info for %s: %s", model, e)
            return {}

    def generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a response using /api/generate with retry logic.
        
        Returns a dict containing:
            - 'response': str
            - 'total_duration_ms': float
            - 'eval_count': int
            - 'model': str
            - 'raw': dict
        """
        url = f"{self.host}/api/generate"
        payload_options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            payload_options["seed"] = seed
        if options:
            payload_options.update(options)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": payload_options,
        }
        if system:
            payload["system"] = system

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                start_t = time.perf_counter()
                resp = self.session.post(url, json=payload, timeout=self.request_timeout)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "").strip()
                    eval_count = data.get("eval_count", 0)
                    total_dur = data.get("total_duration", 0) / 1e6 or elapsed_ms
                    return {
                        "response": response_text,
                        "total_duration_ms": total_dur,
                        "eval_count": eval_count,
                        "model": model,
                        "raw": data,
                    }
                else:
                    raise OllamaResponseError(
                        f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                    )
            except requests.exceptions.ConnectionError as ce:
                last_err = OllamaConnectionError(
                    f"Could not connect to Ollama at {self.host}. Is `ollama serve` running?"
                )
            except requests.exceptions.Timeout as te:
                last_err = OllamaClientError(
                    f"Request timed out after {self.request_timeout}s on attempt {attempt + 1}"
                )
            except Exception as e:
                last_err = e

            wait_time = self.backoff_factor ** attempt
            logger.warning(
                "Generation attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt + 1, self.max_retries, last_err, wait_time
            )
            time.sleep(wait_time)

        raise last_err or OllamaClientError("Failed to generate response after retries")

    def embed(self, text: str, model: str = "all-minilm") -> List[float]:
        """Generate semantic embedding vector for text using /api/embeddings."""
        url = f"{self.host}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text if text.strip() else " ",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.request_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data.get("embedding")
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                    raise OllamaResponseError(f"Malformed embedding received: {data}")
                raise OllamaResponseError(f"HTTP {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                last_err = OllamaConnectionError(f"Could not connect to Ollama at {self.host}")
            except Exception as e:
                last_err = e

            time.sleep(self.backoff_factor ** attempt)

        raise last_err or OllamaClientError(f"Failed to embed text using model '{model}'")
