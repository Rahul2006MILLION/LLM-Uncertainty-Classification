"""Unit tests for Ollama generator, connection error handling, and context window verification."""

import unittest
from unittest.mock import patch, MagicMock
import requests

from src.generation.ollama_generator import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    generate_response,
    generate_multiple_responses,
    verify_runtime_context,
)


class TestOllamaGenerator(unittest.TestCase):
    def test_ollama_connection_error_handling(self):
        with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
            with self.assertRaises(OllamaConnectionError) as ctx:
                generate_response(question="Hello", model="llama3.1:8b-research")
            self.assertIn("Cannot connect to Ollama server", str(ctx.exception))
            self.assertIn("ollama serve", str(ctx.exception))

    def test_ollama_model_not_found_handling(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("requests.post", return_value=mock_resp):
            with self.assertRaises(OllamaModelNotFoundError) as ctx:
                generate_response(question="Hello", model="non-existent-model")
            self.assertIn("not found in Ollama", str(ctx.exception))

    def test_verify_runtime_context(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "parameters": "num_thread 4\nnum_ctx 1096\nstop <|eot_id|>"
        }
        with patch("requests.post", return_value=mock_resp):
            ctx = verify_runtime_context(model="llama3.1:8b-research")
            self.assertEqual(ctx, 1096)

    def test_empty_or_malformed_response_handling(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": ""}
        with patch("requests.post", return_value=mock_resp):
            res = generate_response(question="Hello", model="llama3.1:8b-research")
            self.assertEqual(res, "")


if __name__ == "__main__":
    unittest.main()
