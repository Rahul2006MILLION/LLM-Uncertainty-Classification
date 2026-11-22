#!/usr/bin/env python3
"""Minimal HTTP adapter between the web frontend and the existing research pipeline.

This file is an integration shim ONLY. It contains no ML logic: every analysis
request is delegated unchanged to ``experiments/ask.py::run_pipeline`` (the same
function the project's CLI uses), and the resulting ``ExperimentRecord`` is
returned verbatim as JSON.

It uses only the Python standard library, so no new dependencies are required.

Run from the project root:

    .venv/bin/python webapp/server.py            # http://127.0.0.1:8765
    .venv/bin/python webapp/server.py --port 9000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import mimetypes
import os
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBAPP_DIR)
STATIC_DIR = os.path.join(WEBAPP_DIR, "static")

# The pipeline resolves configs/ and models/ relative to the project root.
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_ask_module():
    """Import experiments/ask.py (the existing CLI pipeline) without modifying it."""
    path = os.path.join(PROJECT_ROOT, "experiments", "ask.py")
    spec = importlib.util.spec_from_file_location("uncertainty_ask", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ask = _load_ask_module()

from src.classification.hitl_engine import ConversationState  # noqa: E402
from src.classification.hitl_prompts import (  # noqa: E402
    contains_sensitive_credentials,
    is_user_declining,
    sanitize_user_input,
)
from webapp.clarify import merge_clarification, suggest_interpretations  # noqa: E402
CONFIG_PATH = os.path.join("configs", "default.yaml")
MAX_QUESTION_CHARS = 2000

# Ollama runs one model at a time on a laptop; serialise pipeline runs so
# concurrent browser requests don't interleave generations.
_pipeline_lock = threading.Lock()


def _json_safe(value: Any) -> Any:
    """Replace NaN/Inf (invalid JSON) with None; recurse through containers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def health() -> Dict[str, Any]:
    """Read-only readiness probe: config, Ollama reachability, classifier file."""
    cfg = ask.load_config(CONFIG_PATH)
    ollama_cfg = cfg.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    gen_model = ollama_cfg.get("generation_model")
    emb_model = ollama_cfg.get("embedding_model")
    model_path = cfg.get("classifier", {}).get("model_save_path", "models/uncertainty_classifier.joblib")

    ollama_ok, available, err = False, [], None
    try:
        client = ask.OllamaClient(host=host, request_timeout=5.0)
        ollama_ok = bool(client.is_available())
    except Exception as e:  # pragma: no cover - environment dependent
        err = str(e)
    if ollama_ok:
        try:
            import requests

            r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
            available = [m.get("name") for m in r.json().get("models", [])]
        except Exception:
            available = []

    def _has(name):
        if not name or not available:
            return None
        return name in available or f"{name}:latest" in available

    return {
        "ok": ollama_ok and os.path.exists(model_path),
        "ollama": {"host": host, "reachable": ollama_ok, "error": err},
        "generation_model": {"name": gen_model, "installed": _has(gen_model)},
        "embedding_model": {"name": emb_model, "installed": _has(emb_model)},
        "classifier": {
            "path": model_path,
            "exists": os.path.exists(model_path),
            "type": cfg.get("classifier", {}).get("model_type"),
        },
        "multi_response_count": cfg.get("pipeline", {}).get("multi_response_count"),
        "fast_path_enabled": cfg.get("pipeline", {}).get("fast_path_enabled"),
    }


def _needs_clarification(record: Dict[str, Any]) -> bool:
    """The pipeline's own HITL signal: decision pending on the user."""
    return (
        record.get("clarification_decision") == "CLARIFICATION_REQUIRED"
        and record.get("interaction_status") == "PENDING"
    )


def analyze(
    question: str,
    force_multi: bool,
    clarification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the existing pipeline (with its HITL gates) and return its record unchanged.

    The pipeline's interactive HITL gates call ``clarification_callback`` when they
    decide clarification is required. In a web request we cannot block on terminal
    input, so the callback records the gate's decision and returns ``None`` — the
    pipeline then finishes exactly as it does when a CLI user skips the prompt, and
    the UI asks the user afterwards.

    When ``clarification`` is given, the clarified question is re-run through the same
    pipeline with a ``ConversationState`` recording the turn (the pipeline's own
    mechanism for clarified follow-ups).
    """
    config = ask.load_config(CONFIG_PATH)
    gate_decisions: List[Any] = []

    def record_gate(decision, _state):
        gate_decisions.append(decision)
        return None  # defer to the web UI

    state = None
    merge_method = None
    run_question = question
    if clarification:
        state = ConversationState(original_question=question, current_question=question)
        for prev in clarification.get("previous", [])[: state.max_rounds]:
            state.add_turn(
                clarification_prompt=str(prev.get("prompt", "")),
                user_response=sanitize_user_input(str(prev.get("text", ""))),
                consolidated_question=sanitize_user_input(str(prev.get("question", ""))) or question,
                ambiguity_type=str(prev.get("ambiguity_type") or ""),
            )
        text = sanitize_user_input(str(clarification.get("text", "")))
        proposed = sanitize_user_input(str(clarification.get("question", "")))
        with _pipeline_lock:
            if proposed and proposed.lower() != state.current_question.lower():
                run_question, merge_method = proposed, "suggestion"
            else:
                merged = merge_clarification(state.current_question, text, config)
                run_question, merge_method = merged["question"], merged["method"]
        state.add_turn(
            clarification_prompt=str(clarification.get("prompt", "")),
            user_response=text,
            consolidated_question=run_question,
            ambiguity_type=str(clarification.get("ambiguity_type") or ""),
        )

    with _pipeline_lock:
        record = ask.run_pipeline(
            question=run_question,
            config=config,
            force_multi_response=force_multi,
            interactive_hitl=True,
            conversation_state=state,
            clarification_callback=record_gate,
        )
        record_dict = _json_safe(record.model_dump())

        result: Dict[str, Any] = {"record": record_dict, "clarification": None}
        if merge_method:
            result["clarified"] = {"question": run_question, "method": merge_method}

        if _needs_clarification(record_dict):
            gate = gate_decisions[-1] if gate_decisions else None
            # Grounded on the question and the gate's own clarification prompt. Sampled
            # answers are deliberately not passed: they describe how the model answered,
            # not what the user meant, and they pull a small model toward generic readings.
            suggestions = suggest_interpretations(
                record_dict["question"],
                config,
                ambiguity_type=record_dict.get("ambiguity_type"),
                pipeline_prompt=getattr(gate, "clarification_prompt", None),
            )
            result["clarification"] = {
                "pipeline_prompt": getattr(gate, "clarification_prompt", None),
                "pipeline_options": list(getattr(gate, "options", []) or []),
                "why_it_matters": getattr(gate, "why_it_matters", None),
                "gate_ambiguity_type": getattr(getattr(gate, "ambiguity_type", None), "value", None),
                "round": record_dict.get("clarification_rounds", 0) + 1,
                "max_rounds": state.max_rounds if state else ConversationState("", "").max_rounds,
                "suggestions": suggestions,
            }
    return _json_safe(result)


class Handler(BaseHTTPRequestHandler):
    server_version = "UncertaintyAdapter/1.0"

    # ---- helpers -------------------------------------------------------
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        rel_path = rel_path.lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(STATIC_DIR, rel_path))
        if not full.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(full):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[adapter] %s - %s\n" % (self.address_string(), fmt % args))

    # ---- routes --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            try:
                self._send_json(HTTPStatus.OK, health())
            except Exception as e:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})
            return
        self._send_static(path)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/analyze":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be JSON."})
            return

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Please enter a question."})
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Question exceeds {MAX_QUESTION_CHARS} characters."})
            return

        clarification = payload.get("clarification")
        if clarification is not None:
            if not isinstance(clarification, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "clarification must be an object."})
                return
            text = str(clarification.get("text", "")).strip()
            if not text:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Please choose an interpretation or describe what you mean."})
                return
            if contains_sensitive_credentials(text):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "That looks like a password or key. Please don't include credentials in a clarification."})
                return
            if is_user_declining(text):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Use “Show analysis without clarifying” to skip clarification."})
                return

        started = time.time()
        try:
            result = analyze(question.strip(), bool(payload.get("force_multi", False)), clarification)
        except Exception as e:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": str(e) or e.__class__.__name__,
                    "error_type": e.__class__.__name__,
                    "elapsed_seconds": round(time.time() - started, 2),
                },
            )
            return
        self._send_json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the uncertainty-analysis web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Uncertainty UI → http://{args.host}:{args.port}  (project root: {PROJECT_ROOT})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
