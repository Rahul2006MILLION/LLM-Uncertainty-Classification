# Web UI

An interactive frontend for the uncertainty-classification pipeline.

```
webapp/
├── server.py        # stdlib-only HTTP adapter → experiments/ask.py::run_pipeline
├── clarify.py       # isolated helpers: interpretation suggestions + question merge
└── static/          # index.html, styles.css, app.js (no build step)
```

## Run

Ollama must be running with the models from `configs/default.yaml` pulled. Then, from the project root:

```bash
.venv/bin/python webapp/server.py
```

Then open http://127.0.0.1:8765.

## How it connects

| Endpoint | What it does |
| --- | --- |
| `GET /api/health` | Read-only checks: is Ollama reachable, are the configured models installed, and does the classifier file exist. |
| `POST /api/analyze` `{"question", "force_multi", "clarification"?}` | Calls the existing `run_pipeline(...)` with its HITL gates enabled. Returns `{"record": ExperimentRecord, "clarification": {...} \| null, "clarified"?}`; the record is passed through unchanged. |

The adapter contains no ML logic and changes nothing under `src/`, `experiments/`, `configs/` or `models/`.
It does not write `experiments/results/ask_*.json` records; only the CLI does that.
Pipeline failures, such as Ollama being down, come back as HTTP 502 with the real error message, and the UI shows that error instead of results.

## Human clarification flow

1. **Signal.** The pipeline runs with `interactive_hitl=True`, as the CLI does by default. When a HITL gate decides clarification is needed, it calls `clarification_callback`. The adapter records the gate's decision (its prompt, options and reason) and returns `None`, so the pipeline finishes as if a CLI user had skipped the prompt.
2. **Trigger.** If the record says `clarification_decision == "CLARIFICATION_REQUIRED"` and `interaction_status == "PENDING"`, the UI shows the clarification panel instead of the answer. Every other outcome renders normally, including KNOWN, UNKNOWN (`ABSTAIN`), and AMBIGUOUS without a pending decision.
3. **Suggestions.** `clarify.suggest_interpretations` asks the configured Ollama generation model for 2–3 readings of the question. Its input is the question plus the gate's own clarification prompt. If the model fails, the UI falls back to the gate's own `options`.
4. **Re-run.** The user picks an interpretation or types their own clarification. Picked suggestions come with a rewritten question. For typed text, `clarify.merge_clarification` asks the model to rewrite the question; if the rewrite isn't usable, it falls back to the pipeline's `"<question> [Context: …]"` convention. The adapter builds a `ConversationState`, records the turn with `add_turn`, and re-runs `run_pipeline(clarified_question, conversation_state=…)`. The pipeline's own fields (`is_clarified`, `original_question`, `clarified_question`, `interaction_status: RESOLVED`) drive the results view. The pipeline's limit of 2 rounds still applies.
5. **Safety.** Clarification text goes through the pipeline's `sanitize_user_input`. Requests containing credentials are rejected (`contains_sensitive_credentials`). Typing "skip" or similar is redirected to the "Show analysis without clarifying" button.
