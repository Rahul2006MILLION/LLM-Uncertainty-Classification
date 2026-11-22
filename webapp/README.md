# Web UI

An interactive frontend for the uncertainty-classification pipeline.

```
webapp/
├── server.py        # stdlib-only HTTP adapter → experiments/ask.py::run_pipeline
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
| `POST /api/analyze` `{"question": str, "force_multi": bool}` | Calls the existing `run_pipeline(question, config, force_multi_response, interactive_hitl=False)` and returns `{"record": ExperimentRecord.model_dump()}` unchanged. |

The adapter contains no ML logic and changes nothing under `src/`, `experiments/`, `configs/` or `models/`.
Human-in-the-loop clarification is turned off for web requests because it relies on interactive terminal prompts.
The adapter also does not write `experiments/results/ask_*.json` records; only the CLI does that.
Pipeline failures, such as Ollama being down, come back as HTTP 502 with the real error message, and the UI shows that error instead of results.
