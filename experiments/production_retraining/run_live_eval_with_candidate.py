#!/usr/bin/env python3
"""Live 107-case evaluation with the candidate classifier swapped in — in this process only.

No production file is touched: the production model, config, pipeline and web adapter are
unchanged. Inside this process, `UncertaintyClassifier.load` is redirected to the candidate
artifact, so both the pipeline (experiments/ask.py) and the evaluation runner's raw re-scoring
use the candidate. Everything else (generation, features, safeguards, HITL gates, adapter) is
the normal code path.

    .venv/bin/python experiments/production_retraining/run_live_eval_with_candidate.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

CANDIDATE = os.path.join(ROOT, "models", "candidates", "uncertainty_classifier_f300_candidate.joblib")
OUT = os.path.join(HERE, "results", "live_eval_candidate.json")

from src.classification.classifier import UncertaintyClassifier  # noqa: E402

_original_load = UncertaintyClassifier.load.__func__
loaded_paths = []


def _load_candidate(cls, filepath):
    loaded_paths.append(filepath)
    clf = _original_load(cls, CANDIDATE)
    if clf.metadata.get("experiment") != "production_retraining":
        raise RuntimeError("Candidate artifact not loaded")
    return clf


UncertaintyClassifier.load = classmethod(_load_candidate)

sys.argv = [os.path.join(ROOT, "tests", "evaluation", "run_model_evaluation.py"), "--out", OUT]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print(f"[candidate-swap] classifier load requests redirected to candidate: {len(loaded_paths)} "
          f"(requested paths: {sorted(set(loaded_paths))})")
