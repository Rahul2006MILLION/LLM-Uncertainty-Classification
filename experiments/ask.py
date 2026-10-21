#!/usr/bin/env python3
"""Interactive and CLI Adaptive QA Pipeline for LLM Uncertainty Classification."""

import sys
import os
import time
import json
import argparse
import datetime
import logging
from typing import Dict, Any, List, Optional
import yaml

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.ollama_client import OllamaClient, OllamaClientError
from src.generation.generator import ResponseGenerator
from src.preprocessing.embeddings import EmbeddingEngine
from src.features.clustering import SemanticClusterer
from src.features.entropy import compute_semantic_entropy_metrics
from src.features.feature_extractor import FeatureExtractor
from src.classification.router import AdaptiveRouter
from src.classification.classifier import UncertaintyClassifier
from data.schema import ExperimentRecord

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ask")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration YAML with safe fallbacks."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def format_terminal_box(title: str, lines: List[str], width: int = 70) -> str:
    """Format clean ASCII box for readable terminal display."""
    border = "=" * width
    sub_border = "-" * width
    out = [border, f"  {title.center(width - 4)}", sub_border]
    for line in lines:
        out.append(f"  {line}")
    out.append(border)
    return "\n".join(out)


def run_pipeline(
    question: str,
    config: Optional[Dict[str, Any]] = None,
    force_multi_response: bool = False,
    generator: Optional[Any] = None,
    classifier: Optional[Any] = None,
    router: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> ExperimentRecord:
    """Execute the end-to-end adaptive QA pipeline on a given question."""
    start_total_t = time.time()
    cfg = config or load_config()

    ollama_cfg = cfg.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    gen_model = ollama_cfg.get("generation_model", "qwen2.5:3b")
    emb_model = ollama_cfg.get("embedding_model", "all-minilm")
    temperature = float(ollama_cfg.get("temperature", 0.7))
    top_p = float(ollama_cfg.get("top_p", 0.9))

    pipe_cfg = cfg.get("pipeline", {})
    fast_path_enabled = pipe_cfg.get("fast_path_enabled", True) and not force_multi_response
    multi_count = int(pipe_cfg.get("multi_response_count", 10))

    clust_cfg = cfg.get("clustering", {})
    sim_threshold = float(clust_cfg.get("similarity_threshold", 0.80))
    linkage = clust_cfg.get("linkage", "average")

    out_cfg = cfg.get("output_selection", {})
    expl_threshold = float(out_cfg.get("explanation_threshold", 0.25))

    clf_cfg = cfg.get("classifier", {})
    model_save_path = clf_cfg.get("model_save_path", "models/uncertainty_classifier.joblib")

    # 1. Initialize Clients and Modules
    client = None
    if generator is None or embedder is None:
        client = OllamaClient(host=host, request_timeout=float(ollama_cfg.get("request_timeout", 45.0)))

    if generator is None:
        if not client.is_available():
            raise OllamaClientError(
                f"Cannot connect to Ollama at {host}. Please verify that the Ollama service is running."
            )
        generator = ResponseGenerator(client=client, model=gen_model, temperature=temperature, top_p=top_p)

    if router is None:
        router = AdaptiveRouter(fast_path_enabled=fast_path_enabled)

    clusterer = SemanticClusterer(similarity_threshold=sim_threshold, linkage=linkage)
    feature_extractor = FeatureExtractor(clusterer=clusterer)
    if embedder is None:
        embedder = EmbeddingEngine(client=client, embedding_model=emb_model)

    # Load classifier if present and not provided
    if classifier is None:
        for candidate_path in [model_save_path, "models/uncertainty_classifier.joblib", "models/uncertainty_classifier.json"]:
            if os.path.exists(candidate_path):
                try:
                    classifier = UncertaintyClassifier.load(candidate_path)
                    break
                except Exception as e:
                    logger.warning("Could not load classifier from %s: %s", candidate_path, e)

    # 2. Step 1: Initial Response Generation
    t0 = time.time()
    initial_res = generator.generate_single(question, temperature=0.0)  # Greedy for initial response
    initial_text = initial_res["text"]

    # 3. Step 2: Route question
    decision = router.route(question, initial_text)

    # 4. Path Branching
    if decision.path == "FAST_PATH":
        elapsed_total = time.time() - start_total_t
        # Extract genuine single-response features
        single_emb = embedder.embed_batch([initial_text])
        features = feature_extractor.extract_features(
            question=question,
            responses=[initial_text],
            embeddings=single_emb,
        )

        clf_status = "CLASSIFIER NOT TRAINED"
        pred_label = None
        pred_conf = None
        if classifier is not None and classifier.is_trained:
            pred_res = classifier.predict(features)
            clf_status = pred_res.get("status", "PREDICTED")
            pred_label = pred_res.get("predicted_label")
            pred_conf = pred_res.get("confidence")

        rec = ExperimentRecord(
            timestamp=datetime.datetime.now().isoformat(),
            question=question,
            path="FAST_PATH",
            final_answer=initial_text,
            response_count=1,
            cluster_count=1,
            dominant_cluster_agreement=1.0,
            second_cluster_agreement=0.0,
            agreement_margin=1.0,
            semantic_entropy=0.0,
            normalized_entropy=0.0,
            mean_pairwise_similarity=1.0,
            elapsed_seconds=elapsed_total,
            classifier_status=clf_status,
            predicted_label=pred_label,
            confidence=pred_conf,
            features=features,
            responses=[initial_res],
            clusters={"labels": [0], "cluster_counts": [1], "dominant_rep_idx": 0, "second_rep_idx": None},
            model=gen_model,
            embedding_model=emb_model,
        )
        return rec

    # 5. Multi-Response Path: exactly 10 responses total
    all_responses = generator.generate_multi(
        question=question,
        initial_response=initial_res,
        target_total_count=multi_count,
    )
    response_texts = [r["text"] for r in all_responses]

    # 6. Preprocessing and Semantic Embeddings
    embeddings = embedder.embed_batch(response_texts)

    # 7. Semantic Clustering
    cluster_res = clusterer.cluster_responses(response_texts, embeddings)
    cluster_counts = cluster_res["cluster_counts"]
    dom_rep_idx = cluster_res["dominant_representative_idx"]
    sec_rep_idx = cluster_res["second_representative_idx"]

    # 8. Entropy, Agreement & Features
    features = feature_extractor.extract_features(
        question=question,
        responses=response_texts,
        embeddings=embeddings,
        cluster_result=cluster_res,
    )

    # 9. Output Selection & Synthesis
    dom_text = response_texts[dom_rep_idx] if dom_rep_idx >= 0 and dom_rep_idx < len(response_texts) else initial_text
    p1 = features["dominant_agreement"]
    p2 = features["second_agreement"]

    if p2 >= expl_threshold and sec_rep_idx is not None and sec_rep_idx < len(response_texts):
        sec_text = response_texts[sec_rep_idx]
        final_answer = (
            f"Multiple plausible interpretations detected:\n"
            f"  • [Interpretation 1 ({p1:.0%} consensus)]: {dom_text}\n"
            f"  • [Interpretation 2 ({p2:.0%} consensus)]: {sec_text}"
        )
    else:
        final_answer = dom_text

    # 10. Classifier Inference
    clf_status = "CLASSIFIER NOT TRAINED"
    pred_label = None
    pred_conf = None
    if classifier is not None and classifier.is_trained:
        pred_res = classifier.predict(features)
        clf_status = pred_res.get("status", "PREDICTED")
        pred_label = pred_res.get("predicted_label")
        pred_conf = pred_res.get("confidence")

    elapsed_total = time.time() - start_total_t

    # Serialize experiment record
    rec = ExperimentRecord(
        timestamp=datetime.datetime.now().isoformat(),
        question=question,
        path="MULTI_RESPONSE",
        final_answer=final_answer,
        response_count=len(all_responses),
        cluster_count=int(features["cluster_count"]),
        dominant_cluster_agreement=p1,
        second_cluster_agreement=p2,
        agreement_margin=features["agreement_margin"],
        semantic_entropy=features["semantic_entropy"],
        normalized_entropy=features["normalized_entropy"],
        mean_pairwise_similarity=features["mean_pairwise_similarity"],
        elapsed_seconds=elapsed_total,
        classifier_status=clf_status,
        predicted_label=pred_label,
        confidence=pred_conf,
        features=features,
        responses=all_responses,
        clusters={
            "labels": cluster_res["labels"],
            "cluster_counts": cluster_counts,
            "dominant_rep_idx": dom_rep_idx,
            "second_rep_idx": sec_rep_idx,
        },
        model=gen_model,
        embedding_model=emb_model,
    )
    return rec


def main():
    parser = argparse.ArgumentParser(description="Ask questions using adaptive LLM uncertainty pipeline.")
    parser.add_argument("question", nargs="*", help="Question to ask (if omitted, prompts interactively)")
    parser.add_argument("-q", "--query", type=str, default=None, help="Question text")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--force-multi", action="store_true", help="Force multi-response analysis (bypass fast path)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON record to stdout")
    args = parser.parse_args()

    # Determine question
    q_tokens = args.question
    question = args.query or (" ".join(q_tokens) if q_tokens else None)

    if not question:
        try:
            print("\nEnter question (or Ctrl+C to exit):")
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not question:
        print("Error: Empty question provided.")
        sys.exit(1)

    config = load_config(args.config)

    try:
        record = run_pipeline(question=question, config=config, force_multi_response=args.force_multi)
    except Exception as e:
        print(f"\n[PIPELINE ERROR]: {e}")
        sys.exit(1)

    # Save timestamped experiment record
    results_dir = config.get("storage", {}).get("results_dir", "experiments/results")
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = os.path.join(results_dir, f"ask_{ts_str}.json")
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)

    if args.json:
        print(json.dumps(record.model_dump(), indent=2))
        return

    # Print clean terminal report
    lines = [
        f"QUESTION:          {record.question}",
        f"EXECUTION PATH:    {record.path}",
        f"TOTAL RESPONSES:   {record.response_count}",
        f"CLUSTERS FOUND:    {record.cluster_count}",
        f"DOMINANT AGREEMENT:{record.dominant_cluster_agreement:.1%}",
        f"SECOND AGREEMENT:  {record.second_cluster_agreement:.1%}",
        f"AGREEMENT MARGIN:  {record.agreement_margin:.1%}",
        f"SEMANTIC ENTROPY:  {record.semantic_entropy:.3f} nats (Norm: {record.normalized_entropy:.3f})",
        f"MEAN PAIRWISE SIM: {record.mean_pairwise_similarity:.3f}",
        f"ELAPSED TIME:      {record.elapsed_seconds:.2f}s",
        f"CLASSIFIER STATUS: {record.classifier_status}",
    ]
    if record.predicted_label:
        lines.append(f"PREDICTED CLASS:   {record.predicted_label} (conf: {record.confidence:.2f})")
    lines.append(f"SAVED RECORD:      {out_filename}")
    lines.append("")
    lines.append("FINAL ANSWER:")
    # Indent answer lines
    for al in record.final_answer.split("\n"):
        lines.append(f"  {al}")

    print("\n" + format_terminal_box("ADAPTIVE QA UNCERTAINTY PIPELINE", lines) + "\n")


if __name__ == "__main__":
    main()
