"""Adaptive LLM Uncertainty Classification - Interactive CLI.

Executes the research prototype:
- Verifies Ollama availability and effective runtime context window.
- Fast path: Direct conversational handling for greetings and single-response fast-path
  for straightforward, unambiguous questions.
- Multi-response path: Sequentially generates 9 additional responses (10 total) when
  material uncertainty or ambiguity is flagged.
- Transparent NLP pipeline: Sentence embeddings, cosine similarity matrix, semantic clustering,
  empirical semantic entropy, and medoid-based final candidate selection.
- Supervised classification: Integrates trained uncertainty classifier or explicitly reports
  'CLASSIFIER NOT TRAINED' if no validated model weights exist.
"""

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_NLP_CONFIG,
    DEFAULT_PATH_CONFIG,
)
from src.generation.ollama_generator import (
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    check_ollama_server,
    generate_response,
    generate_multiple_responses,
    verify_runtime_context,
)
from src.features.semantic_analysis import analyze_semantic_responses
from src.features.answer_extractor import (
    extract_concise_answer,
    select_semantic_final_answer,
)
from src.classification.classifier import UncertaintyClassifier


GREETING_PATTERNS = [
    r"^hello[\s!\.]*$",
    r"^hi[\s!\.]*$",
    r"^hey[\s!\.]*$",
    r"^greetings[\s!\.]*$",
    r"^good (morning|afternoon|evening|day)[\s!\.]*$",
    r"^how are you[\s\?]*$",
    r"^who are you[\s\?]*$",
    r"^thank you[\s!\.]*$",
    r"^thanks[\s!\.]*$",
    r"^bye[\s!\.]*$",
    r"^goodbye[\s!\.]*$",
]


def is_conversational_input(text: str) -> bool:
    """Detect if an input is simple conversational small-talk / greeting."""
    cleaned = text.strip().lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, cleaned):
            return True
    return False


def assess_uncertainty(
    question: str,
    initial_answer: str,
    model: str = DEFAULT_MODEL_CONFIG.model_name,
    context_window: Optional[int] = None,
) -> Tuple[str, str]:
    """Strictly evaluate whether an initial response requires multi-response analysis.

    Returns:
        (decision, reason) where decision is 'FAST_PATH' or 'ANALYZE_10'.
    """
    prompt = f"""You are an objective uncertainty evaluator for an LLM research experiment.

Analyze the question and initial answer below:

Question:
{question}

Initial Answer:
{initial_answer}

Evaluation Criteria:
1. Choose FAST_PATH if:
   - The question has a single clear, well-defined meaning.
   - The answer directly and coherently answers the question.
   - It is a standard definition, established factual calculation, or straightforward concept.
   - There are no major contradictions, hallucinations, or expressions of severe lack of knowledge.
   - NOTE: Having several examples, detailed steps, or a long explanation is completely NORMAL for a definition and does NOT justify ANALYZE_10.

2. Choose ANALYZE_10 if:
   - The question has multiple materially different interpretations (e.g. ambiguity in proper names or geographical entities).
   - The answer explicitly states it lacks the necessary information or cannot answer.
   - The answer contains contradictory factual assertions.
   - The question asks about rapidly changing or current events that cannot be reliably verified.
   - The model expresses substantial uncertainty affecting its core conclusion.

Output Format:
First line: FAST_PATH or ANALYZE_10
Second line: A concise reason explaining your choice.
"""
    try:
        raw_output = generate_response(
            question=prompt,
            model=model,
            temperature=0.0,
            top_p=1.0,
            num_ctx=context_window,
        ).strip()
    except OllamaError:
        # On evaluation error, default safely to ANALYZE_10
        return "ANALYZE_10", "Error communicating with uncertainty assessor model."

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return "ANALYZE_10", "Empty assessment from assessor."

    first_line_upper = lines[0].upper()
    reason = " ".join(lines[1:]) if len(lines) > 1 else "Assessment completed."

    if "FAST_PATH" in first_line_upper:
        return "FAST_PATH", reason or "Clear question and coherent response with no uncertainty flags."
    if "ANALYZE_10" in first_line_upper:
        return "ANALYZE_10", reason or "Potential ambiguity or uncertainty signals detected."

    # Robust fallback: look for keyword in raw output
    if "FAST_PATH" in raw_output:
        return "FAST_PATH", "Fast path selected based on response criteria."

    return "ANALYZE_10", f"Ambiguity or unverified assessment format: {lines[0]}"


def save_experiment_results(data: Dict[str, Any]) -> Path:
    """Save experiment metadata and results under a unique timestamped filename."""
    out_dir = DEFAULT_PATH_CONFIG.raw_generations_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Microsecond precision suffix to ensure strict uniqueness
    ms_suffix = int((time.time() % 1) * 1000)
    out_file = out_dir / f"adaptive_{timestamp}_{ms_suffix:03d}.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_file


def run_pipeline(
    question: str,
    classifier: UncertaintyClassifier,
    interactive: bool = True,
) -> Dict[str, Any]:
    """Execute the end-to-end adaptive uncertainty classification pipeline."""
    start_total_time = time.time()
    model = DEFAULT_MODEL_CONFIG.model_name
    temp = DEFAULT_MODEL_CONFIG.temperature
    top_p = DEFAULT_MODEL_CONFIG.top_p

    # Step 1: Verify Ollama and runtime context window
    try:
        runtime_ctx = verify_runtime_context(model=model)
    except (OllamaConnectionError, OllamaModelNotFoundError) as e:
        print(f"\n[Error] {e}")
        return {"error": str(e)}

    # Step 2: Check for simple conversational inputs
    if is_conversational_input(question):
        t0 = time.time()
        initial_answer = generate_response(
            question=question,
            model=model,
            temperature=temp,
            top_p=top_p,
            num_ctx=runtime_ctx,
        )
        gen_latency = time.time() - t0
        total_latency = time.time() - start_total_time

        print(f"\nQuestion: {question}\n")
        print(f"Answer: {initial_answer}\n")
        print("Path: FAST (Conversational)\n")
        print("Generations: 1\n")
        print("Classification: N/A — Ordinary conversation\n")
        print("Confidence: Uncalibrated (Conversational input)\n")

        experiment_data = {
            "question": question,
            "model": model,
            "parameters": {
                "temperature": temp,
                "top_p": top_p,
                "effective_runtime_context": runtime_ctx,
            },
            "path": "FAST",
            "is_conversational": True,
            "decision": "FAST_PATH",
            "decision_reason": "Ordinary greeting or small talk.",
            "generations_count": 1,
            "responses": [initial_answer],
            "final_answer": initial_answer,
            "classification": "N/A",
            "confidence": "uncalibrated",
            "latencies": {
                "initial_answer_latency": gen_latency,
                "total_latency": total_latency,
            },
        }
        saved_path = save_experiment_results(experiment_data)
        print(f"[Results saved to {saved_path.name}]")
        return experiment_data

    # Step 3: Fast path candidate - Generate initial response
    if interactive:
        print("\nGenerating initial answer...")
    t0 = time.time()
    initial_answer = generate_response(
        question=question,
        model=model,
        temperature=temp,
        top_p=top_p,
        num_ctx=runtime_ctx,
    )
    initial_latency = time.time() - t0

    # Step 4: Evaluate uncertainty
    if interactive:
        print("Assessing uncertainty...")
    t_assess = time.time()
    decision, reason = assess_uncertainty(
        question=question,
        initial_answer=initial_answer,
        model=model,
        context_window=runtime_ctx,
    )
    assessment_latency = time.time() - t_assess

    # FAST PATH BRANCH
    if decision == "FAST_PATH":
        total_latency = time.time() - start_total_time
        concise_ans = extract_concise_answer(initial_answer)

        # Check classifier prediction if trained
        clf_result = classifier.predict({}) if classifier.is_trained else {
            "prediction": "CLASSIFIER NOT TRAINED",
            "confidence": "Uncalibrated (requires multi-response feature extraction and labeled training)",
        }

        classification_str = (
            "KNOWN"
            if classifier.is_trained and clf_result.get("prediction") == "KNOWN"
            else "CLASSIFIER NOT TRAINED"
        )

        print(f"\nQuestion: {question}\n")
        print(f"Answer: {concise_ans}\n")
        print("Path: FAST\n")
        print("Generations: 1\n")
        print(f"Classification: {classification_str}")
        print(f"Confidence: {clf_result.get('confidence', 'Uncalibrated')}\n")

        experiment_data = {
            "question": question,
            "model": model,
            "parameters": {
                "temperature": temp,
                "top_p": top_p,
                "effective_runtime_context": runtime_ctx,
            },
            "path": "FAST",
            "is_conversational": False,
            "decision": decision,
            "decision_reason": reason,
            "generations_count": 1,
            "responses": [initial_answer],
            "final_answer": concise_ans,
            "classification": classification_str,
            "confidence": clf_result.get("confidence", "uncalibrated"),
            "latencies": {
                "initial_answer_latency": initial_latency,
                "uncertainty_assessment_latency": assessment_latency,
                "total_latency": total_latency,
            },
        }
        saved_path = save_experiment_results(experiment_data)
        print(f"[Results saved to {saved_path.name}]")
        return experiment_data

    # MULTI-RESPONSE BRANCH
    print(f"\nQuestion: {question}\n")
    print(f"Initial answer: {initial_answer}\n")
    print(f"Reason for deeper analysis: {reason}\n")
    print("Generating additional responses: 9/9")

    t_add = time.time()

    def progress(current: int, total: int):
        print(f"  • Generating response {current}/{total}...", end="\r", flush=True)

    additional = generate_multiple_responses(
        question=question,
        model=model,
        num_generations=9,
        temperature=temp,
        top_p=top_p,
        num_ctx=runtime_ctx,
        progress_callback=progress,
    )
    print()  # newline after progress
    additional_latency = time.time() - t_add

    responses = [initial_answer] + [item.response for item in additional]

    print("\n" + "=" * 60)
    print("ALL 10 RESPONSES")
    print("=" * 60)
    for idx, r in enumerate(responses, start=1):
        print(f"\nResponse {idx}:\n{r.strip()}")
    print("\n" + "=" * 60)

    # Step 5: Run NLP semantic analysis
    if interactive:
        print("\nRunning NLP semantic analysis and clustering...")
    t_feat = time.time()
    analysis = analyze_semantic_responses(
        responses=responses,
        question=question,
        similarity_threshold=DEFAULT_NLP_CONFIG.similarity_threshold,
    )
    feature_latency = time.time() - t_feat

    # Step 6: Select final candidate answer using semantic clusters
    answer_result = select_semantic_final_answer(
        clusters=analysis.clusters,
        majority_agreement=analysis.majority_cluster_agreement,
        second_agreement=analysis.second_cluster_agreement,
        question=question,
    )
    final_answer = answer_result["final_answer"]

    # Step 7: Classify using ML classifier if trained
    clf_result = classifier.predict(analysis.features)

    if classifier.is_trained:
        classification_str = clf_result["prediction"]
        confidence_str = clf_result["confidence"]
    else:
        classification_str = "CLASSIFIER NOT TRAINED"
        confidence_str = "Uncalibrated (classifier infrastructure ready; awaits validated labeled data)"

    total_latency = time.time() - start_total_time

    # Output according to required specification
    print(f"\nSemantic answer groups: {analysis.num_clusters}")
    print(f"Majority agreement: {analysis.majority_cluster_agreement:.1%}")
    print(
        f"Semantic entropy: {analysis.raw_semantic_entropy:.3f} "
        f"(normalized: {analysis.normalized_semantic_entropy:.3f})\n"
    )
    print(f"Final answer: {final_answer}\n")
    print(f"Classification: {classification_str}")
    print(f"Confidence: {confidence_str}\n")

    # Record and save
    experiment_data = {
        "question": question,
        "model": model,
        "parameters": {
            "temperature": temp,
            "top_p": top_p,
            "effective_runtime_context": runtime_ctx,
            "similarity_threshold": DEFAULT_NLP_CONFIG.similarity_threshold,
        },
        "path": "MULTI_RESPONSE",
        "is_conversational": False,
        "decision": decision,
        "decision_reason": reason,
        "generations_count": 10,
        "responses": responses,
        "semantic_groups_count": analysis.num_clusters,
        "majority_agreement": analysis.majority_cluster_agreement,
        "second_agreement": analysis.second_cluster_agreement,
        "agreement_margin": analysis.agreement_margin,
        "semantic_entropy": analysis.raw_semantic_entropy,
        "normalized_semantic_entropy": analysis.normalized_semantic_entropy,
        "extracted_features": analysis.features,
        "final_answer": final_answer,
        "classification": classification_str,
        "confidence": confidence_str,
        "latencies": {
            "initial_answer_latency": initial_latency,
            "uncertainty_assessment_latency": assessment_latency,
            "additional_generation_latency": additional_latency,
            "feature_extraction_latency": feature_latency,
            "total_latency": total_latency,
        },
    }

    saved_path = save_experiment_results(experiment_data)
    print(f"[Results saved to {saved_path.name}]")
    return experiment_data


def main():
    """Terminal entry point."""
    print("=" * 60)
    print("ADAPTIVE LLM UNCERTAINTY CLASSIFICATION")
    print("Model: llama3.1:8b-research (Q4_K_M)")
    print("=" * 60)

    # Check Ollama connection on startup
    if not check_ollama_server():
        print(
            "\n[Error] Cannot connect to Ollama server at http://localhost:11434.\n"
            "Please ensure Ollama is running (`ollama serve`).\n"
        )
        return

    # Load classifier (will cleanly show 'CLASSIFIER NOT TRAINED' if weights not present)
    classifier = UncertaintyClassifier.load()

    while True:
        try:
            question = input("\nEnter your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Exiting.")
            break

        try:
            run_pipeline(question, classifier=classifier, interactive=True)
        except Exception as e:
            print(f"\n[Execution Error] {e}")


if __name__ == "__main__":
    main()
