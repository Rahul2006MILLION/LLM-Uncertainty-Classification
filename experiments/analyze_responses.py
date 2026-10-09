"""Analyze saved LLM generations using the NLP semantic analysis pipeline.

Replaces naive exact-text voting with sentence embeddings, semantic clustering,
empirical semantic entropy, and medoid-based candidate selection.

Allows targeting a specific experiment file or automatically analyzing the most
recently executed experiment with explicit verification of the question.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_NLP_CONFIG, DEFAULT_PATH_CONFIG
from src.features.semantic_analysis import analyze_semantic_responses
from src.features.answer_extractor import select_semantic_final_answer
from src.classification.classifier import UncertaintyClassifier


def load_experiment_data(file_path: Path):
    """Load and parse responses and metadata from an experiment JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both list of objects (generations_*.json) and dict format (adaptive_*.json)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"File '{file_path}' contains empty list.")
        question = data[0].get("question", "Unknown")
        responses = [item.get("response", "") for item in data]
        model = data[0].get("model", "Unknown")
    elif isinstance(data, dict):
        question = data.get("question", "Unknown")
        responses = data.get("responses", [])
        model = data.get("model", "Unknown")
    else:
        raise ValueError(f"Unexpected JSON format in {file_path}")

    return question, responses, model


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM uncertainty from saved responses.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to specific JSON file in results/raw_generations/",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_NLP_CONFIG.similarity_threshold,
        help="Cosine similarity threshold for semantic clustering (default: 0.80)",
    )
    args = parser.parse_args()

    results_dir = DEFAULT_PATH_CONFIG.raw_generations_dir

    if args.file:
        target_path = Path(args.file)
    else:
        all_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not all_files:
            print("No experiment files found in results/raw_generations/.")
            return
        target_path = all_files[-1]

    if not target_path.exists():
        print(f"Error: File '{target_path}' does not exist.")
        return

    print("=" * 60)
    print("LLM SEMANTIC UNCERTAINTY ANALYSIS")
    print("=" * 60)
    print(f"Target File: {target_path.name}")

    question, responses, model = load_experiment_data(target_path)

    print(f"Question:    {question}")
    print(f"Model:       {model}")
    print(f"Responses:   {len(responses)}")
    print(f"Threshold:   {args.threshold}")
    print("-" * 60)

    if not responses:
        print("No responses found in this file.")
        return

    # Run NLP semantic analysis
    analysis = analyze_semantic_responses(
        responses=responses,
        question=question,
        similarity_threshold=args.threshold,
    )

    answer_result = select_semantic_final_answer(
        clusters=analysis.clusters,
        majority_agreement=analysis.majority_cluster_agreement,
        second_agreement=analysis.second_cluster_agreement,
        question=question,
    )

    print(f"\nSemantic Answer Groups (Clusters): {analysis.num_clusters}")
    print(f"Majority Cluster Agreement:       {analysis.majority_cluster_agreement:.1%}")
    print(f"Second Cluster Agreement:         {analysis.second_cluster_agreement:.1%}")
    print(f"Agreement Margin (FSD):           {analysis.agreement_margin:.3f}")
    print(f"Raw Semantic Entropy:             {analysis.raw_semantic_entropy:.3f}")
    print(f"Normalized Semantic Entropy:      {analysis.normalized_semantic_entropy:.3f}")
    print(f"Mean Pairwise Similarity:         {analysis.mean_pairwise_similarity:.3f}")
    print(f"Min Pairwise Similarity:          {analysis.min_pairwise_similarity:.3f}")

    print("\n" + "=" * 60)
    print("CLUSTER BREAKDOWN")
    print("=" * 60)
    for c in analysis.clusters:
        print(f"\nGroup {c.cluster_id + 1} ({c.size}/{analysis.num_responses} responses, {c.proportion:.1%}):")
        print(f"Representative: {c.representative_text.strip()}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER CANDIDATE")
    print("=" * 60)
    print(answer_result["final_answer"])
    print(f"\nSelection Note: {answer_result['explanation']}")

    classifier = UncertaintyClassifier.load()
    clf_res = classifier.predict(analysis.features)
    print(f"\nClassifier Prediction: {clf_res['prediction']}")
    print(f"Confidence:            {clf_res['confidence']}")


if __name__ == "__main__":
    main()
