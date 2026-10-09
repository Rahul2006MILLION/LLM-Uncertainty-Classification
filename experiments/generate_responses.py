"""Generate multiple independent responses for a question sequentially."""

from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_MODEL_CONFIG, DEFAULT_PATH_CONFIG
from src.generation.ollama_generator import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    check_ollama_server,
    generate_multiple_responses,
    verify_runtime_context,
)


def main():
    print("=" * 60)
    print("LLM RESPONSE GENERATION")
    print("=" * 60)

    if not check_ollama_server():
        print(
            "\n[Error] Cannot connect to Ollama server at http://localhost:11434.\n"
            "Please ensure Ollama is running (`ollama serve`).\n"
        )
        return

    question = input("\nEnter your question: ").strip()
    if not question:
        print("Error: Question cannot be empty.")
        return

    model = DEFAULT_MODEL_CONFIG.model_name
    num_gens = DEFAULT_MODEL_CONFIG.num_generations
    temp = DEFAULT_MODEL_CONFIG.temperature
    top_p = DEFAULT_MODEL_CONFIG.top_p

    try:
        runtime_ctx = verify_runtime_context(model=model)
    except (OllamaConnectionError, OllamaModelNotFoundError) as e:
        print(f"\n[Error] {e}")
        return

    print(f"\nModel: {model}")
    print(f"Generations: {num_gens}")
    print(f"Temperature: {temp}")
    print(f"Top-p: {top_p}")
    print(f"Effective Context Window: {runtime_ctx} tokens")

    print(f"\nGenerating {num_gens} responses sequentially...\n")

    def progress(curr, total):
        print(f"  • Progress: {curr}/{total} generated...", end="\r", flush=True)

    results = generate_multiple_responses(
        question=question,
        model=model,
        num_generations=num_gens,
        temperature=temp,
        top_p=top_p,
        num_ctx=runtime_ctx,
        progress_callback=progress,
    )
    print()

    print("\n" + "=" * 60)
    print(f"THE {num_gens} GENERATED RESPONSES")
    print("=" * 60)

    for result in results:
        print(f"\nResponse {result.generation_index}:")
        print(result.response)

    output_dir = DEFAULT_PATH_CONFIG.raw_generations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generations_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            [result.__dict__ for result in results],
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(f"Raw responses saved to:\n{output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
