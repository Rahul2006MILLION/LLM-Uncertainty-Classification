# Legacy Prototype Benchmark Assets

This directory preserves early prototype reference benchmark artifacts from the initial October 2026 proof-of-concept phase for provenance and historical reference.

## Archived Artifacts

1. **`benchmark_schema.json`**:
   - The initial prototype JSON schema.
   - Used heuristic question-syntax flags (`question_has_disjunction`, `question_has_which`, `question_is_short`) and non-canonical feature keys (`num_clusters`, `raw_semantic_entropy`).
   - **Status**: Obsolete / Archived. The canonical, leak-free benchmark schemas are defined in `data/schema.py` (`UncertaintyFeatures`, `ExperimentRecord`) and `data/benchmarks/gold_benchmark_schema_v1.0.json`.

2. **`reference_benchmark.json`**:
   - The initial 6-question toy prototype benchmark ("What is 2 + 2?", "Who was the 16th president of the United States?", etc.).
   - **Status**: Obsolete / Archived. All reference concepts are fully incorporated and generalized within the active benchmarks:
     - `data/benchmarks/uncertainty_benchmark_v2.jsonl` (60 questions)
     - `data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl` (300 questions)
     - `data/datasets/development_questions.jsonl` (1,050 questions)
     - `data/batch_1_questions.jsonl` (50 questions)

> [!NOTE]
> These files are maintained strictly for historical traceability and must NOT be used for active model training, evaluation, or benchmarking.
