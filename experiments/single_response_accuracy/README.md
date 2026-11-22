# Single-Response Accuracy Improvement Benchmark

This experiment investigates whether the classification accuracy and Macro-F1 of the **TRUE_SINGLE_RESPONSE** baseline can be improved through superior text representations, semantic embeddings, enriched single-response linguistic features, and advanced classifiers.

## Architecture

1. **Track S1 — Existing Baseline (Control):** Canonical 6 linguistic features (`response_len`, `distinct_1gram_ratio`, `distinct_2gram_ratio`, `question_overlap`, `hedging_score`, `negation_count`).
2. **Track S2 — TF-IDF Text Classification:** Single response text $\to$ `TfidfVectorizer(ngram_range=(1,3), max_features=10000, min_df=2, sublinear_tf=True)`.
3. **Track S3 — Semantic Embedding Classification:** `all-MiniLM-L6-v2` 384-dimensional dense sentence embeddings.
4. **Track S4 — Extended Linguistic Features:** 17 genuine single-response features including epistemic markers, certainty density, conditionals, alternatives, abstention indicators, and question-response similarity.
5. **Track S4 — Hybrid Single-Response Fusion:** Semantic embeddings fused with canonical and extended linguistic features.
6. **Classifiers:** Logistic Regression, Linear Support Vector Classifier (LinearSVC), Random Forest, and XGBoost.

## Execution

```bash
python3 experiments/single_response_accuracy/run_experiment.py
```

## Directory Contents

- `feature_extractor.py`: Single-response feature extraction and embedding interface.
- `tfidf_models.py`: Scikit-learn TF-IDF pipeline configurations.
- `embedding_models.py`: Semantic embedding classification pipelines.
- `hybrid_models.py`: Feature fusion architectures.
- `statistical_analysis.py`: Paired $t$-test, Nadeau-Bengio correction, Wilcoxon, McNemar, and Bootstrap $B=2000$.
- `error_analysis.py`: Per-class, category, shortcut, and difficult example analyzers.
- `run_experiment.py`: Master runner executing 5-fold StratifiedGroupKFold on $N=60$.
- `results/`: CSV outputs and confusion matrices.
- `reports/`: `SINGLE_RESPONSE_ACCURACY_EXPERIMENT_REPORT.md`.
