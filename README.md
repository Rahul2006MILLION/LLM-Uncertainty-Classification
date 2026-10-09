# Adaptive LLM Uncertainty Classification Research Project

An adaptive question-answering and uncertainty estimation framework that combines local large language models (Ollama / Llama 3.1 8B), NLP semantic clustering, empirical semantic entropy, and supervised machine learning to detect when an LLM is **KNOWN**, **AMBIGUOUS**, or **UNKNOWN**.

---

## 1. Research Question & Objective

Large Language Models (LLMs) frequently generate answers with high linguistic fluency even when the prompt is inherently ambiguous or when the model lacks factual grounding. Traditional methods evaluate uncertainty using:
1. **Self-reported confidence** (e.g., asking the model "How confident are you?"), which is notoriously uncalibrated and sycophantic.
2. **Exact-text voting / token probabilities**, which fail because the same semantic answer can be phrased in dozens of syntactically distinct ways (e.g., "Atlanta", "The capital of Georgia is Atlanta", and "Atlanta, GA").

### Research Questions
- **RQ1**: Can semantic clustering over multiple sampled responses distinguish true epistemic uncertainty from surface-level lexical variation?
- **RQ2**: Does an adaptive execution pipeline (fast-path for straightforward definitions/facts vs. multi-response path for flagged uncertainty) preserve inference efficiency while providing deep transparency when needed?
- **RQ3**: How can we objectively classify prompt response states into three distinct epistemological categories without label leakage?
  - `KNOWN`: The question has a clear interpretation, and the model's answer is supported by reliable, consistent evidence.
  - `AMBIGUOUS`: The question has multiple materially different plausible interpretations (e.g., entity polysemy, regional ambiguity).
  - `UNKNOWN`: The question is clear, but the system lacks reliable information to answer it (unanswerable, fictional, or future facts).

> **Crucial Axiom**: A wrong answer alone must **NOT** automatically be classified as UNKNOWN. Factual inaccuracy from hallucination differs from epistemic void.

---

## 2. Architecture & Pipeline

```mermaid
flowchart TD
    User([User Prompt]) --> GreetCheck{Is Greeting / Small-Talk?}
    GreetCheck -- Yes --> ConvFast[Return Direct Conversational Response\nGenerations: 1 | Path: FAST]
    GreetCheck -- No --> InitialGen[Generate Initial Answer\nTemp: 0.8, Top-p: 0.9, Generations: 1]
    
    InitialGen --> Assess[Strict Uncertainty Assessment\nZero-Shot Evaluation @ Temp: 0.0]
    Assess -- Clear & Unambiguous --> FastPath[Fast Path: Return Concise Candidate\nGenerations: 1 | Path: FAST]
    
    Assess -- Material Uncertainty Flagged --> MultiPath[Multi-Response Path: Sequential Sampling\nGenerate 9 Additional Responses: 10 Total]
    MultiPath --> Embed[NLP Sentence Embeddings\n384-d Dense Vectors via all-MiniLM-L6-v2]
    Embed --> CosineSim[Pairwise Cosine Similarity Matrix]
    CosineSim --> Cluster[Agglomerative Semantic Clustering\nThreshold: 0.80]
    Cluster --> Entropy[Empirical Semantic Entropy\nH_raw & H_norm in [0, 1]]
    Cluster --> AnswerSelect[Medoid Representative Selection\nor Ambiguity Disambiguation Output]
    Entropy --> MLClassifier[Supervised Classifier: XGBoost / Random Forest]
    AnswerSelect --> Display([Clean Terminal Research Output])
    MLClassifier --> Display
```

### Fast Path vs. Multi-Response Path
1. **Conversational Filter**: Inputs like `Hello`, `Hi`, `How are you?` are immediately handled as natural conversation without triggering expensive multi-sample uncertainty analysis.
2. **Fast Path**: For well-defined questions (e.g., `What is Data Science?`, `What is 2 + 2?`), the model generates a single answer and assesses whether ambiguity or lack of knowledge is present. If clear, it returns the direct answer immediately, saving 90% of LLM inference compute.
3. **Multi-Response Path**: When uncertainty or ambiguity is flagged (e.g., `What is the capital of Georgia?`), the pipeline generates 9 additional responses (10 total) sequentially, prints all 10 responses for research transparency, and executes the NLP semantic pipeline.

---

## 3. Why NLP Is Essential

Relying on exact string matching or character heuristics in LLM evaluation causes severe failure modes:
- **Paraphrase Penalization**: "Paris", "Paris, France", and "The capital city is Paris" have 0% exact-string equality, wrongly inflating uncertainty metrics under naive voting.
- **Syntactic Conflation**: Contradictory statements ("The capital is Atlanta" vs "The capital is Tbilisi") share high word overlap ("The capital is...") but express diametrically opposite claims.

### NLP Components Implemented

#### A. Text Preprocessing (`src/preprocessing/text_cleaner.py`)
- Normalizes superfluous formatting, code fences, and whitespace.
- Strictly preserves critical semantic markers: negations (`not`, `never`), proper nouns, dates, numbers, and contradiction markers.

#### B. Sentence Embeddings (`src/features/embeddings.py`)
- Encodes responses into 384-dimensional dense vectors using the dedicated `all-MiniLM-L6-v2` embedding model.
- L2-normalized so that inner dot products directly compute cosine similarity:
  $$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

#### C. Semantic Clustering (`src/features/semantic_analysis.py`)
- Converts pairwise similarities into a distance matrix $D = 1 - S$.
- Employs **Agglomerative Clustering with average linkage** at a documented, configurable threshold $\theta = 0.80$ (distance threshold $d = 0.20$).
- Avoids the single-linkage chaining problem while grouping genuine semantic paraphrases.

#### D. Empirical Semantic Entropy
Let $N$ be the total number of sampled responses (e.g., $N=10$) grouped into $K$ semantic clusters $\{C_1, C_2, \dots, C_K\}$.
The empirical probability of cluster $k$ is:
$$p_k = \frac{|C_k|}{N}$$

The **Raw Semantic Entropy** is:
$$H_{\text{semantic}} = -\sum_{k=1}^K p_k \ln(p_k)$$

The **Normalized Semantic Entropy** is bounded in $[0.0, 1.0]$:
$$H_{\text{norm}} = \begin{cases} 
\frac{H_{\text{semantic}}}{\ln(N)} & \text{if } N > 1 \\ 
0.0 & \text{if } N = 1 
\end{cases}$$

- **$H_{\text{norm}} = 0.0$**: All 10 responses belong to 1 semantic cluster (total consensus).
- **$H_{\text{norm}} = 1.0$**: 10 singleton clusters (maximal dispersion/disagreement).

---

## 4. Response Consistency vs. Factual Correctness

A fundamental tenet of uncertainty research:
> **Response consistency is NOT equivalent to factual accuracy.**

If an LLM has a deeply entrenched hallucination or systemic training bias, it may produce identical false claims across all 10 generations ($H_{\text{norm}} = 0.0$). High agreement measures **internal model consistency**, not empirical ground truth. Our system explicitly communicates this distinction and reports uncalibrated confidence unless verified against labeled external ground truth.

---

## 5. Supervised ML Classifier (`src/classification/`)

The supervised classifier predicts whether a question's response state is `KNOWN`, `AMBIGUOUS`, or `UNKNOWN`.

### Supported Algorithms
- **XGBoost** (`multi:softprob`) with automatic fallback to **Random Forest**.

### Input Features (Leak-Free)
1. `num_clusters`: Count of semantic answer clusters ($K$).
2. `majority_cluster_agreement`: Size of largest cluster / $N$.
3. `second_cluster_agreement`: Size of second-largest cluster / $N$.
4. `agreement_margin` (FSD): $p_1 - p_2$.
5. `raw_semantic_entropy`: Empirical Shannon entropy over clusters.
6. `normalized_semantic_entropy`: $H_{\text{norm}} \in [0, 1]$.
7. `mean_pairwise_similarity`: Average off-diagonal cosine similarity.
8. `min_pairwise_similarity`: Minimum off-diagonal cosine similarity.
9. `lexical_diversity`: Unique tokens / total tokens.
10. `question_word_count`: Length of input question.
11. `question_has_disjunction`: Independent indicator for questions with "or".

### Scientific Integrity & Leakage Prevention
- **No Ground-Truth Leakage**: Features are calculated strictly from response distributions and independent question syntax.
- **Split Isolation**: Questions are grouped so that identical or paraphrased questions never span across Train, Validation, and Test sets.
- **Honest State Reporting**: If no model has been trained on validated labeled data, the system strictly outputs:
  `Classification: CLASSIFIER NOT TRAINED`
  rather than fabricating pseudo-scientific classifications.

---

## 6. How to Run the Project

### Requirements
- **macOS** with Apple Silicon or Intel.
- **Python 3.12+ / 3.14+**.
- **Ollama server** running locally at `http://localhost:11434`.
- Installed Ollama models:
  ```bash
  ollama run llama3.1:8b-research
  ollama pull all-minilm
  ```

### Interactive CLI (Primary User Experience)
Run the adaptive question-answering CLI:
```bash
python3 experiments/ask.py
```

### Analyzing Saved Experiment Files
Inspect any saved generation file using the full NLP pipeline:
```bash
python3 experiments/analyze_responses.py
# Or specify a particular file:
python3 experiments/analyze_responses.py --file results/raw_generations/adaptive_20261009_205237.json
```

### Generating Raw Responses (Batch)
Generate 10 sequential responses to any question:
```bash
python3 experiments/generate_responses.py
```

### Training the Classifier (When Labeled Data Is Available)
Train and evaluate an XGBoost or Random Forest model:
```bash
python3 experiments/train_classifier.py --data data/reference_benchmark.json --model-type xgboost
```

### Running the Unit Test Suite
Execute all unit tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 7. Viva & Presentation Guide

When presenting this project during a thesis defense or viva examination, emphasize:

1. **The Fast-Path Design Decision**:
   - Sequential generation of 10 responses takes ~15–25 seconds on local hardware. Generating 10 responses for "What is 2+2?" or "Hello" wastes compute.
   - The adaptive assessor allows sub-second turnaround on straightforward prompts while reserving multi-sample analysis for genuine ambiguity.

2. **Why Medoid Representatives Outperform Text Averaging**:
   - In embedding space, the "average" vector is an abstract centroid that cannot be directly decoded into English without a generative pass.
   - Using the **medoid** (the actual generated response with highest mean similarity to all members of the dominant cluster) guarantees grammatically natural, representative text.

3. **Handling Polysemous Ambiguity**:
   - For `What is the capital of Georgia?`, the system identifies two distinct clusters: Atlanta (US State) and Tbilisi (Nation).
   - Rather than forcing an arbitrary majority vote, `select_semantic_final_answer` detects the split and provides an explicit clarification notice to the user.
