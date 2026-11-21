# LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-SA 3.0](https://img.shields.io/badge/License-CC%20BY--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/3.0/)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)]()

A scientifically grounded, reproducible research prototype investigating whether semantic consistency, clustering dynamics, and information-theoretic features extracted from stochastic multi-response local LLM generations can distinguish question ambiguity, answerability, and epistemic uncertainty.

---

## 1. Research Question & Uncertainty Taxonomy

### Core Research Question
> *Can semantic consistency and feature representations extracted from multiple stochastic responses of a local LLM reliably differentiate between answerable knowledge, question ambiguity, and fundamental unanswerability?*

### Three-Class Uncertainty Taxonomy
We formulate uncertainty categorization into three mutually exclusive categories:

| Class | Definition | Operational Criteria | Representative Example |
| :--- | :--- | :--- | :--- |
| **`KNOWN`** | Sufficiently specified question with adequate factual information available to provide a definitive answer. | Single factual answer exists; model exhibits high consistency and concentrated probability. | *"What is 2 + 2?"*, *"What is Data Science?"* |
| **`AMBIGUOUS`** | Question with multiple materially different, equally plausible interpretations or referents. | Query lacks distinguishing criteria (e.g., polysemy, entity confusion, differing metrics); multi-response sampling reveals competing semantic clusters. | *"What is the capital of Georgia?"* (Country vs. US State), *"Who is Jordan?"* |
| **`UNKNOWN`** | Sufficiently clear question for which the model or available empirical evidence lacks adequate information to answer reliably. | Inaccessible private mental states, undetermined future contingencies, or immeasurable states; multi-response sampling reveals dispersion, refusal, or high entropy. | *"What number am I thinking of right now?"*, *"What will Apple's stock price be on Dec 31, 2035?"* |

### Scientific Principles & Safeguards
1. **Ambiguity $\neq$ Lack of Knowledge**: An ambiguous question is not unanswerable; rather, it has multiple valid answers depending on context. Forcing an arbitrary single answer is a failure of calibration.
2. **Agreement $\neq$ Truth**: High consensus across stochastic generations does **not** prove empirical correctness. A model may repeatedly generate a false hallucination with 100% agreement (e.g. on trick or false-premise questions).
3. **Strict Prevention of Label Leakage**: Features are derived exclusively from question text, generated responses, and semantic embeddings. No target labels, expected answers, or gold annotations enter the feature extraction pipeline.
4. **Honest Classifier Status**: In the absence of a large-scale, independently audited benchmark, the system reports `CLASSIFIER NOT TRAINED` rather than fabricating synthetic generalization metrics.

---

## 2. System Architecture

```
                                    +-----------------------+
                                    |     User Question     |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    | Ollama Initial Sample |
                                    |     (Greedy / T=0)    |
                                    +-----------+-----------+
                                                |
                                                v
                                    +-----------------------+
                                    |    Adaptive Router    |
                                    +-----+-----------+-----+
                                          |           |
                     [Simple / Confident] |           | [Ambiguous / Uncertain]
                                          v           v
           +--------------------------------+       +-----------------------------------+
           |           FAST PATH            |       |        MULTI-RESPONSE PATH        |
           |  - Immediate direct return     |       |  - Generate 9 additional samples  |
           |  - Latency: < 1.0s             |       |  - Total: exactly 10 responses    |
           |  - Entropy = 0.0               |       +-----------------+-----------------+
           +--------------------------------+                         |
                                                                      v
                                                    +-----------------------------------+
                                                    |    NLP & Embedding Engine         |
                                                    |  - Boilerplate strip (negation safe)|
                                                    |  - L2-Normalized vectors (all-minilm)|
                                                    +-----------------+-----------------+
                                                                      |
                                                                      v
                                                    +-----------------------------------+
                                                    |    Agglomerative Clustering       |
                                                    |  - Cosine distance threshold τ    |
                                                    |  - Medoid representative selection|
                                                    +-----------------+-----------------+
                                                                      |
                                                                      v
                                                    +-----------------------------------+
                                                    |    Feature & Entropy Extraction   |
                                                    |  - Semantic Entropy H(X) & H_norm |
                                                    |  - Dominant & Second Agreement    |
                                                    |  - First-Second Difference (FSD)  |
                                                    |  - Pairwise similarity matrix     |
                                                    |  - Hedging & negation markers     |
                                                    +-----------------+-----------------+
                                                                      |
                                                                      v
                                                    +-----------------------------------+
                                                    |    Adaptive Answer Synthesis      |
                                                    |  - If p2 >= 0.25: explain both    |
                                                    |  - Otherwise: dominant medoid     |
                                                    +-----------------+-----------------+
                                                                      |
                                                                      v
                                                    +-----------------------------------+
                                                    |  Classification / JSON Export     |
                                                    |  - XGBoost / HistGB Inference     |
                                                    |  - Timestamped experiment log     |
                                                    +-----------------------------------+
```

---

## 3. Mathematical Foundations

### 1. Semantic Entropy
Given $N$ stochastic responses clustered into $K$ semantic equivalence classes $C_1, C_2, \dots, C_K$ with response counts $N_1 \ge N_2 \ge \dots \ge N_K > 0$ such that $\sum_{k=1}^K N_k = N$, the empirical probability of cluster $k$ is:
$$p_k = \frac{N_k}{N}$$

The semantic entropy $H(X)$ is computed as:
$$H(X) = - \sum_{k=1}^K p_k \ln(p_k)$$

### 2. Normalized Semantic Entropy
To enable meaningful comparisons across different sample counts, entropy is normalized by the maximum theoretical entropy $\ln(N)$:
$$H_{norm}(X) = \begin{cases} \frac{H(X)}{\ln(N)} & \text{if } N > 1 \\ 0.0 & \text{if } N = 1 \end{cases}$$
Where $H_{norm}(X) \in [0.0, 1.0]$.
- Complete consensus ($K=1$): $H = 0.0$, $H_{norm} = 0.0$, $p_1 = 1.0$.
- Complete dispersion ($K=N$): $H = \ln(N)$, $H_{norm} = 1.0$, $p_1 = 1/N$.

### 3. Agreement Dynamics & Agreement Margin (FSD)
- **Dominant Agreement ($p_1$)**: $N_1 / N$. High values indicate strong model consensus.
- **Second Cluster Agreement ($p_2$)**: $N_2 / N$ (or $0.0$ if $K=1$).
- **Agreement Margin (First-Second Difference, FSD)**:
$$\Delta p = p_1 - p_2$$
A low margin with $p_2 \ge 0.25$ indicates competing plausible interpretations (hallmark of ambiguity).

### 4. Mean Pairwise Semantic Similarity
Given L2-normalized embedding vectors $\mathbf{v}_1, \dots, \mathbf{v}_N \in \mathbb{R}^D$:
$$\bar{S} = \frac{2}{N(N-1)} \sum_{i < j} \mathbf{v}_i^\top \mathbf{v}_j$$

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- Required local Ollama models:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull all-minilm
  ```

### Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Execution Commands

### 1. Interactive CLI (`experiments/ask.py`)
Run an interactive query or specify a question via command line:
```bash
# Simple fast-path query
python experiments/ask.py -q "What is 2 + 2?"

# Definitional fast-path query
python experiments/ask.py -q "What is Data Science?"

# Ambiguous query triggering 10-response multi-path analysis
python experiments/ask.py -q "What is the capital of Georgia?"

# Ambiguous entity query
python experiments/ask.py -q "Who is Jordan?"

# Criterion ambiguity query
python experiments/ask.py -q "What is the largest city in the world?"

# Unanswerable query
python experiments/ask.py -q "What number am I thinking of right now?"

# Force multi-response analysis on any query
python experiments/ask.py -q "What is 2 + 2?" --force-multi
```

### 2. Automated Benchmark Runner (`experiments/run_benchmarks.py`)
Executes predefined test suites and evaluates path routing and consistency:
```bash
python experiments/run_benchmarks.py
```

### 3. Batch Dataset Pipeline
```bash
# Step 1: Generate stochastic responses for benchmark questions
python experiments/generate_responses.py -i data/benchmarks/pilot_uncertainty_benchmark.jsonl -o data/raw/generated_responses.jsonl -n 10

# Step 2: Extract semantic clustering and entropy features
python experiments/analyze_responses.py -i data/raw/generated_responses.jsonl -o data/features.jsonl

# Step 3: Train and evaluate classifier with baselines and ablations
python experiments/train_classifier.py -f data/features.jsonl --model-type xgboost --save-artifact
```

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 6. Dataset Schema & Provenance

Benchmark datasets follow the Pydantic schema defined in [`data/schema.py`](data/schema.py):

```json
{
  "id": "ambig_12345",
  "question": "When did the simpsons first air on television?",
  "label": "AMBIGUOUS",
  "domain": "entertainment",
  "source": "ambignq_light",
  "license": "CC BY-SA 3.0",
  "ground_truth_answers": ["April 19, 1987", "December 17, 1989"],
  "interpretations": [
    "When did the Simpsons air as animated shorts on The Tracey Ullman Show?",
    "When did the Simpsons premiere as a standalone half-hour series?"
  ],
  "annotation_type": "gold_benchmark",
  "annotation_rationale": "AmbigNQ gold benchmark annotation with multiple distinct QA pairs.",
  "metadata": {}
}
```

### Provenance Sources:
- **`ambignq_light`**: Derived from the Min et al. (2020) AmbigQA benchmark (CC BY-SA 3.0). Questions with `multipleQAs` provide gold `AMBIGUOUS` samples; questions with `singleAnswer` provide gold `KNOWN` samples.
- **`curated_unanswerable`**: Grounded non-epistemic questions spanning private mental states, indeterminate future contingencies, and immeasurable physical parameters (CC0).

---

## 7. Experimental Results & Baselines

### Automated Live-Model Benchmark Run (`qwen2.5:3b` + `all-minilm`)

| Question | Category | Expected Path | Actual Path | Clusters ($K$) | Entropy ($H$) | Latency | Result |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **What is Data Science?** | Concept Definition | `FAST_PATH` | `FAST_PATH` | 1 | 0.000 | 0.74s | **PASS** |
| **What is 2 + 2?** | Arithmetic | `FAST_PATH` | `FAST_PATH` | 1 | 0.000 | 0.26s | **PASS** |
| **What is the capital of Georgia?** | Geographic Polysemy | `MULTI_RESPONSE` | `MULTI_RESPONSE` | 2 | 0.325 | 1.23s | **PASS** |
| **Who is Jordan?** | Ambiguous Entity | `MULTI_RESPONSE` | `MULTI_RESPONSE` | 2 | 0.325 | 7.85s | **PASS** |
| **What is the largest city in the world?** | Criterion Ambiguity | `MULTI_RESPONSE` | `MULTI_RESPONSE` | 3 | 1.090 | 9.64s | **PASS** |
| **What number am I thinking of right now?** | Insufficient Info | `MULTI_RESPONSE` | `MULTI_RESPONSE` | 2 | 0.610 | 7.20s | **PASS** |
| **FIFA Ice Hockey World Cup?** | False Premise Factual | `MULTI_RESPONSE` | `MULTI_RESPONSE` | 1 | 0.000 | 4.43s | **PASS** |

### Key Research Insights:
1. **Adaptive Routing Efficiency**: The fast path delivers answers in **< 0.8s** with zero unnecessary LLM invocations, saving up to 90% compute on unambiguous queries.
2. **Ambiguity Disambiguation**: On polysemous questions (e.g. *"What is the largest city in the world?"*), the pipeline isolates competing semantic clusters and synthesizes a dual-interpretation explanation rather than arbitrarily picking one.
3. **The False Consensus Phenomenon**: On the false-premise question (*"Which organization organizes the FIFA Ice Hockey World Cup?"*), the model displayed 100% agreement and zero entropy on a premise that is strictly incorrect. This confirms our core hypothesis: **semantic consensus reflects internal model consistency, not factual truth**.

---

## 8. Limitations & Future Directions

1. **Model Parameter Scale**: Compact local models ($\le 3$B parameters) exhibit strong mode collapse on certain ambiguous queries, requiring prompt diversity or temperature adjustment to surface minority interpretations.
2. **Clustering Sensitivity**: Hard similarity thresholding ($\tau = 0.80$) can occasionally split semantically equivalent paraphrases if the embedding model is sensitive to syntactic variance.
3. **Sample Size Constraints**: Supervised classifier training requires large-scale multi-response generation corpora to ensure statistical generalization across domains. Pilot datasets are suitable for software pipeline validation, not generalized deployment.
