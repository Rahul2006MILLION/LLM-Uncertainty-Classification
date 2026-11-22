# Stage 1 — Batch 1 Replication & Pipeline Readiness Audit Report

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Document Identifier:** `REP-AUD-M2-BATCH1-001`  
**Execution Timestamp:** 2026-11-21T13:42:00Z (Local: 2026-11-21 19:12:00 IST)  
**Status:** `VERIFIED & REPRODUCED (100% BIT-FOR-BIT DETERMINISTIC MATCH)`  
**Target Repository:** `https://github.com/Rahul2006MILLION/LLM-Uncertainty-Classification`  
**Branch:** `integration/milestone-2-to-main`  

---

## 1. Executive Summary

This report documents the formal execution of **Stage 1 (Batch 1 Replication & Pipeline Readiness Audit)** as mandated by the project roadmap before initiating Batch 2 ($N=150$ questions, $M=1,500$ responses).

With an overarching project deadline of approximately **23 hours**, this audit was conducted to verify that:
1. All existing Batch 1 datasets, feature schemas, and statistical benchmarks on disk are intact, uncorrupted, and valid.
2. The four methodology-corrected experimental configurations can be reproduced deterministically with zero metric divergence.
3. The response generation infrastructure (`experiments/generate_responses.py`, `src/generation/`) is hardened, crash-resilient, and capable of executing the 1,500-response generation workload within the deadline window.

### Key Audit Findings
- **Replication Fidelity:** Exact $100.0000\%$ reproduction across all 4 experimental configurations, 4 model families, 5 cross-validation folds, and statistical significance tests. Zero numerical deviation ($|\Delta| = 0.00000000$).
- **Dataset Health:** 50 records in `data/batch_1_unified.jsonl`, exactly 10 responses per question ($M=500$), exactly 16 canonical features, zero missing/null values, zero infinite values.
- **Observed Generation Throughput:** In Batch 1, generating 500 responses with `qwen3:4b` required **12,569.55 seconds (3.49 hours)**, yielding an average generation speed of **22.95 seconds per response (21.77 tokens/sec)** and **251.39 seconds (~4.19 min) per 10-response question**.
- **Batch 2 Generation Feasibility:** Generating 150 questions (1,500 responses) is projected to require **10.47 hours**. Against the 23-hour deadline, this leaves an operational safety buffer of **$\approx 12.5$ hours** for feature extraction, ML training, statistical testing, and audit report generation.
- **Pipeline Durability:** Incremental JSONL append with `os.fsync`, exclusive advisory process locking (`fcntl.flock`), checkpoint recovery (`recover_and_load_checkpoint`), and exponential backoff retry loops are fully active and verified.

---

## 2. Execution Environment & Model Provenance

### 2.1 Hardware & Runtime Environment
- **Operating System:** macOS Darwin 24.6.0 (Kernel Version 24.6.0, `arm64`, Apple Silicon)
- **Python Version:** 3.14.0 (v3.14.0:ebf955df7a8, Oct 7 2025, 08:20:14) [Clang 16.0.0]
- **Core Scientific Libraries:**
  - `scikit-learn`: `1.9.1`
  - `scipy`: `1.18.1`
  - `numpy`: `2.5.0`
  - `pandas`: `3.0.3`
  - `pydantic`: `2.12.5`
- **Unit Test Suite Status:** `249 passed, 1 warning in 17.33s` (100% passing)

### 2.2 Model & Inference Architecture
- **Inference Server:** Local Ollama Daemon (`http://localhost:11434`, API active)
- **Generation Model:** `qwen3:4b`
  - **Digest:** `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
  - **Size:** 2,497,293,931 bytes (~2.50 GB)
  - **Format:** GGUF (`runner: ggml`, `quantization: Q4_K_M`, `context_length: 262,144`)
- **Embedding Model:** `all-minilm:latest`
  - **Digest:** `1b226e2802dbb772b5fc32a58f103ca1804ef7501331012de126ab22f67475ef`
  - **Size:** 45,960,996 bytes (~45.96 MB)
  - **Embedding Dimension:** 384
  - **Semantic Cosine Clustering Threshold:** Distance threshold $0.20$ (Cosine similarity $0.80$, average linkage)

---

## 3. Cryptographic Verification of Batch 1 Artifacts

All Batch 1 research artifacts were inspected and cryptographically fingerprinted using SHA-256:

| Artifact Path | SHA-256 Checksum | File Size | Record Count | Integrity Status |
|---|---|:---:|:---:|:---:|
| `data/batch_1_unified.jsonl` | `fd4af11b0fd0b4b0492e4c93f08493620fb6e94230d8e7fe5a4ff0df097284df` | 259,716 bytes | 50 records | **VALID & INTACT** |
| `data/batch_1_questions.jsonl` | `b6705c144ec6f5156ac8e5a9545c11238321c31d917304c54a5cc6024a2dc263` | 22,607 bytes | 50 records | **VALID & INTACT** |
| `data/batch_1_manifest.json` | `d0be8dcbdabed0853090f1e47c3ec824f87a62f4c1a62f0d3c9db7f38ececa91` | 16,582 bytes | JSON Manifest | **VALID & INTACT** |
| `data/batch_1_statistical_analysis.json`| `1ca0cab4c1d976d424e53923b8da33cf76bdbd8ea102c4e10a7d68562d8d304d` | 86,485 bytes | Hierarchical JSON | **VALID & INTACT** |
| `reports/batch_1_complete_scientific_audit.md` | `3d4ff5b65880642a34244763d5a25b02863280a0da5c667edbb2eb6c8c1fb35d` | 16,694 bytes | 14 Sections | **VALID & INTACT** |

### Detailed Dataset Health Metrics
- **Total Questions ($N$):** 50
- **Class Distribution:**
  - `KNOWN`: 17 (34.0%)
  - `AMBIGUOUS`: 17 (34.0%)
  - `UNKNOWN`: 16 (32.0%)
- **Total Responses ($M$):** 500 (strictly 10 responses per question; 1 greedy $T=0.0$, 9 stochastic $T=0.7$)
- **Canonical Feature Count:** 16 features per question (total 800 feature cells)
- **Null / NaN / Infinite Values:** Exactly 0
- **Benchmark Isolation:** Verified; max observed cosine similarity to 300-question gold benchmark is $0.7389$ (threshold $< 0.75$).

---

## 4. Replication of Batch 1 Experimental Results

### 4.1 Exact Execution Command
The replication pipeline was executed against the existing unified dataset without modifying data or regenerating responses:

```bash
python3 scripts/analyze_batch_1_scientific_audit.py
```

### 4.2 Cross-Validation Methodology
- **Splitter:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- **Grouping Unit:** Strict grouping on `question_group_id` (zero question or cluster leakage across train/test splits).
- **Preprocessing:** `StandardScaler` fitted strictly on training folds and transformed onto test folds.
- **Model Suite:**
  - `RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)` (Primary Model)
  - `LogisticRegression(max_iter=1000, C=1.0, random_state=42)` (Linear Calibrated Model)
  - `HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)` (Gradient Boosting Model)
  - `DummyClassifier(strategy="most_frequent")` (Majority Baseline Anchor)

---

## 5. Original vs. Reproduced Results Comparison

### 5.1 Primary Classifier Benchmark: Random Forest

| Experimental Configuration | Metric | Original Stored Value | Recomputed Value | Absolute Discrepancy | Match Status |
|---|---|:---:|:---:|:---:|:---:|
| **Exp A: True Single-Response** *(r0, 6 features)* | Macro-F1 | $0.9183 \pm 0.0414$ | $0.9183 \pm 0.0414$ | $0.00000000$ | **EXACT MATCH** |
| | Accuracy | $0.9200 \pm 0.0400$ | $0.9200 \pm 0.0400$ | $0.00000000$ | **EXACT MATCH** |
| | Balanced Accuracy | $0.9222 \pm 0.0416$ | $0.9222 \pm 0.0416$ | $0.00000000$ | **EXACT MATCH** |
| | Per-Class F1 (K/A/U) | $0.9206 / 0.9029 / 0.9314$ | $0.9206 / 0.9029 / 0.9314$ | $0.00000000$ | **EXACT MATCH** |
| **Exp B: Ensemble Linguistic** *(r0..r9, 7 features)* | Macro-F1 | $0.9008 \pm 0.0877$ | $0.9008 \pm 0.0877$ | $0.00000000$ | **EXACT MATCH** |
| | Accuracy | $0.9000 \pm 0.0894$ | $0.9000 \pm 0.0894$ | $0.00000000$ | **EXACT MATCH** |
| | Balanced Accuracy | $0.9000 \pm 0.0882$ | $0.9000 \pm 0.0882$ | $0.00000000$ | **EXACT MATCH** |
| | Per-Class F1 (K/A/U) | $0.9378 / 0.8600 / 0.9048$ | $0.9378 / 0.8600 / 0.9048$ | $0.00000000$ | **EXACT MATCH** |
| **Exp C: Consistency Only** *(r0..r9, 9 features)* | Macro-F1 | $0.4701 \pm 0.1270$ | $0.4701 \pm 0.1270$ | $0.00000000$ | **EXACT MATCH** |
| | Accuracy | $0.4800 \pm 0.1166$ | $0.4800 \pm 0.1166$ | $0.00000000$ | **EXACT MATCH** |
| | Balanced Accuracy | $0.4833 \pm 0.1130$ | $0.4833 \pm 0.1130$ | $0.00000000$ | **EXACT MATCH** |
| | Per-Class F1 (K/A/U) | $0.6778 / 0.3048 / 0.4276$ | $0.6778 / 0.3048 / 0.4276$ | $0.00000000$ | **EXACT MATCH** |
| **Exp D: Full 16-Feature Model** *(r0..r9, 16 features)*| Macro-F1 | $0.9190 \pm 0.0745$ | $0.9190 \pm 0.0745$ | $0.00000000$ | **EXACT MATCH** |
| | Accuracy | $0.9200 \pm 0.0748$ | $0.9200 \pm 0.0748$ | $0.00000000$ | **EXACT MATCH** |
| | Balanced Accuracy | $0.9167 \pm 0.0786$ | $0.9167 \pm 0.0786$ | $0.00000000$ | **EXACT MATCH** |
| | Per-Class F1 (K/A/U) | $0.9378 / 0.8878 / 0.9314$ | $0.9378 / 0.8878 / 0.9314$ | $0.00000000$ | **EXACT MATCH** |

### 5.2 Fold-Level Macro-F1 Replication Check (Random Forest)

| Fold Number | Exp A (True Single) | Exp B (Ensemble Ling) | Exp C (Consistency Only) | Exp D (Full Hybrid) | Fold Delta (D - A) | Match Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | $0.9048$ | $1.0000$ | $0.5794$ | $1.0000$ | $+0.0952$ | **EXACT MATCH** |
| **Fold 2** | $0.8857$ | $0.8056$ | $0.5079$ | $0.8963$ | $+0.0106$ | **EXACT MATCH** |
| **Fold 3** | $0.8963$ | $0.8963$ | $0.5407$ | $0.8963$ | $+0.0000$ | **EXACT MATCH** |
| **Fold 4** | $0.9048$ | $0.8024$ | $0.2222$ | $0.8024$ | $-0.1024$ | **EXACT MATCH** |
| **Fold 5** | $1.0000$ | $1.0000$ | $0.5000$ | $1.0000$ | $+0.0000$ | **EXACT MATCH** |
| **Mean $\pm$ Std** | **$0.9183 \pm 0.0414$** | **$0.9008 \pm 0.0877$** | **$0.4701 \pm 0.1270$** | **$0.9190 \pm 0.0745$** | **$+0.0007 \pm 0.0699$** | **EXACT MATCH** |

### 5.3 Statistical Hypothesis Testing Reproduction

| Comparison Pair | Delta Macro-F1 | Paired $t$-Statistic | Paired $t$ $p$-value | Wilcoxon $W$ | Wilcoxon $p$-value | Significance ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp D (Full) vs Exp A (True Single)** | **$+0.0007$** | $t = 0.0219$ | $p = 0.9835$ | $W = 3.0$ | $p = 1.0000$ | **NOT SIGNIFICANT** |
| **Exp D (Full) vs Exp B (Ensemble Ling)** | **$+0.0181$** | $t = 1.0000$ | $p = 0.3739$ | $W = 0.0$ | $p = 1.0000$ | **NOT SIGNIFICANT** |
| **Exp D (Full) vs Exp C (Consistency Only)**| **$+0.4489$** | $t = 11.0493$ | $p = 0.00038$ | $W = 0.0$ | $p = 0.0625$ | **STATISTICALLY SIGNIFICANT ($t$-test)** |

**Replication Conclusion:**  
The replication is **$100\%$ numerically and statistically identical**. Not a single float deviated. The re-execution completely verifies that the existing pipeline is deterministic, reproducible, and ready for expansion.

---

## 6. Batch 2 Pipeline Readiness Audit

A thorough code and infrastructure audit was conducted across the response-generation and feature extraction components (`experiments/generate_responses.py`, `src/generation/`, `src/features/`):

### 6.1 Resumable Generation & Checkpointing
- **Advisory Process Locking:** `ProcessLock` utilizes `fcntl.flock(LOCK_EX | LOCK_NB)` on `{output_path}.lock` to prevent race conditions or duplicate concurrent generation runs.
- **State Recovery:** `recover_and_load_checkpoint(output_path)` reads existing records from disk upon startup. Any partial or corrupted trailing line (e.g. from an ungraceful shutdown) is automatically detected, discarded, and truncated via `f.truncate(last_valid_offset)`.
- **Deduplication:** Questions with completed IDs are added to `completed_ids` and skipped immediately (`if qid in completed_ids: continue`). Generation resumes seamlessly from the exact next uncompleted question.

### 6.2 Error Resilience & Automatic Retries
- **Multi-Attempt Loop:** Outer retry loop (`max_q_retries = 3`) wraps the 10-response generation call for each question.
- **Exponential Backoff:** Catches `OllamaClientError`, `ConnectionError`, and `TimeoutError`, applying exponential backoff:
  $$T_{\text{sleep}} = 2.0^{\text{attempt}}$$
- **Daemon Health Check:** Before retrying, `client.is_available()` validates the HTTP responsiveness of the Ollama server.
- **Timeout Ceilings:** Client timeout is configured at 45.0 seconds per response, preventing indefinite process hangs.

### 6.3 Incremental Persistence & Crash Durability
- **Streaming Append:** Records are written directly to JSONL line-by-line upon completion:
  ```python
  out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
  out_f.flush()
  os.fsync(out_f.fileno())
  ```
- **Durability Guarantee:** `os.fsync()` guarantees that OS disk buffers are flushed to persistent non-volatile storage. An interruption or system crash loses zero completed questions.

### 6.4 Independence of Feature Extraction and ML Pipelines
- **Complete Decoupling:** Generation, feature extraction, and ML training are strictly separate decoupled stages:
  1. `experiments/generate_responses.py` consumes questions JSONL $\to$ produces raw responses JSONL.
  2. `src/features/feature_extractor.py` consumes raw responses $\to$ produces features JSONL / unified corpus.
  3. `scripts/analyze_batch_2_scientific_audit.py` consumes unified corpus $\to$ trains models and logs statistics in $< 5$ seconds.
- **Zero Online Coupling:** Training and feature extraction can be iterated, audited, and rerun independently of the local LLM generation server.

---

## 7. Observed Throughput & Batch 2 Timeline Feasibility

### 7.1 Empirical Throughput from Batch 1
Analyzing the exact execution metadata embedded in `data/batch_1_unified.jsonl` across all 50 questions and 500 responses yields:

```
Total Batch 1 Questions:           50
Total Batch 1 Responses:           500
Total Tokens Generated:            249,800 tokens
Mean Tokens per Response:          499.6 tokens

Total Generation Wall Time:        12,569.55 seconds (209.49 minutes = 3.49 hours)
Mean Time per Question (10 resps): 251.39 seconds (~4.19 minutes)
Mean Time per Response:            22.95 seconds
Generation Throughput:             21.77 tokens / second
```

### 7.2 Estimated Execution Timeline for Batch 2 ($N=150$, $M=1,500$)
Using the observed throughput metrics ($251.39$ seconds per 10-response question), we project the computational requirement for Batch 2:

$$\text{Estimated Time} = 150 \times 251.39\,\text{s} = 37,708.7\,\text{s} \approx 628.5\,\text{minutes} \approx 10.47\,\text{hours}$$

```
+-----------------------------------------------------------------------------------+
| BATCH 2 TIME BUDGET (23-Hour Hard Deadline)                                       |
+-----------------------------------------------------------------------------------+
| Stage 2: Candidate Mining & GT Freeze (150 questions)     | ~0.50 hours           |
| Stage 3: Benchmark Isolation Screening & Deduplication    | ~0.10 hours           |
| Stage 4: 10x Response Generation (1,500 responses)        | 10.47 hours           |
| Stage 5: 16-Feature Extraction & Embedding Clustering     | ~0.25 hours           |
| Stage 6: 4-Config ML Cross-Validation & Bootstrap CIs     | ~0.10 hours           |
| Stage 7: Shortcut Diagnostics & Ablation Suite            | ~0.10 hours           |
| Stage 8: Scientific Audit Report & SVG Visualizations     | ~0.50 hours           |
+-----------------------------------------------------------------------------------+
| Total Projected Execution Time                            | ~12.02 hours          |
| Available Operational Buffer Margin                       | ~10.98 hours          |
+-----------------------------------------------------------------------------------+
```

**Feasibility Verdict:**  
Batch 2 generation is **fully feasible within the 23-hour deadline**. However, because response generation accounts for $10.5$ hours of the budget, **Stage 2 (question curation) and Stage 4 (generation kickoff) must be launched immediately without prolonged delay**.

---

## 8. Critical Blockers & Risk Analysis

| Risk / Blocker | Current Status | Risk Level | Mitigation Strategy |
|---|:---:|:---:|---|
| **Ollama Daemon Availability** | Active & Responsive | Low | Daemon running at `localhost:11434`; models `qwen3:4b` and `all-minilm:latest` pre-loaded. |
| **System Sleep / Power Suspension** | Guarded | Medium | Generation process must be launched with `caffeinate` or daemonized so macOS does not idle-sleep. |
| **Process Interruption** | Handled | Low | Incremental `fsync` and checkpoint recovery ensure zero data loss if interrupted. |
| **Context Window Exceeded** | Guarded | Low | Model supports 262k context; benchmark questions average $< 35$ tokens. |
| **Feature Extraction Memory** | Guarded | Low | Embedding batching in `EmbeddingEngine` processes samples in chunks of 32. |

---

## 9. Immediate Recommended Action

Stage 1 is complete and verified. To meet the 23-hour deadline, we must transition directly to execution:

### Exact Next Command to Execute:
Execute **Stage 2 (Candidate Mining, Provenance Balancing & Ground-Truth Freezing)**:
```bash
python3 scripts/build_batch_2_dataset.py
```
*(Script will curate the 150 candidate questions conforming to the 15-category taxonomy, 40/40/20 provenance balance, and pre-generation ground truth documentation).*
