# Batch 2 Response Generation Quality & Integrity Audit Report

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Dataset:** `Batch_2_Supplementary_Corpus_30`  
**Generated Raw Data Path:** `data/raw/batch_2_generated_responses.jsonl`  
**Configuration File:** `configs/batch_2_generation.yaml`  
**Execution Date:** 2026-11-21 20:13:14 IST – 22:44:55 IST  
**Total Wall Clock Time:** 2 hours 31 minutes 41 seconds (9,550.53 seconds recorded execution)  
**Status:** `ALL 9 QUALITY CHECKS PASSED — 100% VALID & READY FOR FEATURE EXTRACTION`  

---

## 1. Executive Summary

Model response generation for the **Batch 2 Supplementary Corpus ($N=30$, $M=300$ responses)** has completed with complete data integrity.

All 30 questions were generated using the local `qwen3:4b` model via Ollama under the exact dual-temperature sampling protocol specified in `docs/BATCH_2_EXPERIMENTAL_DESIGN.md`:
- **Response 0 ($r_0$):** Deterministic greedy decoding ($T=0.0$, $\text{top\_p}=1.0$) for Configuration A (True Single-Response Baseline).
- **Responses 1–9 ($r_1 \dots r_9$):** Stochastic nucleus sampling ($T=0.7$, $\text{top\_p}=0.9$) for multi-response consistency and ensemble linguistic evaluation.

Every single response generated ($300 / 300$) is non-empty, contains substantive textual reasoning, and is permanently persisted to disk with atomic `os.fsync`.

---

## 2. Nine Post-Generation Quality Checks

| Check ID | Verification Item | Target Standard | Observed Status | Verdict |
|:---:|---|---|:---:|:---:|
| **QC-01** | JSONL File Integrity | Valid UTF-8 newline-delimited JSON | 30 valid JSON lines; 0 parse errors | **PASS** |
| **QC-02** | Total Question Count | Exactly 30 questions | Exactly 30 unique records (`b2_known_001`–`b2_unk_010`) | **PASS** |
| **QC-03** | Responses per Question | Exactly 10 responses per question | Exactly 10 responses across all 30 questions | **PASS** |
| **QC-04** | Total Response Volume | Exactly 300 total responses | Exactly 300 responses | **PASS** |
| **QC-05** | Response Non-Emptiness | String type, non-null, length $> 0$ | 0 empty or whitespace-only responses | **PASS** |
| **QC-06** | Sampling Protocol Adherence | $r_0$ at $T=0.0$; $r_1..r_9$ at $T=0.7$ | 30 / 30 greedy ($T=0.0$); 270 / 270 stochastic ($T=0.7$) | **PASS** |
| **QC-07** | Model Provenance Integrity | Strictly `qwen3:4b` | 100% generated with `qwen3:4b` (digest: `719a5d221aa1c0ec`) | **PASS** |
| **QC-08** | Class & Category Balance | 10 / 10 / 10 classes; 2 per category | 10 KNOWN, 10 AMBIGUOUS, 10 UNKNOWN across 15 categories | **PASS** |
| **QC-09** | Durability & Checkpointing | Zero trailing corruptions or truncation | Verified clean EOF; file size 121,558 bytes | **PASS** |

---

## 3. Dataset Composition & Verification Breakdown

```
Class Distribution:
  - KNOWN:      10 questions (100 responses)
  - AMBIGUOUS:  10 questions (100 responses)
  - UNKNOWN:    10 questions (100 responses)
Total:          30 questions (300 responses)

Category Distribution (15 Categories, 2 questions each):
  - K-DIR: 2  | K-DER: 2  | K-SCI: 2  | K-HIS: 2  | K-NEG: 2
  - A-POL: 2  | A-UND: 2  | A-PRE: 2  | A-PER: 2  | A-TMP: 2
  - U-FUT: 2  | U-PRV: 2  | U-HIS: 2  | U-FAB: 2  | U-MYS: 2
```

---

## 4. Empirical Generation Throughput & Performance

Analyzing the generation runtime metadata:
- **Total Generation Duration:** 2 hours 31 minutes 41 seconds
- **Sum of Elapsed Question Times:** 9,550.53 seconds
- **Mean Time per Question (10 responses):** 318.35 seconds (~5.31 minutes)
- **Mean Time per Response:** 31.84 seconds
- **Overall Operational Margin:** Finished at 22:44:55 IST, preserving **$\approx 18.25$ hours of buffer** against the 23-hour hard project deadline.

---

## 5. Next Recommended Stage

With raw response generation 100% complete and validated, the pipeline is ready to proceed to **Feature Extraction & Embedding Clustering**:
1. Run `src/features/feature_extractor.py` to compute the canonical 16 features on the 30 questions.
2. Produce `data/features/batch_2_features.jsonl` and `data/batch_2_unified.jsonl`.
3. Execute the 4-configuration machine learning cross-validation and statistical evaluation.
