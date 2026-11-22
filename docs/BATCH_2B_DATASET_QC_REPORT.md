# Batch 2B Dataset Quality Control & Validation Report

**Dataset:** `Batch_2B_Targeted_Corpus_30`  
**Path:** `data/batch_2b_questions.jsonl`  
**SHA-256:** `5fa7e2bb64c2a56e28431251de0982d3e0ceed3ba0660eae59375e37e7af8c78`  
**Total Questions:** 30 (Balanced 10 KNOWN, 10 AMBIGUOUS, 10 UNKNOWN)  

---

## 1. Class and Category Balance

| Class | Count | Target | Status |
|---|:---:|:---:|:---:|
| **KNOWN** | 10 | 10 | PASS |
| **AMBIGUOUS** | 10 | 10 | PASS |
| **UNKNOWN** | 10 | 10 | PASS |

### Categories (15 categories, 2 items each):
{
  "K-DIR": 2,
  "K-DER": 2,
  "K-SCI": 2,
  "K-HIS": 2,
  "K-NEG": 2,
  "A-POL": 2,
  "A-UND": 2,
  "A-PRE": 2,
  "A-PER": 2,
  "A-TMP": 2,
  "U-FUT": 2,
  "U-PRV": 2,
  "U-HIS": 2,
  "U-FAB": 2,
  "U-MYS": 2
}

### Provenance Distribution:
- Mined: 12 (40.0%)
- Synthesized: 12 (40.0%)
- Expert Curated: 6 (20.0%)

---

## 2. Question Length Calibration & Shortcut Neutralization

| Class | Mean Length (chars) | Std Dev | Min | Max |
|---|:---:|:---:|:---:|:---:|
| **KNOWN** | 143.4 | 5.6 | 135 | 152 |
| **AMBIGUOUS** | 145.6 | 4.3 | 140 | 153 |
| **UNKNOWN** | 145.4 | 3.4 | 140 | 152 |

**Kruskal-Wallis Length Test:** $H = 1.2269$, $p = 0.5415$ (Statistically non-significant; zero length bias).

---

## 3. Semantic Isolation Screening

| Benchmark Comparison | Threshold | Max Observed Similarity | Status | Violations |
|---|:---:|:---:|:---:|:---:|
| **300-Item Benchmark Isolation** | < 0.75 | 0.669 | PASS | 0 |
| **Batch 1 Separation** | < 0.80 | 0.7496 | PASS | 0 |
| **Batch 2A Separation** | < 0.80 | 0.5407 | PASS | 0 |

---

## 4. Verification Sign-Off

All quality control gates for Batch 2B dataset construction passed successfully with zero errors.
