# Category Generalization and Lexical Shortcut Audit

This experiment provides a rigorous, isolated audit of the single-response TF-IDF Linear SVM classifier (which scored 83.3% accuracy in standard cross-validation) to determine whether it genuinely learns uncertainty mechanics or relies on lexical shortcuts and domain vocabulary memorization.

## Directory Structure

```text
experiments/generalization_and_lexical_audit/
├── README.md                           # This overview and execution guide
├── REPORT.md                           # Comprehensive scientific audit report
├── config.json                         # Configuration parameters
├── reproduce_baseline.py               # Stage 1: Exact reproduction of 83.3% baseline
├── category_heldout_evaluation.py      # Stage 2: 15-Fold Leave-One-Category-Out evaluation
├── lexical_audit.py                    # Stages 3 & 4: Systematic lexical masking audit (7 experiments)
├── statistical_analysis.py             # Stage 5: McNemar and non-parametric bootstrap analysis
├── generate_figures.py                 # Stage 6: Publication-grade figure generation
├── results/                            # CSV artifacts of all benchmarks
│   ├── baseline_reproduction.csv
│   ├── category_heldout_results.csv
│   ├── lexical_audit_results.csv
│   ├── per_category_results.csv
│   ├── statistical_comparisons.csv
│   └── oof_predictions.csv
├── figures/                            # 4 publication-quality visualizations
│   ├── model_comparison.png
│   ├── category_performance.png
│   ├── lexical_audit.png
│   └── confusion_matrices.png
└── audit/                              # Regression and integrity checks
    └── regression_audit.txt
```

## How to Run

1. **Reproduce Baseline:**
   ```bash
   python3 experiments/generalization_and_lexical_audit/reproduce_baseline.py
   ```

2. **Run Category-Held-Out Evaluation:**
   ```bash
   python3 experiments/generalization_and_lexical_audit/category_heldout_evaluation.py
   ```

3. **Run Lexical Shortcut Audit:**
   ```bash
   python3 experiments/generalization_and_lexical_audit/lexical_audit.py
   ```

4. **Run Statistical Analysis:**
   ```bash
   python3 experiments/generalization_and_lexical_audit/statistical_analysis.py
   ```

5. **Generate Figures:**
   ```bash
   python3 experiments/generalization_and_lexical_audit/generate_figures.py
   ```

## Key Findings

- Under Leave-One-Category-Out, TF-IDF Linear SVM collapses from **83.3% to 65.0%** (Macro-F1 drops from **0.8365 to 0.6490**, Exact Binomial $p = 0.000977$).
- Filtering text to pure epistemic markers drops standard CV accuracy from **83.3% to 68.3%** ($p = 0.0490$).
- Proves conclusively that single-response TF-IDF models rely on domain vocabulary shortcuts rather than robust uncertainty detection.
