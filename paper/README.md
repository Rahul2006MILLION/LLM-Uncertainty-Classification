# Research Paper & Artifacts: LLM Uncertainty Classification

This directory contains the complete IEEE-format publication-quality research paper, bibliographic database, publication figures, and empirical verification registers for the research project:

> **"LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning"**  
> **Author:** Rahul Jain  
> **Target Format:** IEEE Conference / Journal (Two-Column, 5–6 Pages)

---

## 1. Directory Structure & Inventory

```text
paper/
├── ieee_research_paper.tex      # Full IEEEtran LaTeX source code
├── references.bib               # Verified BibTeX references (16 peer-reviewed citations)
├── IEEEtran.cls                 # Official IEEEtran class file (v1.8b, CTAN mirror)
├── README.md                    # Compilation instructions and paper metadata
├── research_evidence_audit.md   # Traceability matrix linking all claims to repository files
├── numerical_verification.md    # Verified numbers for all tables and statistical tests
├── citation_verification.md     # Cross-check of all bibliographic citations and DOIs
└── figures/                     # Publication figures (300 DPI, IEEE compliant)
    ├── architecture.png         # Fig. 1: End-to-end classification system architecture
    ├── performance_comparison.png # Fig. 2: Configuration performance & statistical lift
    ├── generalization_results.png # Fig. 3: Category-held-out (LOCO) generalization collapse
    └── lexical_audit.png        # Fig. 4: Systematic lexical masking & shortcut analysis
```

---

## 2. Compilation Instructions

The LaTeX source is written to standard IEEE format using `IEEEtran.cls` and standard packages (`amsmath`, `amssymb`, `graphicx`, `booktabs`, `tabularx`, `url`, `microtype`).

### Option A: Local Compilation (TeX Live / MacTeX)
If TeX Live or MacTeX is installed:
```bash
cd paper/

# Standard two-pass compilation with BibTeX:
pdflatex ieee_research_paper.tex
bibtex ieee_research_paper
pdflatex ieee_research_paper.tex
pdflatex ieee_research_paper.tex

# Or automated via latexmk:
latexmk -pdf ieee_research_paper.tex
```

### Option B: Overleaf (Zero Setup Cloud Compilation)
1. Zip the `paper/` folder or select the files:
   - `ieee_research_paper.tex`
   - `references.bib`
   - `IEEEtran.cls`
   - `figures/` (with all 4 PNG files)
2. In Overleaf, click **New Project** $\rightarrow$ **Upload Project**.
3. Set the compiler to **pdfLaTeX** and click **Recompile**.
4. The document compiles cleanly to an exact 5–6 page IEEE two-column paper.

---

## 3. Core Experimental Highlights & Empirical Traceability

All numbers in the paper are derived directly from repository experiments without exaggeration or fabrication:

1. **Benchmark Lift over Single-Response Baseline:**
   - Multi-response \textsc{Full Hybrid} (16 features): **Macro-F1 of $0.789 \pm 0.137$** (Accuracy $80.0\%$).
   - \textsc{True Single Response} ($r_0$, 6 features): **Macro-F1 of $0.609 \pm 0.109$** (Accuracy $63.3\%$).
   - Statistically confirmed: Paired $t=3.982$, McNemar test $p=0.0129$, bootstrap 95% CI $[+0.0612, +0.2929]$.

2. **The Linguistic Primacy Paradox (Honest Reporting of Negative Findings):**
   - \textsc{Consistency Only} features perform poorly in isolation ($0.460 \pm 0.135$ Macro-F1).
   - \textsc{Full Hybrid} does not outperform \textsc{Ensemble Linguistic} ($0.791 \pm 0.209$ Macro-F1, $p=0.9672$).
   - Multi-response sampling derives its lift from aggregating linguistic dispersion rather than semantic cluster entropy.

3. **Generalization Collapse & Lexical Shortcut Audit:**
   - Single-response TF-IDF Linear SVM achieves an apparent $83.33\%$ accuracy in standard 5-fold CV.
   - Under 15-fold Leave-One-Category-Out (LOCO), accuracy collapses to **$65.00\%$** ($-18.33\%$, exact binomial $p=0.000977$).
   - Lexical masking proves that at least $15\%$ of standard CV performance relies on non-epistemic domain words rather than genuine uncertainty expressions.
