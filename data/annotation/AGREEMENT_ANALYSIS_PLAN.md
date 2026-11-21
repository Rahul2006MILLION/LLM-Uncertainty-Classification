# Inter-Annotator Agreement Analysis Plan

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Target:** 300-Question Human Validation Evaluation  
**File:** `data/annotation/AGREEMENT_ANALYSIS_PLAN.md`  

---

## 1. Objectives

This plan specifies the mathematical, statistical, and diagnostic procedures used to quantify inter-annotator agreement (IAA) between Annotator A and Annotator B on the 300-question benchmark.

The primary objective is to verify whether human annotators achieve high, statistically robust consensus ($\kappa \ge 0.85$) under the blinded annotation protocol, establishing the scientific validity of the benchmark's gold labels.

---

## 2. Statistical Metrics & Mathematical Formulation

Let $\mathcal{D} = \{q_i\}_{i=1}^N$ be the set of $N = 300$ benchmark questions.  
Let $\mathcal{C} = \{\text{KNOWN}, \text{AMBIGUOUS}, \text{UNKNOWN}\}$ be the set of $K = 3$ epistemic classes.  
Let $y_{A, i} \in \mathcal{C}$ and $y_{B, i} \in \mathcal{C}$ be the labels assigned to question $q_i$ by Annotator A and Annotator B, respectively (mapped via the blinding key to original question IDs).

### 2.1 Observed Raw Agreement ($P_o$)

The raw proportion of questions on which both annotators assign identical labels:

$$P_o = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_{A, i} = y_{B, i})$$

where $\mathbb{I}(\cdot)$ is the indicator function.

---

### 2.2 Expected Agreement by Chance ($P_e$)

Under the marginal probability distribution of each annotator's ratings across classes $c \in \mathcal{C}$:

$$P(A = c) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_{A, i} = c)$$

$$P(B = c) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_{B, i} = c)$$

$$P_e = \sum_{c \in \mathcal{C}} P(A = c) \cdot P(B = c)$$

---

### 2.3 Cohen's Kappa Coefficient ($\kappa$)

Cohen's Kappa normalizes the observed agreement by the expected chance agreement:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

* Standard Error ($\text{SE}_\kappa$):

$$\text{SE}_\kappa = \sqrt{\frac{P_o(1 - P_o)}{N (1 - P_e)^2}}$$

* 95% Confidence Interval:

$$\text{CI}_{95\%} = \left[ \kappa - 1.96 \cdot \text{SE}_\kappa, \; \kappa + 1.96 \cdot \text{SE}_\kappa \right]$$

---

### 2.4 Acceptance Criteria & Qualitative Benchmark

Following Landis & Koch (1977) and research-grade natural language processing guidelines:

| $\kappa$ Range | Agreement Interpretation | Project Action |
|---|---|---|
| **$\ge 0.85$** | **Near-Perfect / Strong Consensus** | **Target benchmark passed.** Proceed to standard adjudication on discordant items ($A \ne B$). |
| **$0.70 - 0.84$** | **Substantial Agreement** | Review systematic boundary confusion patterns; mandatory senior panel adjudication for discordant cluster. |
| **$< 0.70$** | **Insufficient Agreement** | Halt workflow; trigger rubric calibration, conduct annotator debrief, and re-annotate ambiguous boundary subset. |

*Target threshold for this project:* **$\kappa \ge 0.85$** (as specified in `data/benchmarks/annotation_manifest_v1.1.json`).

---

## 3. Class-Specific Confusion Matrix & Per-Class Metrics

A $3 \times 3$ contingency table $M \in \mathbb{R}^{3 \times 3}$ is constructed where cell $M_{j, k}$ represents the count of items labeled class $j$ by Annotator A and class $k$ by Annotator B:

$$M_{j, k} = \sum_{i=1}^{N} \mathbb{I}(y_{A, i} = j \land y_{B, i} = k), \quad j, k \in \{\text{KNOWN}, \text{AMBIGUOUS}, \text{UNKNOWN}\}$$

For each class $c \in \mathcal{C}$, treating Annotator A as reference and Annotator B as evaluation (and symmetrically):

- **Per-Class Agreement Proportion ($P_c$):**
  $$P_c = \frac{2 \cdot M_{c, c}}{\sum_{k} M_{c, k} + \sum_{j} M_{j, c}}$$
- **Per-Class Dice / F1 Score:**
  $$F_{1, c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$

---

## 4. Diagnostic Confusion Analysis Plan

Disagreements ($y_{A, i} \ne y_{B, i}$) will be stratified across three primary boundary axes:

1. **Axis 1: `KNOWN` vs `AMBIGUOUS`**
   - *Hypothesis:* Confusion arises when a query has one dominant real-world answer but also minor or historical secondary interpretations (e.g. *"What is the capital of Georgia?"*).
   - *Diagnostic:* Check whether the annotator classifying as `KNOWN` missed polysemy or assumed default Western/US jurisdiction.
2. **Axis 2: `AMBIGUOUS` vs `UNKNOWN`**
   - *Hypothesis:* Confusion arises on subjective queries without consensus (e.g. *"What is the most beautiful flower?"*) vs unknowable future contingencies (e.g. *"Who will win the next election?"*).
   - *Diagnostic:* Verify application of the epistemic boundary rule: multiple valid subjective answers $\to$ `AMBIGUOUS`; zero existing factual answers $\to$ `UNKNOWN`.
3. **Axis 3: `KNOWN` vs `UNKNOWN`**
   - *Hypothesis:* Rare confusion, typically arising from presupposition failures (e.g. *"Who is the king of France?"*) or obscure niche facts that an annotator believed were unrecorded.

---

## 5. Automated Evaluation Script Execution

Upon completion of the blinded annotation files, the project analysis script `scripts/analyze_annotation_agreement.py` will be executed to generate:
1. Exact counts of unanimous agreements ($y_A = y_B$).
2. Exact count and list of discordant items ($y_A \ne y_B$).
3. Computed $P_o$, $P_e$, $\kappa$, and $\text{SE}_\kappa$.
4. Formatted Markdown and LaTeX confusion tables.
5. Export of discordant items directly to `data/annotation/adjudication_input.jsonl`.
