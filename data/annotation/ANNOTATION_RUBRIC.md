# Epistemic Uncertainty Annotation Rubric (v1.1)

**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  
**Scope:** Human Annotation and Gold-Label Validation for the 300-Question Benchmark  
**Document ID:** `data/annotation/ANNOTATION_RUBRIC.md`  

---

## 1. Overview & Theoretical Framework

The objective of this annotation framework is to establish an authoritative, reproducible ground-truth classification of user queries into three mutually exclusive epistemic uncertainty classes:

1. **`KNOWN`** (Determinate Epistemic Knowledge)
2. **`AMBIGUOUS`** (Under-specified, Polysemous, or Multi-referent Queries)
3. **`UNKNOWN`** (Epistemically Inaccessible, Contingent, or Presupposition-Failed Queries)

This taxonomy is grounded in formal epistemic logic, question-answering pragmatics, and semantic calibration theory. Annotators must evaluate queries solely on their **epistemic answerability** under public human knowledge—not whether an arbitrary LLM can guess or fabricate an answer.

---

## 2. Core Epistemic Principles

Annotators must adhere to four foundational principles:

1. **Ambiguity Is Not Ignorance ($\text{Ambiguity} \neq \text{Lack of Knowledge}$):**  
   An ambiguous query possesses multiple legitimate, valid answers depending on unstated context (e.g., *"What is the capital of Georgia?"*). It is not "unanswerable"; rather, any single definitive answer is incomplete or arbitrarily scoped without clarification.
2. **Consensus Is Not Truth ($\text{Agreement} \neq \text{Fact}$):**  
   The fact that a majority of people or an LLM default to a popular answer on a subjective query (e.g., *"What is the best programming language?"*) does not render the query `KNOWN`. Subjective value judgments without objective criteria are fundamentally `AMBIGUOUS`.
3. **Unanswerability Is Distinct From Disambiguation:**  
   If no entity in the universe possesses the requested empirical information (e.g., private passwords, chaotic future outcomes), the query is `UNKNOWN`, not ambiguous.
4. **Presupposition Failures Are Epistemically Invalid:**  
   Queries that presuppose nonexistent entities or impossible physical conditions (e.g., *"What is the boiling point of liquid vacuum?"* or *"Who is the current king of France?"*) cannot have a factual affirmative answer; they are classified as `UNKNOWN` under standard semantic presupposition failure criteria.

---

## 3. Class Definitions & Operational Criteria

### 3.1 Class: `KNOWN`

* **Epistemic Definition:**  
  A sufficiently specified query for which a single, definitive, verifiable answer exists within authoritative public knowledge (encyclopedias, scientific consensus, statutory records, standardized reference texts).
* **Operational Criteria:**
  - The entity, timeframe, jurisdiction, and subject matter are unambiguous or uniquely resolved by global pragmatic context.
  - An objective consensus answer exists across reliable sources.
  - Minor syntactic variations in answer formatting (e.g., *"4"* vs *"four"*, *"Paris"* vs *"Paris, France"*) represent paraphrases of the same single fact.
* **Key Subcategories:**
  1. *Determinate Factual:* Mathematical facts, geographical truths, historical milestones, standardized measurements.  
     *(e.g., "What is the capital of Japan?", "In what year did the Apollo 11 mission land on the Moon?")*
  2. *Definitional & Scientific Standards:* Established scientific laws, anatomical terms, statutory definitions.  
     *(e.g., "What is photosynthesis?", "What is an Application Programming Interface (API)?")*
  3. *Uniquely Anchored Superlatives:* Questions asking for extremes where objective measurement metrics are explicit.  
     *(e.g., "What is the highest mountain peak above sea level on Earth?", "Who was the first person to walk on the Moon?")*

---

### 3.2 Class: `AMBIGUOUS`

* **Epistemic Definition:**  
  A query that admits two or more materially distinct, competing, valid interpretations or answers because key scope parameters are omitted or unconstrained.
* **Operational Criteria:**
  - The query can be legitimately answered in multiple incompatible ways without contradicting facts.
  - The ambiguity cannot be resolved without making an unstated assumption about entity referents, temporal jurisdiction, geographic location, or personal values.
* **Key Subcategories:**
  1. *Entity Polysemy & Name Collisions:* Queries referencing names shared by multiple prominent entities, locations, or works.  
     *(e.g., "What is the capital of Georgia?" [Tbilisi vs Atlanta], "Who is Michael Jordan?" [Athlete vs Actor vs Professor])*
  2. *Unconstrained Temporal / Jurisdictional Scope:* Questions whose truth values vary across jurisdictions or times without specification.  
     *(e.g., "What is the legal voting age?", "Who is the Minister of Finance?")*
  3. *Subjective Superlatives Without Evaluation Metrics:* Questions requesting the "best", "greatest", or "most beautiful" item without defining the criteria or benchmark.  
     *(e.g., "What is the best automobile to buy?", "What is the greatest movie ever made?")*
  4. *Polysemous Common Nouns & Lexical Ambiguity:* Questions asking about words with disparate technical senses.  
     *(e.g., "What is Bridge?" [Card game vs Civil engineering structure])*

---

### 3.3 Class: `UNKNOWN`

* **Epistemic Definition:**  
  A query whose answer is fundamentally inaccessible, unrecorded, objectively undetermined, non-existent, or epistemically blocked under the laws of nature and information theory.
* **Operational Criteria:**
  - Even with omniscient access to all public libraries and the internet, no authoritative factual answer exists.
  - Any direct factual assertion is speculation, hallucination, or conjecture.
* **Key Subcategories:**
  1. *Private & Cryptographically Inaccessible Information:* Passwords, private keys, undisclosed personal thoughts, unrecorded private actions.  
     *(e.g., "What is the secret passcode stored in my personal safe?", "What was John Doe thinking at 3:15 PM yesterday?")*
  2. *Future Contingencies & Unsettled Events:* Non-deterministic future human events, stochastic outcomes, future sporting scores, future stock prices.  
     *(e.g., "What will Apple stock close at on December 31, 2035?", "Which country will win the FIFA World Cup in 2062?")*
  3. *Lost, Unrecorded, or Epistemically Inaccessible Historical Arcana:* Historic events that occurred without any surviving documentation or archaeological evidence.  
     *(e.g., "What were the exact words spoken by the first human in history?", "What notes were written on the discarded scrap draft of Shakespeare's lost youth play?")*
  4. *Presupposition Failures & Category Errors:* Questions predicated on false existential premises or logical impossibilities.  
     *(e.g., "What is the name of the current King of the United States?", "How many corners does a Euclidean circle have?", "What is the boiling point of pure vacuum?")*

---

## 4. Boundary Disambiguation Decision Tree

When evaluating boundary items, apply the following deterministic sequence:

```
[Start Query Evaluation]
        |
        v
Does the query contain a false presupposition, logical contradiction, or non-existent entity?
  ├── YES ──> [UNKNOWN] (Presupposition Failure)
  └── NO
        |
        v
Does the query request future contingent events, unrecorded private secrets, or physically unprovable facts?
  ├── YES ──> [UNKNOWN] (Epistemically Inaccessible)
  └── NO
        |
        v
Does the query use subjective evaluative terms ("best", "favorite", "coolest") without objective constraints?
  ├── YES ──> [AMBIGUOUS] (Value-based Multi-interpretation)
  └── NO
        |
        v
Does the query refer to multiple distinct prominent entities or omit critical geographical/temporal scope?
  ├── YES ──> [AMBIGUOUS] (Polysemy / Scope Ambiguity)
  └── NO
        |
        v
Is there a single, objectively verifiable, authoritative answer in public knowledge?
  ├── YES ──> [KNOWN] (Determinate Fact)
  └── NO  ──> Flag [requires_adjudication = true] for panel review.
```

---

## 5. Specific Edge Cases & Precedents

| Query Type | Case Example | Correct Label | Epistemic Rationale |
|---|---|---|---|
| **Subjective Superlative** | *"What is the best movie of all time?"* | `AMBIGUOUS` | Multiple legitimate artistic and subjective criteria exist (Sight & Sound critics poll, IMDb popular vote, box office gross). |
| **Objectively Anchored Superlative** | *"What is the highest-grossing film of all time (unadjusted for inflation)?"* | `KNOWN` | Objective metric and definitive historical box office record exist. |
| **Name Collision** | *"Who won the US Open in 2020?"* | `AMBIGUOUS` | Refers ambiguously to Men's Tennis, Women's Tennis, or Golf. |
| **Temporally Bound Historic Fact** | *"Who was President of the United States in 1863?"* | `KNOWN` | Single historical fact (Abraham Lincoln). |
| **Future Contingency** | *"Who will be President of the United States in 2045?"* | `UNKNOWN` | Undetermined future event dependent on contingent human actions. |
| **Presupposition Failure** | *"What color is the hair of the Queen of the United States?"* | `UNKNOWN` | False existential premise; no Queen of the United States exists. |
| **Scientific Mystery / Frontier Knowledge** | *"What caused the initial singularity of the Big Bang?"* | `UNKNOWN` | Beyond empirical consensus in modern theoretical physics; currently undetermined. |
| **Definitional Standard** | *"What is the freezing point of pure water at 1 atm?"* | `KNOWN` | Standardized thermodynamic constant ($0^\circ\text{C} / 32^\circ\text{F}$). |

---

## 6. Confidence Scoring Guidelines

Annotators must record their confidence on every item:
- **`HIGH`:** Clear-cut application of the rubric with zero hesitation. Verified reference exists (for KNOWN) or clear epistemic barrier exists (for UNKNOWN/AMBIGUOUS).
- **`MEDIUM`:** Minor ambiguity in phrasing, but category assignment firmly matches rubric logic after brief consideration.
- **`LOW`:** Genuine boundary case where arguments could reasonably be made for two categories. **Must set `requires_adjudication = true` and provide detailed rationale.**
