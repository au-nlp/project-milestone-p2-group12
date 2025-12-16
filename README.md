# Distilling GPT-4 Preferences for Summarization: Metric Calibration and DPO Training Under Limited Compute

## Abstract
This repository contains the complete code and experiments for the final project
“Distilling GPT-4 Preferences for Summarization: Metric Calibration and DPO Training Under Limited Compute.”

The project implements an end-to-end pipeline for preference-aligned Reddit TL;DR summarization, including supervised fine-tuning, automatic preference construction, Direct Preference Optimization (DPO), and evaluation with both automatic metrics and GPT-4 judgments.

### What This Project Does

Trains a supervised TL;DR summarization baseline (Flan-T5)

Generates diverse candidate summaries from multiple models

Scores candidates using ROUGE, BERTScore, and NLI-based factuality

Calibrates metric weights using GPT-4 pairwise judgments

Constructs automatic preference pairs

Trains a DPO-aligned summarization model

Evaluates improvements using automatic metrics and GPT-4 win rate

---
## Contributions

1. **Hybrid Candidate Generation Strategy**  
Candidate summaries are generated not only from the SFT baseline, but also from external summarization models (DistilBART-CNN and BART-XSum), producing stylistically and factually diverse outputs for preference construction.

2. **LLM-Calibrated Scoring Mechanism** 
A data-driven metric calibration procedure is implemented. GPT-4 is used as a pairwise evaluator on a sampled subset of summaries, and grid search is applied to determine optimal weights for ROUGE, BERTScore, and factuality metrics.

3. **Dual-Strategy Preference Construction**  
The code constructs two types of preference pairs: human-vs-model pairs that anchor model quality to human-written TL;DRs, and model-vs-model pairs that provide dense relative preference signals among generated candidates.

4. **Factuality-Aware Evaluation**  
 Factual consistency is explicitly considered during both preference construction and evaluation using an NLI-based factuality estimator, and its impact on DPO training is analyzed.

5. **Analysis of Proxy Preference Validity**  
The project includes analysis comparing automatic metric-based preferences with GPT-4 judgments, highlighting limitations of proxy rewards in preference alignment.


---

## Dataset
### Source
We use the **TRL-lib TL;DR dataset** from Hugging Face, containing over 117,000 Reddit posts and human-written summaries.  
Each entry includes:
- **`prompt`**: full Reddit post with title, body, and subreddit metadata;  
- **`completion`**: TL;DR summary written by the user.

---

## Methodology
- All experiments are implemented in `main.ipynb`

### Step 0 - Preprocessing

The raw Reddit TL;DR dataset was parsed into structured fields and cleaned by removing noise, normalizing demographic tags (e.g., [F/22] → <GENDER_FEMALE> <AGE_22>), and filtering incomplete or duplicate entries.

Basic statistics such as length and compression ratio were computed, and the data were split into train (99,929), validation (5,570), and test (5,577) sets.
The final processed dataset (tldr_cleaned) serves as the foundation for model fine-tuning.


### Step 1 — Supervised Fine-Tuning (SFT)
We begin by fine-tuning an encoder–decoder model (`google/flan-t5-base`) on the cleaned TL;DR pairs.  
This serves as our baseline summarization model.  
It learns to produce concise summaries but may not align with human preference in tone or factual accuracy.

This SFT model plays three roles in the pipeline:

- it serves as the baseline model for comparison,

- it is used as the frozen reference policy during DPO training,

- and it generates candidate summaries for preference construction.


---


### Step 2 — Candidate Summary Generation
To construct informative preference data, a diverse pool of candidate summaries is generated for each Reddit post.
Summaries are produced using multiple encoder–decoder models with different inductive biases, including the SFT-tuned Flan-T5 model, DistilBART-CNN, and BART-Large-XSum.

Each model is decoded under multiple strategies such as beam search and temperature-based sampling to induce stylistic variation and surface different error modes.
All generated summaries are stored and associated with their source posts for subsequent scoring.

This step produces a large and heterogeneous candidate set that is critical for reliable preference learning.

---


### Step 3 — Automatic Metric Computation and Calibration
Each candidate summary is scored along three complementary dimensions:

| Dimension            | Metric              | What it Measures                     |
| -------------------- | ------------------- | ------------------------------------ |
| Coverage             | ROUGE-L             | Content overlap with reference TL;DR |
| Semantic Quality     | BERTScore           | Semantic similarity to reference     |
| Factual Faithfulness | NLI-based factuality | Consistency with source post         |

To determine how to rank these candidates objectively, we avoid manual weight tuning. Instead:
1. We sample 400 pairs of summaries.
2. We use GPT-4 as an evaluator to judge which summary is better based on Factuality, Conciseness, and Coherence.
3. We calculate ROUGE, BERTScore, and Factuality scores locally.
4. We perform a Grid Search to find the optimal weights $(w_R, w_B, w_F)$ that maximize the agreement between our linear formula and GPT-4's decisions.
   
$$S_{\text{calibrated}} = w_R \times \text{ROUGE} + w_B \times \text{BERTScore} + w_F \times \text{Factuality}$$



The highest-scoring candidate is marked as **chosen**, and the lowest as **rejected**, provided the score gap ≥ 0.05.  
Pairs with minimal difference are discarded to avoid label noise.

---

### Step 4 — Preference Pair Construction
Using the calibrated scoring function, we construct a mixed preference dataset:

Factuality Safety Lock: Any candidate with a Factuality score below a threshold (e.g., 0.5) is immediately discarded to prevent hallucinations from becoming chosen samples.

Type A: Alignment Pairs (Human vs. Model)

- Chosen: Human Reference (Gold TL;DR).

- Rejected: The highest-scoring model candidate.

- Goal: Raise the quality ceiling and force alignment with human intent.

Type B: Robustness Pairs (Model vs. Model)

- Chosen: Higher-scoring model candidate.

- Rejected: Lower-scoring model candidate (requires Score Gap ≥ 0.05).

- Goal: Teach the model fine-grained discrimination and self-correction within its own distribution.

---

### Step 5 — Direct Preference Optimization (DPO)
The preference-aligned summarization model is trained using Direct Preference Optimization.
The SFT-tuned Flan-T5 model is used as both the initial policy and the frozen reference policy.

During training, the model is optimized to increase the relative likelihood of preferred summaries over rejected ones under the DPO objective.
Multiple models are trained using different values of the regularization coefficient β to study the effect of preference strength.

---

### Step 6 — Evaluation
The SFT baseline and DPO-trained models are evaluated using two complementary evaluation settings:
1. **Automatic Metrics**
  Generated summaries are evaluated using ROUGE-L, BERTScore, and the NLI-based factuality metric to quantify changes in coverage, semantic similarity, and factual consistency.


2. **LLM-Based Pairwise Evaluation**
  GPT-4 is used as an independent evaluator to perform blind pairwise comparisons between summaries generated by the SFT and DPO models.
Win rates are computed based on GPT-4’s preferences to assess alignment with LLM-level judgments.

---
## Timeline

| Week  | Task                          | Deliverable                                      |
| ----- | ----------------------------- | ------------------------------------------------ |
| **1** | Data exploration and cleaning | Dataset report and preprocessing scripts         |
| **2** | Baseline SFT fine-tuning      | Model checkpoint and qualitative samples         |
| **3** | Candidate generation          | 3–5 diverse summaries per post                   |
| **4** | Preference construction       | JSONL file of `(prompt, chosen, rejected)` pairs |
| **5** | DPO training                  | Preference-aligned model checkpoint              |
| **6** | Evaluation                    | Metrics comparison, win-rate curve visualization |
| **7** | Discussion and integration    | Final README and project report                  |

---
## Team Organization

- **Yaning Wen**  
  Responsible for overall project coordination, system design, and primary implementation and execution of all code and experiments.

  Leads and participates in all stages of the project, including data preprocessing, supervised fine-tuning, candidate generation, metric computation, preference construction, DPO training, and evaluation.

  Responsible for writing and structuring the final project report.

- **Wei Xiong**  
  Assists Yaning Wen with model training, with a focus on supporting Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) experiments.
   
  Focuses on implementing fine-tuning procedures, managing training configurations, conducting hyperparameter tuning, and monitoring performance under the coordination of the teamembers.
  
  Assists in writing and reviewing the final project report.

- **Yunting Wang**  
  Provides comprehensive project assistance and coordination support.  
  Assists other members in tasks such as dataset handling, metric computation, and result verification.  
  Contributes to documentation, organization of experiment records, maintenance of the repository, and overall workflow coordination.
  Assists in report writing and proofreading, helping verify experimental descriptions and results consistency.

All members will participate in weekly coordination meetings to discuss progress and ensure that milestones are met. 


