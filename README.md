# Preference-Aligned TL;DR Summarization Using Direct Preference Optimization

## Abstract
Traditional summarization models, trained with supervised objectives like maximum likelihood estimation, often optimize for lexical overlap rather than human judgment. As a result, their summaries may be fluent yet unfaithful, verbose, or stylistically inconsistent.  
This project explores preference alignment for Reddit TL;DR summarization using **Direct Preference Optimization (DPO)**. Unlike standard fine-tuning, which imitates reference summaries, DPO directly learns to prefer summaries that better reflect human-like preferences—conciseness, coherence, and factual faithfulness.  
We fine-tune a baseline model on the Reddit TL;DR dataset and automatically construct preference pairs (`chosen`, `rejected`) using a combination of ROUGE, BERTScore, and factual-consistency metrics. These serve as proxy signals for human judgment. The DPO-aligned model is expected to outperform the baseline on both automatic metrics and human win rate, showing that preference alignment can be achieved without manual labeling.  
This work also investigates the reliability of automatic preference generation and discusses potential limitations and mitigation strategies.

---

## Contributions

1. **Automatic Preference Generation Framework**  
   Develop an automatic pipeline that constructs preference pairs from model-generated summaries using ROUGE, BERTScore, and factuality metrics, reducing the need for costly human annotations.

2. **Preference Alignment on User-Generated Text**  
   Apply DPO to Reddit TL;DR posts—informal and noisy user-generated content—to evaluate the robustness and generalization of preference alignment in real-world conditions.

3. **Factuality-Aware Evaluation**  
   Integrate factual consistency metrics (SummaC, QAFactEval) into evaluation, quantifying how DPO affects truthfulness and hallucination reduction.

4. **Analysis of Proxy Preference Validity**  
   Provide empirical analysis of whether metric-based preferences approximate human judgments, and propose validation and correction methods.



---

## Dataset
### Source
We use the **TRL-lib TL;DR dataset** from Hugging Face, containing over 117,000 Reddit posts and human-written summaries.  
Each entry includes:
- **`prompt`**: full Reddit post with title, body, and subreddit metadata;  
- **`completion`**: TL;DR summary written by the user.


---

## Methodology

### Step 0 - Preprocessing

The raw Reddit TL;DR dataset was parsed into structured fields and cleaned by removing noise, normalizing demographic tags (e.g., [F/22] → <GENDER_FEMALE> <AGE_22>), and filtering incomplete or duplicate entries.

Basic statistics such as length and compression ratio were computed, and the data were split into train (99,929), validation (5,570), and test (5,577) sets.
The final processed dataset (tldr_cleaned) serves as the foundation for model fine-tuning.

### Step 1 — Supervised Fine-Tuning (SFT)
We begin by fine-tuning an encoder–decoder model (`google/flan-t5-base`) on the cleaned TL;DR pairs.  
This serves as our baseline summarization model.  
It learns to produce concise summaries but may not align with human preference in tone or factual accuracy.

**Purpose:**  
- Establish a strong baseline for comparison.  
- Provide the foundation for generating candidate summaries used in preference training.

---

### Step 2 — Candidate Generation
For each Reddit post, the SFT model generates multiple candidate summaries using diverse decoding parameters:
- Beam search 
- Nucleus sampling 
- Temperature scaling 

This ensures diversity among outputs, producing both strong and weak candidates for comparison.

---

### Step 3 — Automatic Preference Construction
Each candidate summary is scored along three complementary dimensions:

| Dimension            | Metric              | What it Measures                     |
| -------------------- | ------------------- | ------------------------------------ |
| Coverage             | ROUGE-L             | Content overlap with reference TL;DR |
| Semantic Quality     | BERTScore           | Semantic similarity to reference     |
| Factual Faithfulness | SummaC / QAFactEval | Consistency with source post         |


We propose a simple weighted aggregation of ROUGE, BERTScore, and factual consistency metrics as an automatic proxy for human preference, with empirically motivated weights (0.5, 0.3, 0.2).

$S = 0.5 \times \text{ROUGE} + 0.3 \times \text{BERTScore} + 0.2 \times \text{Factuality}$

The highest-scoring candidate is marked as **chosen**, and the lowest as **rejected**, provided the score gap ≥ 0.05.  
Pairs with minimal difference are discarded to avoid label noise.

---

### Why Automatic Preferences Are Reasonable
Prior studies (Zhang et al., 2020; Laban et al., 2021; Kryscinski et al., 2022; Ye et al., 2024) have demonstrated that automatic metrics such as ROUGE, BERTScore, and similarity-based factuality scores correlate strongly (ρ ≈ 0.6–0.8) with human judgments of summary quality and factual consistency.

Since DPO relies only on *relative* ranking, not absolute reward values, even noisy but directionally correct preference pairs can drive effective optimization.  
Thus, metric-based pairs serve as a valid approximation of human preferences when manual labels are unavailable.

---

### Step 4 — Direct Preference Optimization (DPO)
We train a preference-aligned model using the DPO loss:

$L_{\mathrm{DPO}} = -\log \sigma(\beta [\log \pi_\theta(y_c|x) - \log \pi_\theta(y_r|x) - \log \pi_{\mathrm{ref}}(y_c|x) + \log \pi_{\mathrm{ref}}(y_r|x)])$

where β controls the regularization strength and  π_ref  is the frozen baseline model.  
The objective encourages the model to increase the relative likelihood of preferred summaries without explicit reward modeling or reinforcement learning.

---

### Step 5 — Evaluation
We compare the SFT and DPO models on:
1. **ROUGE and BERTScore:** lexical and semantic coverage.  
2. **Factual Consistency (SummaC, QAFactEval):** reduction in hallucinations.  
3. **Win Rate (Preference Agreement):** how often DPO summaries are preferred to SFT outputs.  
4. **Qualitative Case Studies:** manual inspection of improved examples.

The evaluation also tests different sampling temperatures to reproduce and analyze DPO’s robustness trends, as shown in the original paper.

---

### Step 6 — Reliability and Limitations
Automatic preferences may contain errors (e.g., ROUGE favoring longer outputs).  
To assess reliability:
- We analyze score distributions (mean and variance of `S_chosen - S_rejected`).  
- We manually check 100 random pairs for human agreement.  
- We test whether DPO’s improvements hold across subsets with large vs. small score gaps.

If inconsistencies are detected, we will adjust metric weights or thresholds, or combine automatic and GPT-4-based judgments for hybrid verification.

Even if the automatic ranking is imperfect, DPO remains robust as long as >50% of pairs reflect the correct preference direction.

---

### Step 7 — Feasibility Justification
1. **Computationally Light:** DPO avoids reinforcement learning; it only needs log-likelihood computations.  
2. **Data Availability:** Reddit TL;DR dataset is public and sufficiently large for fine-tuning and preference sampling.  
3. **Scientific Validity:** Automatic metrics have established correlation with human evaluations; thus, the experiment is methodologically grounded.  
4. **Evaluability:** Both automatic and manual checks can verify whether DPO improves factuality and alignment.

---


## Expected Outcomes
- A DPO-aligned summarization model that generates **more factual, concise, and human-aligned TL;DRs** than the baseline.  
- Quantitative improvements in factual consistency and preference win rate.  
- An analysis of when automatic preference construction succeeds or fails.  
- A reproducible open-source pipeline demonstrating **automatic preference alignment** without human annotation.

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
| **7** | Discussion and integration    | Final README and project presentation            |

---
## Team Organization

- **Yaning Wen**  
   Responsible for overall project coordination, system design, and implementation of the core pipeline.  
  Leads the fine-tuning of the baseline summarization model (SFT), the construction of automatic preference pairs, and the Direct Preference Optimization (DPO) training stage.  
  Also oversees experiment planning, evaluation design, and integration of results into the final report.

- **Wei Xiong**  
  Responsible for model training and optimization.  
  Focuses on implementing fine-tuning procedures, managing training configurations, conducting hyperparameter tuning, and monitoring performance under the coordination of the teamembers.  
  Contributes to evaluation and internal testing of trained models.

- **Yunting Wang**  
  Provides comprehensive project assistance and coordination support.  
  Assists other members in tasks such as dataset handling, metric computation, and result verification.  
  Contributes to documentation, organization of experiment records, maintenance of the repository, and overall workflow coordination.  

All members will participate in weekly coordination meetings to discuss progress and ensure that milestones are met. 

---
## References
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290. 
arXiv

- Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. TACL. 
direct.mit.edu

- Fabbri, A. R., Wu, C.-S., Liu, W., & Xiong, C. (2022). QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization. NAACL. arXiv:2112.08542. 
arXiv

- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR. 
arXiv

- Fabbri, A., Kryscinski, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021). SummEval: Re-evaluating Summarization Evaluation. TACL. 
aclanthology.org

- Ye, Y. et al. (2024). Using Similarity to Evaluate Factual Consistency in Summaries. arXiv:2409.15090.

