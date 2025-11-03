# Preference-Aligned TL;DR Summarization Using Direct Preference Optimization

## Abstract
Traditional summarization models, trained with supervised objectives like maximum likelihood estimation, often optimize for lexical overlap rather than human judgment. As a result, their summaries may be fluent yet unfaithful, verbose, or stylistically inconsistent.  
This project explores preference alignment for Reddit TL;DR summarization using **Direct Preference Optimization (DPO)**. Unlike standard fine-tuning, which imitates reference summaries, DPO directly learns to prefer summaries that better reflect human-like preferences—conciseness, coherence, and factual faithfulness.  
We fine-tune a baseline model on the Reddit TL;DR dataset and automatically construct preference pairs (`chosen`, `rejected`) using a combination of ROUGE, BERTScore, and factual-consistency metrics. These serve as proxy signals for human judgment. The DPO-aligned model is expected to outperform the baseline on both automatic metrics and human win rate, showing that preference alignment can be achieved without manual labeling.  
This work also investigates the reliability of automatic preference generation and discusses potential limitations and mitigation strategies.

---

## Contributions

1. **Automatic Preference Generation Framework**  
   Develop an unsupervised pipeline that automatically generates preference pairs from model outputs using ROUGE, BERTScore, and factuality metrics.

2. **Preference Alignment on User-Generated Text**  
   Apply DPO to Reddit TL;DR data—informal, emotional, and noisy user content rarely used in alignment studies—to test robustness in real-world conditions.

3. **Factuality-Aware Evaluation**  
   Integrate factual consistency metrics (SummaC, QAFactEval) into evaluation, quantifying how DPO affects truthfulness and hallucination reduction.

4. **Analysis of Proxy Preference Validity**  
   Examine whether automatic metrics can reliably approximate human preferences, and propose validation and correction methods.

5. **Reproducible and Scalable Pipeline**  
   Provide an open, reproducible implementation of SFT → Candidate Generation → Preference Construction → DPO → Evaluation, facilitating future research on automatic alignment.

---

## Dataset
### Source
We use the **TRL-lib TL;DR dataset** from Hugging Face, containing over 117,000 Reddit posts and human-written summaries.  
Each entry includes:
- **`prompt`**: full Reddit post with title, body, and subreddit metadata;  
- **`completion`**: TL;DR summary written by the user.

The dataset reflects the informal, emotional, and stylistically varied nature of Reddit writing.  
It is ideal for testing how preference alignment behaves on non-news, non-curated text.

### Data Processing
- Remove incomplete or extremely short posts and summaries.  
- Normalize line breaks and remove redundant “TL;DR:” tokens.  
- Preserve subreddit and title information to maintain contextual richness.  
- Tokenize and truncate to manageable sequence lengths (e.g., ≤1024 input tokens).  
- Split into training (117k), validation (6.4k), and test (6.5k) subsets.

---

## Methodology

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
- Beam search (`num_beams = 4`)
- Nucleus sampling (`top_p = 0.9–0.95`)
- Temperature scaling (`T = 0.7–1.0`)

This ensures diversity among outputs, producing both strong and weak candidates for comparison.

---

### Step 3 — Automatic Preference Construction
Each candidate summary is scored along three complementary dimensions:

| Dimension            | Metric              | What it Measures                     |
| -------------------- | ------------------- | ------------------------------------ |
| Coverage             | ROUGE-L             | Content overlap with reference TL;DR |
| Semantic Quality     | BERTScore           | Semantic similarity to reference     |
| Factual Faithfulness | SummaC / QAFactEval | Consistency with source post         |

These scores are aggregated into a weighted sum:  

$S = 0.5 \times \text{ROUGE} + 0.3 \times \text{BERTScore} + 0.2 \times \text{Factuality}$

The highest-scoring candidate is marked as **chosen**, and the lowest as **rejected**, provided the score gap ≥ 0.05.  
Pairs with minimal difference are discarded to avoid label noise.

---

### Why Automatic Preferences Are Reasonable
Previous studies (Zhang et al., 2020; Laban et al., 2021; Kryscinski et al., 2022) show that these metrics correlate strongly (ρ ≈ 0.6–0.8) with human evaluation of summary quality.  
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

## Expected Outcomes
- A DPO-aligned summarization model that generates **more factual, concise, and human-aligned TL;DRs** than the baseline.  
- Quantitative improvements in factual consistency and preference win rate.  
- An analysis of when automatic preference construction succeeds or fails.  
- A reproducible open-source pipeline demonstrating **automatic preference alignment** without human annotation.

---

## References
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290.  
- Laban, P. et al. (2021). *SummaC: Re-evaluating Summarization Evaluation for Faithfulness.* ACL.  
- Kryscinski, W. et al. (2022). *Evaluating the Factual Consistency of Abstractive Summaries with QA-based Metrics.* TACL.  
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.  
- Fabbri, A. et al. (2021). *SummEval: Re-evaluating Automatic Metrics for Summarization.* TACL.  

