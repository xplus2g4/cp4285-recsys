# Popularity-Opportunity Bias in Collaborative Filtering

- **Authors:** Ziwei Zhu, Yun He, Xing Zhao, Yin Zhang, Jianling Wang, James Caverlee
- **Venue / year:** WSDM 2021 (14th ACM International Conference on Web Search and Data Mining), pp. 85–93
- **Link:** https://doi.org/10.1145/3437963.3441820
- **Local PDF:** `docs/literature/papers/zhu2021popopp.pdf` (author-hosted copy, http://zziwei.github.io/pubs/Ziwei_WSDM_2021.pdf; Semantic Scholar lists the paper as closed access)
- **Source basis:** full text

## Problem
Most popularity-bias work measures a **statistical-parity** gap: whether popular items get recommended more, without conditioning on whether users like them. The authors propose a measure grounded in **equal opportunity**: *among items a user actually likes in the test set*, are more popular ones ranked higher? They call this popularity-opportunity bias and study it on two sides:
- **uPO (user side):** within one user's liked test items, does rank follow popularity?
- **iPO (item side):** across items, does popularity predict an item's expected rank among the users who like it?

## Method
- **Metrics (this paper's own; it does not define PopRSP/PopREO):**
  - **PRU** = −(1/N) Σ_u SRC(pop(Õ_u⁺), rank_u(Õ_u⁺)). This is the negated Spearman correlation between popularity and rank position over each user's test positives, averaged over users. Rank 0 is the top.
  - **PRI** = −SRC(pop(I), avg_rank(I)), where avg_rank(i) is item i's mean rank over the users who have i in their test set.
  - Higher values mean more bias. The authors also checked P(top-k | matched user), which showed the same pattern.
- **Theory:** under two assumptions, MF (RMSE loss) provably produces both uPO and iPO bias: the score for a liked item scales with the item's training popularity. Assumption 1: near-zero training loss, so R̂ ≈ R. Assumption 2: the "key" user–item pair scores in a normalised preference matrix are identically distributed. BPR is reduced to the MF case by re-centring each user's scores.
- **Debiasing:**
  - **Popularity Compensation (PC, post-processing):** C_ui = (1/pop(i))·(R̂_ui·β + 1 − β), added as R̂*_ui = R̂_ui + α·C_ui·n_u/m_u. Here n_u and m_u are per-user norms of the scores and of the compensation over un-interacted items. α ∈ [0.1, 1.5], β ∈ [0, 1].
  - **Reg (in-processing):** adds γ·PCC(R̂⁺, pop(I))² to the loss, i.e. the squared Pearson correlation between scores on positive pairs and item popularity. γ ∈ {1e2,…,1e7}.
- **Baselines:** loss weighting w_i ∝ 1/pop(i)^e (MF-weight) and rescaling of the training data (MF-rescale). The authors class both as fixes for "conventional" popularity bias.

## Assumptions held fixed
- **Test-set positives are taken as ground-truth preference.** The test set is a random 20% of the *logged* interactions, so it inherits exposure/popularity bias. Equal opportunity is conditioned on a biased label.
- Popularity is the number of feedback actions in the data and is treated as a static, per-item scalar.
- The theory assumes perfect fitting and exchangeable latent scores. Item properties other than popularity are not modelled.
- Debiasing is tuned to "minimise bias while preserving acceptable utility", which is not a fixed operating point.

## Experimental setup
- Datasets (10-core): ML1M (6,040 users / 3,260 items, density 3.55%), Ciao (5,047 / 8,102, 0.21%), Epinions (12,168 / 11,283, 0.18%), Amazon-App (16,201 / 4,869, 0.23%). Random 60/20/20 train/val/test split.
- Models: MF (RMSE) and BPR, both with latent dim 64, lr 0.001, 2 negatives per positive, batch 1024.
- Metrics: NDCG@20/@50, PRU, PRI.

## Key results
- The bias is present everywhere (Table 2). PRU, MF/BPR: ML1M 0.835/0.779, Ciao 0.542/0.591, Epinions 0.684/0.708, App 0.567/0.636. PRI: ML1M 0.980/0.969, Ciao 0.363/0.433, Epinions 0.535/0.573, App 0.609/0.692.
- ML1M MF (Table 3):

  | Model | NDCG@20 | PRU | PRI |
  |---|---|---|---|
  | MF | 0.2726 | 0.8350 | 0.9799 |
  | MF-weight | 0.1484 | 0.4845 | 0.6407 |
  | MF-rescale | 0.1361 | 0.4365 | 0.6936 |
  | MF-Reg | 0.1492 | 0.1910 | 0.5916 |
  | MF-PC | 0.1435 | 0.4552 | 0.5594 |

- On Ciao, Epinions and App, PC preserves utility best and drives PRI to about zero or below. Epinions MF-PC: NDCG@20 0.0605 vs MF 0.0693, PRI −0.0415. App MF-PC: 0.0965 vs 0.1026, PRI −0.0487.
- ML1M BPR (Table 4): BPR-PC NDCG@20 0.2308 vs BPR 0.2983, PRU 0.5712, PRI 0.5080. BPR-Reg has the lowest PRU (0.2862).
- Every debiasing method lowers NDCG. Reg reduces PRU the most, and PC reduces PRI the most at the best utility.

## Reported limitations and trade-offs
- The utility–bias trade-off is explicit in all cases. Reg's utility loss is expected because popularity is continuous and skewed, and improving it is left as future work.
- The theory depends on idealised assumptions.
- Evaluation uses random splits of biased logs. There is no unbiased (random-exposure) test set and no user-group or calibration analysis.

## Relevance to this project
- **§6 Popularity metrics (spec: "PopRSP / PopREO (Zhu et al. WSDM '21)"):** this paper defines **PRU/PRI**, not PopRSP/PopREO. Ranking-based Statistical Parity (RSP) and Equal Opportunity (REO) come from Zhu, Wang, Caverlee, *"Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems"*, SIGIR 2020 (DOI 10.1145/3397271.3401177). The "Pop-" variants with popularity groups are the Elliot framework's implementation, which cites that SIGIR '20 paper:
  - RSP = std_a P(R@k | g_a) / mean_a P(R@k | g_a), where P(R@k | g_a) is the share of a user's un-interacted group-a items that land in the top-k.
  - REO is the same ratio using P(R@k | g_a, y=1), counted over test positives only.
  - Elliot's default groups are the 20% most popular items vs the rest.
  
  The citation and definitions in the spec should be fixed. PRU/PRI are worth adding as continuous, group-free alternatives.
- **§6 Duration metrics:** PRU/PRI generalise directly to duration: replace pop(·) with item duration and get "DRU/DRI". This avoids arbitrary quantile cut-offs for DurRSP/DurREO.
- **§4 evaluation / RQ1:** REO-type metrics depend on what counts as y=1. On KuaiRand's random-exposure log, positives are not exposure-filtered, so the equal-opportunity conditioning is far more meaningful than on random splits of biased logs.
- **§5 Popularity row:** PDA's "BPR-PC" baseline is this paper's Popularity Compensation (PDA tunes α ∈ [0.1, 2.0] and β ∈ [0, 1]). It is a cheap post-hoc popularity knob for the Pareto sweeps if a non-causal re-ranker is wanted.
