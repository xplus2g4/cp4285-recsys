# Popularity Bias Is Not Always Evil: Disentangling Benign and Harmful Bias for Recommendation

- **Authors:** Zihao Zhao, Jiawei Chen, Sheng Zhou, Xiangnan He, Xuezhi Cao, Fuzheng Zhang, Wei Wu
- **Venue / year:** IEEE Transactions on Knowledge and Data Engineering (TKDE), 2022 (DOI 10.1109/TKDE.2022.3218994; arXiv preprint Sep 2021)
- **Link:** https://arxiv.org/abs/2109.07946 (DOI https://doi.org/10.1109/TKDE.2022.3218994)
- **Local PDF:** `docs/literature/papers/zhao2022popularity.pdf` (arXiv v1 preprint)
- **Source basis:** full text (arXiv v1); venue/DOI from IEEE Xplore / Semantic Scholar listing via web search

## Problem
Popularity-debiasing methods try to remove popularity entirely. The authors argue popularity mixes two causes: **item quality** (benign, should be kept) and **conformity** (harmful herd behaviour, should be removed). PDA re-injects *all* predicted popularity, including the conformity part. Without an explicit signal separating the two, the paper uses time: quality is static, conformity depends on recent clicks.

## Method
TIDE (Time-aware DisEntangled framework). Causal graph with three paths to the prediction Y:
- I → Q → Y: a static per-item quality scalar q_i.
- (I, t) → C → Y: time-aware conformity c_i^t = β_i Σ_{past clicks l on i} exp(−|t − t_l| / τ), i.e., time-decayed count of earlier clicks, scaled per item by β_i and controlled by temperature τ.
- (U, I) → M → Y: matching score m_ui from any backbone (MF used).
- Score ŷ_ui^t = tanh(q_i + c_i^t) · Softplus(m_ui), trained with BPR and uniform negative sampling.
- **Inference (TIDE-int):** intervene on conformity, set c_i^t to a constant c (c = 0 in experiments): ŷ_ui = tanh(q_i + c) · Softplus(m_ui). TIDE-full keeps conformity at inference.
- Section 3.3 frames PDA as ŷ = P_i · ELU(m_ui) (one popularity path) and DICE/IPS as cutting the whole popularity path; TIDE keeps the quality part and cuts only conformity.

## Assumptions held fixed
- **Hypothesis 1 (stated):** popularity comes mainly from conformity plus item quality; quality is stable and static, conformity is time-sensitive and driven by recent clicks.
- Conformity is item-level (β_i), with no per-user susceptibility. The time-decay form is exponential with one global τ.
- Item quality is one scalar per item, constant over the whole training window.
- Quality and conformity act multiplicatively with matching, through a tanh-bounded factor.
- Explicit ratings stand in for "true preference" when evaluating, and high ratings (e.g., 5) are taken as positives.
- Other item attributes (content length/duration, position, exposure policy) are absent. Any item property correlated with both popularity and satisfaction gets absorbed into q_i.

## Experimental setup
- Motivation analysis: Amazon (many categories), Ciao, Douban, MovieLens. Items fall into 30 popularity groups, and the Pearson correlation between average rating and popularity is computed. Each item also gets a correlation between its rating and its "instant popularity" (clicks in the past half year).
- Main datasets: Douban-Movie (48,799 users / 26,813 items / 7,409,868 interactions, 2010.1–2017.3, 10-core), Amazon-CDs (75,258 / 64,443 / 1,097,592, 5-core), Ciao (5,868 / 10,724 / 143,217, 5-core).
- Split: follows PDA. Each dataset is cut into 10 equal time parts; the first 9 are used for training, and the last is split by user, half validation and half test. Ratings are binarised to clicks.
- Two tasks: (1) **click prediction**, ranking all non-interacted items with CP-Rec/Pre/NDCG@20; (2) **preference prediction**, ranking only the rated test items with high ratings as positives, using PP-Rec/Pre@3.
- Baselines: MF (BPR), MF-IPS (popularity propensity with max-capping), DICE, PD, PDA. Ablations: TIDE-noq, TIDE-noc, TIDE-e (matching only), TIDE-fixq.

## Key results
- Observation 1: average rating correlates positively with popularity on a large share of datasets, but weakly or negatively on a considerable share. Observation 2: for many items, rating correlates negatively with instant popularity.
- Click prediction (Table 2): TIDE-full beats PDA, e.g., Douban CP-NDCG@20 0.0648 vs 0.0618. The biggest gain is on Amazon-CDs (+23.43% / +16.45% / +19.69% in the reported columns). On Ciao NDCG, TIDE-full is 6.13% below the best baseline.
- Preference prediction (Table 3): **PDA, which beats PD on click prediction, is worse than PD on preference prediction** (Douban PP-Pre@3 0.4216 vs PD 0.4254 and MF 0.4422; Ciao PP-Rec@3 0.2429 vs PD 0.2727). TIDE-full is worse than TIDE-int (Douban PP-Pre@3 0.3915 vs 0.4648). TIDE-int is best in most columns.
- The learned q_i ranks items closer to average rating than raw popularity does (Kendall τ with AR: q 0.1268 / 0.1038 / 0.0808 vs popularity 0.0384 / −0.0648 / −0.0336 on Douban / Amazon-CDs / Ciao), though all correlations are small.
- TIDE-int beats TIDE-fixq on Douban and Amazon-CDs and roughly ties on Ciao, where conformity seems to dominate.

## Reported limitations and trade-offs
- Accuracy gains on click prediction partly come from reproducing popularity, and this hurts preference accuracy. The paper's own evidence shows that injecting popularity (PDA, TIDE-full) helps click metrics on biased, time-split test data but hurts satisfaction-like metrics.
- The authors call the low absolute rank correlations a sign of how hard the problem is.
- No fairness, user-group, calibration, or exposure-distribution metrics are reported. Disentanglement relies only on time dynamics, with no external signal.

## Relevance to this project
- **§5 Popularity row / RQ1–RQ2:** this is the strongest published evidence that PDA's apparent gains depend on the evaluation target. It supports the project's choice to evaluate on KuaiRand's random-exposure log instead of a time-split, biased test.
- **H1 / RQ3 (duration ↔ popularity):** TIDE's static q_i would absorb any persistent item attribute that correlates with popularity, including video duration on KuaiRand. This is a concrete mechanism by which popularity correction can move mass along duration.
- **§6 metrics:** the click-vs-preference split maps onto KuaiRand's click vs. watch-ratio/long-view labels. It suggests reporting PDA under both an exposure-based label and a satisfaction label.
- **Optional extension (§5):** TIDE is a natural third popularity method (knob: inference conformity value c, or τ), next to PDA and MACR.
