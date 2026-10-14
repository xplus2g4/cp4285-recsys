# Disentangling User Interest and Conformity for Recommendation with Causal Embedding

- **Authors:** Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Depeng Jin, Yong Li (Tsinghua University; HKU; USTC)
- **Venue / year:** The Web Conference (WWW '21), Ljubljana, 2021
- **Link:** https://arxiv.org/abs/2006.11011 (DOI: https://doi.org/10.1145/3442381.3449788)
- **Local PDF:** `docs/literature/papers/zheng2021dice.pdf`
- **Source basis:** full text (arXiv v2)

## Problem
Observed clicks mix users' *interest* with *conformity* toward popular items. Existing popularity debiasing (IPS reweighting, CausE with unbiased data) treats popularity as a global, item-side scalar, ignoring that conformity varies by user and item, and bundles both causes into one representation, which the authors argue hurts robustness when the popularity distribution shifts between training and serving (non-IID).

## Method
- **Additive causal model:** S_ui = S_ui^interest + S_ui^conformity, with click as a *collider* of the two causes. Each user and item gets two embeddings (interest, conformity); scores are inner products, summed (equivalently, concatenated embeddings).
- **Cause-specific data via the colliding effect** (item popularity as the conformity proxy), from BPR triples (u, positive i, negative j):
  - O₁ — negative less popular than positive: only conformity and click inequalities hold (M^C_ui > M^C_uj; total score ordering).
  - O₂ — negative more popular than positive: the click is attributed mainly to interest, giving M^I_ui > M^I_uj, M^C_ui < M^C_uj, plus total ordering.
- **Losses:** L = L_click(O₁+O₂) + α(L_interest(O₂) + L_conformity(O₁+O₂)) + β·L_discrepancy, where discrepancy is −L1, −L2 or distance correlation (dCor) between interest and conformity embedding sets.
- **PNSM** (popularity-based negative sampling with margin): negatives drawn from items with popularity > p + m_up or < p − m_down.
- **Curriculum:** margins and α decay by 0.9 per epoch (easy → hard).
- **Variants used for analysis:** DICE-int (interest embeddings only) and DICE-con (conformity only). The main DICE ranking uses **both** (concatenated).
- Hyper-parameters: α = 0.1, β = 0.01; embedding size 64 per cause (128 total, matching 128-d IPS baselines); BPR; Adam. Backbones BPR-MF and LightGCN. Code: https://github.com/tsinghua-fib-lab/DICE

## Assumptions held fixed
- Item popularity (interaction count) is the proxy for conformity; no other bias (exposure policy, position, duration) is modelled. Exposure is implicitly taken as given.
- Click = additive combination of exactly two independent causes; the O₂ inequalities hold only "with high probability" when the popularity gap is large (hence PNSM).
- Positives are binarised 5-star ratings; the method is designed for implicit, single-label feedback.
- The evaluation's "intervention" is simulated by resampling logged ratings, not by a randomised exposure policy.

## Experimental setup
- **Datasets** (Table 1): MovieLens-10M (37,962 users, 4,819 items, 1,371,473 interactions; item-interaction entropy train 6.22 / test 7.97) and Netflix Prize (32,450 / 8,432 / 2,212,690; 6.85 / 8.54).
- **Intervened split:** 40% of records sampled with probability ∝ inverse item popularity (capped at 0.9) form intervened data; split stated as 70/10/20 — training = 60% normal + intervened portion, validation 10% intervened, test 20% intervened. (§4.5 describes the training intervened share as 10%; §4.1 says "60% normal and 20% intervened" — the paper is internally inconsistent here.)
- **Baselines:** None, IPS, IPS-C (capping), IPS-CN (+normalisation), IPS-CNSR (+smoothing/renormalisation), CausE; scalar-bias variants BIAS-U/-I/-UI.
- **Metrics:** Recall, HR, NDCG @20, @50; IOU with ItemPop's top-K; t-SNE of embeddings.

## Key results
- Table 2, MF backbone, MovieLens-10M @20 Recall/HR/NDCG: None 0.1286 / 0.4429 / 0.0846; IPS 0.1335 / 0.4434 / 0.0852; IPS-CN 0.1412 / 0.4700 / 0.0925; CausE 0.1157 / 0.4066 / 0.0744; **DICE 0.1634 / 0.5197 / 0.1084**. Netflix MF @20: None 0.1122 / 0.5194 / 0.0943; DICE 0.1258 / 0.5545 / 0.1070. LightGCN MovieLens @20: None 0.1378 → DICE 0.1812 Recall.
- IPS variants are unstable: e.g. IPS on Netflix-MF is below None (Recall@20 0.1058 vs. 0.1122); IPS-CN with LightGCN drops to 0.1119 on MovieLens.
- Embeddings beat scalar user/item bias terms (Fig. 4).
- Fig. 5a: DICE-con top-K overlaps ItemPop with IOU > 0.5 for K > 40; DICE-int has almost no overlap; IPS and CausE < 0.2.
- PNSM vs. random negatives (Table 3, MovieLens @20): Recall 0.1634 vs. 0.1274; NDCG 0.1084 vs. 0.0843.
- DICE beats IPS-CNSR at intervention caps 0.5/0.7/0.9 (Fig. 6) and at 0–20% intervened training data, including 0% (Fig. 8b).
- dCor discrepancy ≈ 2% better than L1/L2-inv but ~100 s vs. ~44 s per epoch.

## Reported limitations and trade-offs
- dCor is computationally expensive; L1/L2-inv recommended for scale.
- Removing the conformity-modelling task barely changes accuracy but degrades embedding structure (Fig. 7).
- Without curriculum, performance drops at large α.
- Intervened training data improves all methods; obtaining it is "too expensive" in practice (random policy hurts user experience).
- Only accuracy on the intervened test set is reported; no popularity-distribution metrics (ARP, Gini, coverage) or user-group breakdown of the final DICE lists.

## Relevance to this project
- **§5 optional extension:** DICE's own final ranking *keeps* conformity (concatenated embeddings); the "remove the conformity embedding" test in the spec corresponds to the paper's **DICE-int** variant. The spec should name DICE-int explicitly and also report full DICE.
- **§5 Popularity row / H3:** Fig. 5a (IOU with ItemPop) is a ready-made diagnostic; on KuaiRand, compute ARP/Gini/APLT for DICE vs. DICE-int vs. naive to see where popularity mass moves, and NDCG gaps by mainstreaminess group.
- **§2 / §10:** DICE's "intervened" test set is inverse-popularity resampling of logged data, not real randomised exposure; KuaiRand's `is_rand` log is a stronger test — a point for the related-work discussion.
- **§5 Exposure row / H2:** Table 2 shows IPS/IPS-C/IPS-CN/IPS-CNSR swinging above and below the naive baseline across datasets/backbones — prior evidence for instability of IPS-family methods.
- **RQ3:** DICE's conformity proxy is pure interaction count; if KuaiRand popularity correlates with duration (§3), DICE's popularity-margin negative sampling will also shift duration mix — measurable with the §7.3 heatmap. Not evaluated on KuaiRand.
