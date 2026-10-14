# Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System

- **Authors:** Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi, Xiangnan He
- **Venue / year:** KDD 2021 (27th ACM SIGKDD Conference)
- **Link:** https://arxiv.org/abs/2010.15363 (DOI https://doi.org/10.1145/3447548.3467289)
- **Local PDF:** `docs/literature/papers/wei2021macr.pdf` (arXiv v2, 7 Jun 2021)
- **Source basis:** full text (main body; appendices on α/β sensitivity not read in detail)

## Problem
Models trained to reconstruct long-tailed interaction data recommend popular items more often than the (non-IID) test data warrants (Fig. 1, Adressa, MF and LightGCN). IPW-style fixes are sensitive to the weighting scheme and down-weight popular items regardless of whether they match the user.

## Method
- Causal graph (Fig. 2c): U, I → K (matching) → Y, plus direct edges I → Y (item popularity) and U → Y (user conformity/"easiness").
- Training: three branches combined multiplicatively, ŷ_ui = ŷ_k · σ(ŷ_i) · σ(ŷ_u). Here ŷ_k is any backbone (MF, LightGCN), and ŷ_i = Y_i(I), ŷ_u = Y_u(U) are small modules (MLPs) on the item and user embeddings. The multi-task loss is L = L_O + α·L_I + β·L_U, with all terms BCE against the interaction labels.
- Counterfactual inference: rank by TIE = TE − NDE, which gives **ŷ_k·σ(ŷ_i)·σ(ŷ_u) − c·σ(ŷ_i)·σ(ŷ_u)** (Eq. 9). c is the reference value of ŷ_k when K takes its reference value (mean embeddings). This is a hyper-parameter. Larger c removes more of the item's direct effect. The size of the adjustment is user-specific through σ(ŷ_u), so "easy" (active) users get larger adjustments.

## Assumptions held fixed
- Popularity bias is the **direct effect I → Y**, and all of it is harmful. Unlike PDA/TIDE, MACR keeps no "benign" popularity at inference.
- Interaction frequency is the only bias considered. §3.5 explicitly says the paper ignores other causes of clicks (category, quality, exposure mechanism) and leaves side-information nodes to future work. Duration, position and exposure policy are therefore held fixed and unmodelled.
- The effects combine multiplicatively and the reference status is a single scalar c shared by all users and items. Matching at the reference (mean) embeddings is assumed to stand in for "no matching".
- The user branch assumes that conformity tracks activity level (empirically σ(ŷ_u) rises with user interaction count, Fig. 10).
- Evaluation assumes that an item-uniform resample of logged interactions is a valid "debiased" test set.

## Experimental setup
- Datasets: Adressa (13,485 users / 744 items / 116,321 interactions), Globo (158,323 / 12,005 / 2,520,171), ML10M binarised (69,166 / 8,790 / 5,000,415), Yelp (31,668 / 38,048 / 1,561,406), Gowalla (29,858 / 40,981 / 1,027,370).
- **Debiased evaluation:** 10% of interactions are sampled with equal probability per item as the test set, another 10% as validation, and the rest is training data. This is *not* random-exposure data: it is a resample of the biased log.
- Metrics: HR, Recall, NDCG @20, all-ranking. Item-group recommendation frequency and per-item recall by popularity group.
- Baselines (on MF and LightGCN): ExpoMF, CausE, BS (bias score removed at test), Reg (Abdollahpouri et al. 2017 regulariser), IPW, DICE.

## Key results
- MACR wins on every dataset and metric (Table 2, K=20). Average HR@20 gain over the backbone: 153.13% (MF) and 241.98% (LightGCN). Adressa MACR_MF: HR 0.140 / Rec 0.109 / NDCG 0.050, against IPW_MF 0.128 / 0.096 / 0.039 and DICE_MF 0.133 / 0.098 / 0.041. The largest gains are on Globo (MACR_LightGCN HR 0.132 vs LightGCN 0.017), which the authors attribute to its very skewed popularity.
- LightGCN amplifies popularity more than MF, so it does worse than MF under debiased evaluation. With MACR it does better.
- c sweep (Fig. 7, Adressa): MACR_LightGCN improves as c goes from 0 to 29 and then degrades. LightGCN is more sensitive to c than MF.
- Ablation (Adressa, MACR_MF): removing the item branch drops HR@20 0.140 → 0.116, removing the user branch → 0.137, removing L_I → 0.124, removing L_U → 0.138. The item effect dominates.
- MACR lowers how often popular items are recommended but **raises recall for the most popular group the most**, i.e. popular items go to better-matched users. Gains for unpopular items are small because their embeddings are poorly learned.

## Reported limitations and trade-offs
- The authors do not model other causes (category, quality, exposure). The extension is left as future work.
- Unpopular items still improve only a little because of sparse training signal.
- c must be tuned and the optimum differs by backbone. Too large a c hurts accuracy.
- The test set is item-uniform resampled from biased logs, so "debiased" accuracy depends on this simulation. CausE suffers from the small debiased training split.

## Relevance to this project
- **§5 Popularity row:** the second popularity method, with knob c (spec: "MACR c"). The paper's sweep (0 → ~29 then down on Adressa) gives a starting grid. α and β are secondary knobs.
- **§4/§2 evaluation and RQ1:** MACR's "unbiased" test is an item-uniform resample. KuaiRand's random-exposure log is a stronger test, so this project can check whether MACR's large gains survive under genuine randomisation.
- **H1/RQ3:** MACR removes the *entire* direct item effect. On KuaiRand that effect would also absorb duration-driven click/watch propensity (duration is an item property). So MACR may shift mass along duration as well as popularity, which is a key candidate for the mass-shift plot in §7.
- **§6 User-side / H3:** because the adjustment scales with σ(ŷ_u), which tracks user activity, MACR treats active users more strongly. Group gaps by `user_active_degree` are therefore expected and should be measured.
