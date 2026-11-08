# Methods

The three debiasing methods and two baselines compared on KuaiRand-Pure. All of them share one backbone: a 64-dimensional matrix factorisation (MF) with user and item biases. Each knob value is its own sweep; the tuned learning rate × weight decay is chosen on the calibration split (30% of random-log users), and the best point is trained with 5 seeds and scored on the test split (the other 70%). Results live in `assets/data.js` (built by `build_data.py`); this file describes the methods only.

| Method | Paper | Bias targeted | Mechanism | Stage | Knob grid | Baseline |
|---|---|---|---|---|---|---|
| Naive MF | — | — | BCE on clicks | — | — | — |
| VR | Zhan et al., KDD 2022 (its baseline) | — | MSE on raw watch time | — | — | — |
| MACR | Wei et al., KDD 2021 | popularity (item direct effect) | counterfactual inference | training loss + inference score | c ∈ {0, 10, 20, 30, 40, 60, 100} | Naive MF |
| IPS-feat | Schnabel et al., ICML 2016 (covariate propensity); FM: Rendle, ICDM 2010 | exposure | loss reweighting | training loss | clip ∈ {10, 30, 100, 300, 1000, 2000, 4000, 8000, ∞} | Naive MF |
| D2Q | Zhan et al., KDD 2022 | duration | label transformation | training label + inference score | M ∈ {1, 10, 20, 30, 60, 100} | VR |
| D2Q-Q (our variant) | — | duration | label transformation | training label + inference score | same as D2Q | VR |

## Baselines

**Naive MF.** MF trained with BCE on the click label of the standard (policy-chosen) log, ranking by the MF logit. Baseline for MACR and IPS-feat.

**VR (value regression).** MF trained with MSE on raw watch time scaled by its training mean w̄ = 23.6 s (the scale only helps the optimiser): L = Σ (f(u, i) − w ÷ w̄)², ranking by f. This is D2Q's own naive baseline. VR, D2Q and D2Q-Q never see clicks, so they select lr × weight decay on calibration NDCG@10 of raw watch time (raw watch time has no binary relevance, so GAUC is undefined); click-trained methods select on GAUC.

## MACR

**Problem.** A click can come from the user–item match or from the item being popular and widely known (conformity). Only the first is personal. MACR keeps the match and removes the item's direct effect; unlike IPS it does not down-weight popular items during training, so a popular item that matches the user can still rank high.

**Causal graph.** User U, item I, matching K, click Y, with three paths: U → K ← I → Y (the match, score ŷ_k), I → Y (item direct effect, score ŷ_i), U → Y (user direct effect, score ŷ_u).

**Training.** Compute the three branch logits for each logged exposure and fuse them, σ being the sigmoid:

ŷ = ŷ_k · σ(ŷ_i) · σ(ŷ_u)

L = BCE(ŷ, y) + α · BCE(ŷ_i, y) + β · BCE(ŷ_u, y), α = β = 10⁻³ (the paper's values; not swept).

**Inference.** Rank by the total indirect effect, the total effect minus the natural direct effect with the match fixed at a reference value c:

score(u, i) = (ŷ_k − c) · σ(ŷ_i) · σ(ŷ_u)

c = 0 keeps the full prediction (MACR training, no counterfactual subtraction). Two properties: σ(ŷ_u) is one positive factor per user, so it never changes a user's ranking; and what re-orders a list is c × (difference in σ(ŷ_i) between items), so a narrow σ(ŷ_i) range needs a large c.

**Implementation.**
- ŷ_k is the MF backbone logit, unchanged. ŷ_i and ŷ_u are linear layers on the backbone's item and user embeddings (the paper allows MLPs).
- The same adjustment is applied to candidate scoring (GAUC, NDCG) and to full top-k lists; a unit test checks this.
- Departure: the paper's MF has no bias terms, but the shared backbone keeps the per-item bias b_i inside ŷ_k. A per-item bias is itself an item-only effect, so part of the item's direct effect sits inside the matching branch and is never subtracted.
- Grid up to 100 because a calibration-only check found the learned σ(ŷ_i) spans only about 0.98–1.00, while ŷ_k spans about −1.6 to 1.0 between its 1st and 99th percentiles. The re-ordering push c × Δσ(ŷ_i) is therefore about 0.2 at c = 10 and about 2 at c = 100. The grid still contains the paper's tuned range (20–40).
- c = 0 tuned to lr 10⁻³ / wd 10⁻⁶; Naive MF and every c ≥ 10 tuned to lr 10⁻² / wd 10⁻⁵. The c = 0 setting therefore mixes MACR's training change with an optimisation change.

**Family.** Model-agnostic for backbones of the form "user and item representations → matching score". The paper uses MF and LightGCN on Adressa, Globo, ML10M, Yelp and Gowalla. It needs no popularity counts or side features: the item branch learns popularity from clicks.

## IPS-feat

**Problem.** Every training row is an exposure the platform's recommender chose, so a model trained on them learns the user's taste and the old recommender's choices (exposure bias). A popularity propensity corrects this with one number per item, identical for every user. Schnabel et al. also propose estimating the propensity from user and item covariates; IPS-feat is that option.

**Propensity model.** Each (user, item) pair becomes 32 categorical codes: 25 user fields (activity degree, follow / fan / friend / register-day ranges, 18 anonymous one-hot fields, …) and 7 item fields (video type, upload type, upload date, music type, tag, orientation, and the item's duration decile in the catalogue). No user or item ids; near-identifiers (author id, music id) and constant columns are left out; each field is coded 0 = missing, 1..K = sorted raw values.

Every logged training exposure (u, i) is a positive; each gets one negative (u, j) with the same user and j uniform over the 7,344 items, redrawn every epoch. A factorization machine is trained with BCE for 15 epochs (Adam, lr 10⁻², batch 65,536):

s(u, i) = b + Σ_f w_{x_f} + Σ_{f<g} ⟨v_{x_f}, v_{x_g}⟩, with 8-dimensional factors v.

With one negative per positive, the FM's odds estimate a density ratio:

e^{s(u, i)} ≈ P_log(i | u) ÷ P_uniform(i | u) ⇒ ρ(u, i) = e^{−s(u, i)}

Because each negative shares its positive's user, any user-only term cancels; user fields act only through user × item interactions. This is why an FM is used instead of the paper's logistic regression, which would drop the user side entirely under this sampling.

**Weights and loss.** w̃ = ρ ÷ min ρ → w̄ = min(w̃, clip) → w = w̄ ÷ mean(w̄); the MF backbone is trained with L = (1/B) Σ w_r · BCE(f(u_r, i_r), y_r) on clicks. Inference is the plain MF score, as in Naive MF; the FM is not used.

**Knob.** clip is the largest allowed ratio between two weights: clip = 1 would be Naive MF, clip = ∞ the full inverse propensity (unbiased if the propensity is right, highest variance). The first six values {10 … 1000} turned out uninformative because the smallest FM weight is an outlier (relative to it, the 1st percentile of w̃ is 68 and the median 386), so clip ≤ 300 clips most rows. 2000, 4000 and 8000 were added afterwards to trace the informative range, so the grid is partly post hoc. The amount of clipping depends on the propensity's tail, so the effective sample size share ESS = (Σw)² ÷ (N·Σw²) measures correction strength better than clip.

**Departures from the paper.**
- FM instead of logistic regression (see above).
- No random-exposure data in the propensity: the FM separates logged exposures from uniform random items drawn by us, so its target is uniform item exposure per user, conditioned on features.
- Implicit clicks with BCE and Adam, not rating MSE/MAE with L-BFGS.
- Clipping, which the paper names but does not evaluate, is the knob; clip is relative to the smallest weight.
- The FM is re-fitted inside each run with the run's seed, so seed variance mixes MF and propensity noise.

**Diagnostics logged per run.** Propensity AUC (FM logit, logged rows vs fresh uniform negatives; a fit diagnostic on training positives, not a held-out check), Spearman ρ with the popularity inverse propensity, ESS share, share of rows clipped and the largest weight after normalisation. The tuned lr / wd equals Naive MF's at every clip.

**Family.** Model-agnostic for any backbone with a pointwise loss; needs categorical side features, not popularity counts. Pairwise losses such as BPR would need a per-pair weight; sequential or LLM recommenders would need a context-aware propensity.

## D2Q and D2Q-Q

**Problem.** Watch time grows with video length, so a watch-time regressor learns "long video = high score". The paper's causal graph over user U, video V, duration D and watch time W keeps D → W as legitimate and treats D → V (a watch-time-maximising platform shows long videos more) as the bias; duration confounds the video–watch-time relation.

**Training.**
1. Sort training rows by duration and cut them into M groups with equal row counts. Each cut is an actual duration, so rows of one video share a group.
2. Replace watch time by its within-group empirical quantile: y_i = Φ̂_k(w_i) = #{rows j in group k with w_j ≤ w_i} ÷ n_k ∈ (0, 1].
3. Fit one MF on all groups: h(u, i) = σ(f(u, i)) with soft-target BCE. The paper uses MSE; both have the expected quantile as minimiser.

The justification is backdoor adjustment, E[W | do(U, V)] = Σ_d P(D = d) · E[W | U, V, D = d]: fitting within duration groups approximates the inner term, and quantile labels let one model share parameters across groups.

**Inference.**
- D2Q (published): map the predicted quantile back to seconds with the item's group inverse CDF, ŵ(u, i) = Φ̂_{k(i)}⁻¹(h(u, i)), and rank by ŵ. Long-video groups have larger inverse-CDF values, so the legitimate D → W effect is kept.
- D2Q-Q (our variant): rank by h(u, i) itself, a duration-normalised score ("how well will this user watch this video compared with videos of the same length"). The paper only ranks by Φ̂⁻¹(h); D2Q-Q is reported because it is the duration-normalised label the methodology asks for. It is not shown on the poster.

**Knob M.** M = 1 is one group: a global rank transform with no duration adjustment, so D2Q and D2Q-Q rank identically (it still differs from VR, because a rank target damps a few very long watches). Large M compares each watch only with watches of almost equally long videos, at the cost of noisier CDFs; with 1,117,036 training rows, M = 100 still leaves about 11,000 rows per group. The grid is the paper's {1, 10, 20, 30, 50, 100} with 50 replaced by 60, the setting D2Co and CWM used on KuaiRand-Pure. Each M is run with both rankings (12 sweeps plus VR).

**Departures from the paper.**
- MF backbone instead of a feature-based deep network, and no duration input feature; the item id fixes the item's group, so the item bias can absorb group-level differences. Res-D2Q is not implemented.
- Soft-target BCE instead of MSE.
- Each group's inverse CDF is stored at 1,001 quantile levels and interpolated linearly; items outside the training duration range take the first or last group.
- Selection on calibration NDCG@10 of raw watch time, as for VR.

**Diagnostics.** `d2q_groups`, the number of groups actually formed, equals M in every run (no durations tied at a cut). M = 1 keeps VR's lr 10⁻³ / wd 10⁻⁶ while most M ≥ 10 sweeps tune to lr 10⁻² / wd 10⁻⁵, so the M = 1 → 10 jump mixes the grouping with an optimisation change.

**Family.** Model-agnostic: the paper states "one can apply any off-the-shelf regression model" for h. It needs each row's and candidate's duration, a model with a [0, 1] output trained on a graded target, and the stored inverse CDFs. The paper used Kuaishou production logs (about 1.2 billion rows) with a feature-based deep network; later KuaiRand studies (D2Co, CWM) used D2Q with 60 groups on FM, DeepFM, AutoInt and DCN.
