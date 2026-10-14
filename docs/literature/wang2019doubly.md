# Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random

- **Authors:** Xiaojie Wang, Rui Zhang, Yu Sun, Jianzhong Qi (University of Melbourne; Twitter)
- **Venue / year:** ICML 2019 (PMLR 97)
- **Link:** https://proceedings.mlr.press/v97/wang19n.html
- **Local PDF:** `docs/literature/papers/wang2019doubly.pdf`
- **Source basis:** full text (PMLR PDF)

## Problem
Ratings are MNAR. Two existing estimators both have weaknesses:
- **Error-imputation-based (EIB) estimators** (Steck) are biased when the imputed errors are inaccurate.
- **IPS** (Schnabel et al. 2016) is unbiased only with accurate propensities and suffers from high variance, because inverse propensities "can cause training losses to oscillate".

A linear mix of the two still loses bias when imputation degrades.

## Method
- **Doubly robust (DR) estimator:** E_DR = (1/|D|) Σ_{u,i∈D} [ê_{u,i} + o_{u,i}(e_{u,i} − ê_{u,i})/p̂_{u,i}].
  - It imputes errors for *all* pairs and corrects the error deviation δ = e − ê on observed pairs with IPS.
  - Bias (Lemma 3.1) = (1/|D|) Σ Δ_{u,i}·δ_{u,i}, where Δ = (p̂ − p)/p̂. It is zero if either the propensities or the imputed errors are accurate (Cor. 3.1).
  - Tail bound (Lemma 3.2): it scales with Σ(δ/p̂)², not Σ(e/p̂)². If 0 ≤ ê ≤ 2e, then DR has a lower tail bound than IPS for any propensities (Cor. 3.2).
- **Generalisation bound (Thm 4.1):** both the bias and the variance terms grow with |δ|, so imputation must be accurate.
- **Joint learning (MF-DR-JL):**
  - Prediction model f_θ is trained on the DR loss over all pairs, with ê = (f − g·stopgrad(f))².
  - Imputation model g_φ is trained on observed pairs with loss Σ (ê − e)²/p̂, where e = r − f is a *signed* difference.
  - Training alternates between the two (Alg. 1). Unobserved pairs are down-sampled.
  - MF-JL is the same scheme with p̂ = 1.

## Assumptions held fixed
- Positive propensities, and independent Bernoulli observation indicators.
- Prediction and imputed errors are treated as constants with respect to O when taking expectations.
- Propensities are fixed inputs learned beforehand by Schnabel's NB or LR. DR does not re-estimate them.
- The target is explicit-rating MSE/MAE; ranking is not optimised.
- Only selection on the rating value (and on features, via LR) is corrected. Item popularity and duration are not modelled separately.

## Experimental setup
- **Datasets:**
  - **COAT:** 4,640 MAR + 6,960 MNAR ratings, 290×300.
  - **YAHOO:** 54,000 MAR + 311,704 MNAR ratings, 15,400×1,000. The MAR test set is used for both.
  - AMAZON (1,000,086 ratings) and MOVIE (10,000,054 ratings) are MNAR-only. They use a 90/10 random split and also report SNIPS-weighted MSE.
- **Propensities:** LR on COAT features. Elsewhere, NB using 5% of the test ratings.
- **Hyperparameters:** 5-fold CV.
- **Baselines:** MF, PMF, AutoRec, Gaussian-VAE, CPT-v, MF-HI, MF-MNAR (HL), MF-IPS. FM- and NFM-based variants are also tested.
- **Semi-synthetic Yahoo estimator study:** uses the five Schnabel predictors (ONE, FOUR, ROT, SKEW, CRS). Propensity is p for rating 5 and pα^{min(4,6−r)} otherwise, with α = 0.5, 5% sparsity and 50 samples of O.

## Key results
- **Table 2** (MAR test):

  | Method | COAT MAE | COAT MSE | YAHOO MAE | YAHOO MSE |
  |---|---|---|---|---|
  | MF-DR-JL | 0.778 | 0.990 | 0.747 | 0.966 |
  | MF-IPS | 0.860 | 1.093 | 0.810 | 0.989 |
  | MF-JL | 0.866 | 1.136 | 0.899 | 1.256 |
  | MF | 0.920 | 1.257 | 1.154 | 1.891 |

  Removing either the propensities (MF-JL) or the imputation (MF-IPS) hurts.
- **Table 4:** NFM-DR-JL reaches COAT MAE 0.756 and YAHOO MAE 0.736.
- **Table 3** (MNAR-only datasets): debiasing methods show no clear advantage over biased ones when tested on MNAR data.
- **Figure 4** (propensity variance sweep):
  - Propensities are rescaled so that the minimum ρ ∈ {10⁻², …, 10⁻⁴}.
  - MF-DR-JL has lower MSE and needs fewer epochs than MF-IPS at every ρ, and degrades more slowly as the variance grows.
- **Table 5** (bias % / standard deviation % of true MSE): DR has the lowest bias and standard deviation on all five predictors. For example:
  - ONE: DR 9.9/0.9 versus SNIPS 20.7/1.8.
  - FOUR: DR 24.1/0.6 versus IPS 66.8/1.8.
  - NCIS, i.e. propensities clipped to (0, 1/40], was the worst. For ONE it scored 26.0.
- **Figure 3a:** under imputation learning, more than 99% of pairs satisfy 0 ≤ ê ≤ 2e.
- **Figure 5:** DR stays consistently less biased than EIB as imputation degrades, and degrades more gradually than SNIPS as propensities are corrupted.

## Reported limitations and trade-offs
- Theorem 4.1 assumes fixed imputed errors, so the guarantees cover only the inner loop.
- The imputation model can hurt if it is inaccurate. Heuristic imputation (MF-HI) was worse than plain MF.
- Training over all |D| pairs is costly and requires sampling of unobserved pairs.
- The paper studies rating prediction only; extending to ranking is left as future work.

## Relevance to this project
- **§5 Exposure row:** This is the named DR method. Implement it as MF-DR-JL: an imputation head trained with a 1/p̂-weighted squared error, alternating with the predictor. Reuse the same p̂ (popularity and feature-based) as IPS and SNIPS so that only the estimator varies.
- **H2:**
  - The paper explicitly sweeps propensity variance (min-ρ rescaling, Fig. 4). This is the analogue of our clip-threshold knob, and DR is predicted to flatten the variance curve.
  - The audit should check whether DR keeps IPS's tail-coverage gain while lowering seed standard deviation.
- **§6 Exposure metrics:** log the share of pairs satisfying 0 ≤ ê ≤ 2e as a diagnostic for when DR's variance advantage holds.
- **§10 Propensity misspecification:** Double robustness speaks directly to this threat, but only if the imputation model is accurate on the random-pool items.
- **Caution for §6:** The results are MSE on ratings. On KuaiRand's binary or ratio labels and ranking metrics (NDCG, GAUC), the gains are unverified.
