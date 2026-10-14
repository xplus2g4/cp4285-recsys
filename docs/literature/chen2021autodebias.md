# AutoDebias: Learning to Debias for Recommendation

- **Authors:** Jiawei Chen, Hande Dong, Yang Qiu, Xiangnan He, Xin Xin, Liang Chen, Guli Lin, Keping Yang
- **Venue / year:** SIGIR 2021
- **Link:** https://arxiv.org/abs/2105.04170 (DOI 10.1145/3404835.3462919)
- **Local PDF:** `docs/literature/papers/chen2021autodebias.pdf`
- **Source basis:** full text (arXiv v5)

## Problem
Existing debiasing methods target one or two specific biases, such as IPS for selection or click models for position. They also need hand-specified configurations (propensities, pseudo-labels, generative assumptions). The authors argue these lack *universality* and *adaptivity*, which makes mixed or unknown biases a problem.

## Method
- **Risk discrepancy view:** all biases make p_T(u,i,r) ≠ p_U(u,i,r).
  - Selection and exposure bias skew p(u,i).
  - Conformity bias skews p(r|u,i).
  - Position bias skews both.
- **General empirical risk (Eq. 7):** L̂_T(f|φ) = (1/|D_T|) Σ_k w⁽¹⁾_k δ(f(u_k,i_k), r_k) + Σ_{u,i} w⁽²⁾_{ui} δ(f(u,i), m_{ui}).
  - The first term reweights observed data.
  - The second term imputes pseudo-labels m on all pairs. This is needed because p_T has zero mass on part of the support of p_U (region S₀, e.g. never-exposed unpopular items), where reweighting alone cannot help.
- **Special cases recovered:** IPS (w⁽²⁾ = 0), imputation, doubly robust, conformity offset, WMF-style negative weighting, Saito's implicit IPS variant and position-bias IPS.
- **AutoDebias:** learns φ = {w⁽¹⁾, w⁽²⁾, m} by bi-level meta-learning on a small **uniform** set D_U.
  - The φ are parameterised by a linear meta-model:
    - w⁽¹⁾ = exp(φ₁ᵀ[x_u ∘ x_i ∘ e_r]);
    - w⁽²⁾ = exp(φ₂ᵀ[x_u ∘ x_i ∘ e_O]);
    - m = σ(φ₃ᵀ[e_r ∘ e_O]).
  - Each iteration: (1) a tentative SGD step θ′(φ) on D_T; (2) backpropagate the loss of θ′ on D_U to φ, updated with Adam; (3) the actual update of θ.
- **Theory:** Lemma 3 bounds generalisation by √(S_w/|D_T|) + √(1/|D_U|) terms, where S_w is the mean square of w⁽¹⁾. Lemma 4: a low-capacity meta-model adds inductive-bias error but shrinks S_w, i.e. variance.

## Assumptions held fixed
- The uniform data are collected by a random logging policy and are treated as unbiased gold evidence of p_U. p_U(u,i) is "often supposed as uniform".
- The base model is MF. Debiasing parameters depend only on user id, item id, label and the observed flag, plus position in the list experiment, through a linear model. No item attributes such as duration are used.
- The loss must satisfy E[δ(·)] = δ(·, E[·]) for imputation (L2, L1, cross-entropy).
- The label definition is fixed: ratings binarised at a threshold of 3 (> 3 is positive). The method optimises whatever the uniform-set loss measures.

## Experimental setup
- **Data:**
  - **Yahoo!R3:** 15,400 users / 1,000 items; train 311,704; uniform 2,700; validation 2,700; test 48,600.
  - **Coat:** 290 / 300; 6,960 / 232 / 232 / 4,176.
  - **Simulation** (500×500) with position and selection bias.
  - The random data are split 5% D_U, 5% validation and 90% test, following KDRec.
  - Implicit variants (Yahoo!R3-Im, Coat-Im) drop negative training feedback.
- **Metrics:** NLL, AUC, NDCG@5.
- **Baselines:** MF trained on biased, uniform or combined data; IPS (NB propensity); DR; CausE; KD-Label. Implicit: WMF, RI-MF, AWMF. List feedback: DLA, HeckE.
- **Hyperparameters:** grid search over learning rate and weight decay ∈ {1e-4, 1e-3, 1e-2, 1e-1} on the validation set.

## Key results
- **Table 3:**

  | Method | Yahoo NLL | Yahoo AUC | Yahoo NDCG@5 | Coat NLL | Coat AUC | Coat NDCG@5 |
  |---|---|---|---|---|---|---|
  | AutoDebias | −0.419 | 0.741 | 0.645 | −0.512 | 0.766 | 0.522 |
  | MF (biased) | −0.587 | 0.727 | 0.550 | −0.539 | 0.747 | 0.500 |
  | IPS | −0.448 | 0.723 | 0.549 | −0.515 | 0.759 | 0.509 |
  | DR | −0.444 | 0.723 | 0.552 | −0.512 | 0.765 | 0.521 |
  | KD-Label | −0.632 | 0.740 | 0.580 | −0.593 | 0.748 | 0.504 |
  | MF (uniform) | | | 0.449 | | | 0.358 |

  On Yahoo the paper reports gains of 5.6% in NLL and 11.2% in NDCG@5. On Coat, DR is almost tied (0.521 versus 0.522).
- **Table 4** (ablation, NDCG@5):
  - Yahoo: w1 = 0.573, w1+m = 0.581, full = 0.645.
  - Coat: full = 0.521 is *lower* than w1+m = 0.526, so adding w⁽²⁾ hurt Coat slightly.
- **Figure 4:** AutoDebias remains strong with only 1% uniform data, while KD-Label falls to MF-biased level.
- **Figure 5:**
  - The learned item weights w⁽¹⁾_i *decrease with item popularity*, with higher variance among popular items.
  - The gains come mainly from unpopular items (the bottom 80% by positive frequency), and unpopular items get more exposure.
- **Tables 5–6:**
  - Learned imputation for missing pairs is −0.913, close to the value for negatives (−0.992 is for positive pairs, as printed; the text says positive > missing > negative). The table and the text disagree, so read with caution.
  - Learned w_r flips between AutoDebias-w1 (r = −1: 3.37, r = 1: 1.56) and the full model (0.29 / 4.77).
- **Table 7:** Yahoo!R3-Im AUC 0.730 versus AWMF 0.675.
- **Table 8:** Simulation NDCG@5 0.707 versus HeckE 0.648.

## Reported limitations and trade-offs
- The bi-level optimisation is alternating and approximate, with no global-optimum guarantee.
- A linear meta-model may not contain the optimal φ (inductive bias). The authors suggest richer meta-models.
- Uniform data are costly because they degrade user experience.
- The framework assumes static biases; dynamic bias is left as future work.
- Exponentiated weights are not explicitly clipped. The variance control comes only from the meta-model's low capacity (Lemma 4).

## Relevance to this project
- **§5 Joint (reference) row:** AutoDebias meta-learns on the KuaiRand calibration split (30% of the random log). In the paper D_U is only 5% of the random data, and validation is a further 5%.
  - Our calibration split must be divided into a D_U part and a hyperparameter-validation part, or else the reported gain is selected on the same data used to meta-learn φ. This is a protocol detail the spec does not state.
- **§7.3 Mass-shift / H3:** AutoDebias implicitly down-weights popular items (Fig. 5a), so it should be audited as a de facto popularity correction. Its off-diagonal duration effect is unknown because the meta-model has no duration features. Adding duration to x_i is a natural extension.
- **§4 Labels:** φ is tuned to the loss on uniform data under one label. Under our label matrix it must be meta-learned per training label, and its "win" on its own label must be checked against the other labels.
- **H2 / §6:** Report max w⁽¹⁾ and ESS for AutoDebias as well, because its weights are unbounded exp(·).
