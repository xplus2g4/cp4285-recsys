# The Self-Normalized Estimator for Counterfactual Learning

- **Authors:** Adith Swaminathan, Thorsten Joachims (Cornell)
- **Venue / year:** NeurIPS (then NIPS) 2015
- **Link:** https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html
- **Local PDF:** `docs/literature/papers/swaminathan2015snips.pdf`
- **Source basis:** full text (NeurIPS proceedings PDF)

## Problem
This paper is about batch learning from logged bandit feedback (BLBF). The loss is observed only for the action the logging policy h₀ took, with logged propensity p_i = h₀(y_i|x_i). The Counterfactual Risk Minimization (CRM/POEM) recipe minimises the IPS risk estimate R̂(h) = (1/n) Σ δ_i h(y_i|x_i)/p_i, but this estimator has unbounded variance. The authors identify a new failure mode, **propensity overfitting**:
- ERM can lower R̂ by manipulating the sum of weights Ŝ(h) = (1/n) Σ h(y_i|x_i)/p_i rather than by finding low-loss actions.
- For positive losses, it selects hypotheses that avoid the logged actions. For negative losses, it selects hypotheses that over-represent them.
- The IPS estimator is not equivariant: translating the loss by a constant changes which hypothesis is learned.

## Method
- **Clipping baseline:** R̂^M(h) uses min(M, h/p). Smaller M lowers variance but adds bias.
- **CRM:** minimises R̂^M plus λ·√(V̂ar/n), an empirical-Bernstein variance regulariser (Theorem 1).
- **Diagnostic:** E[Ŝ(h)] = 1 for every h (Eq. 5), so a large deviation of Ŝ(h) from 1 flags propensity overfitting.
- **Self-normalised estimator** (Eq. 7):
  - R̂^SN(h) = Σ δ_i w_i / Σ w_i, with w_i = h(y_i|x_i)/p_i. Ŝ acts as a multiplicative control variate.
  - It is a convex combination of the observed losses, so it is bounded within the loss range and equivariant to loss translation.
  - It is biased in finite samples but strongly consistent (Theorem 2, which requires full support of h₀).
- **Norm-POEM:** CRM with R̂^SN and a delta-method variance estimate (Eq. 8). It still keeps clipping M and variance-regularisation strength λ as hyperparameters, and is optimised with L-BFGS.

## Assumptions held fixed
- The logging policy h₀ is stationary, has full support, and its propensities p_i are known and logged. There is no propensity estimation and no misspecification analysis.
- Contexts are i.i.d., and the loss is a fixed function of (x, y).
- Only the *action-selection distribution* is corrected. The reward/loss definition and the hypothesis class (exponential-family linear rules) are fixed.

## Experimental setup
- Supervised → bandit conversion on four LibSVM multi-label datasets: Scene, Yeast, TMC and LYRL.
- h₀ is a CRF trained on 5% of the data. Training data are replayed 4 times, sampling labels from h₀, and the loss is Hamming loss.
- **Compared methods:** h₀, POEM (IPS-based CRM), Norm-POEM, and a full-information CRF as a "skyline".
- **Protocol:** 10 runs. Hyperparameters (λ, M) are validated counterfactually on a 25% hold-out.

## Key results
- **Table 1** (test Hamming loss; lower is better). Norm-POEM is significantly better than POEM on all four datasets.

  | Dataset | h₀ | POEM | Norm-POEM | CRF |
  |---|---|---|---|---|
  | Scene | 1.511 | 1.200 | 1.045 | 0.657 |
  | Yeast | 5.577 | 4.520 | 3.876 | 2.830 |
  | TMC | 3.442 | 2.152 | 2.072 | 1.187 |
  | LYRL | 1.459 | 0.914 | 0.799 | 0.222 |

- **Table 2:** Ŝ(ĥ) for POEM is far from 1 and depends on loss translation. For example, on TMC it is 0.000 when δ > 0 and 2.802 when δ < 0. With δ > 0, POEM's TMC loss is 17.305, worse than random. Norm-POEM keeps Ŝ near 1 (0.94–0.98 on most datasets, 0.84/0.82 on Yeast) and is unaffected by translation.
- **Table 3:** Variance regularisation still helps on top of self-normalisation. For example, on TMC Norm-IPS scores 3.609 versus Norm-POEM 2.072.
- **Table 4:** Norm-POEM converges faster, e.g. Scene 7.28 s versus POEM 78.69 s.

## Reported limitations and trade-offs
- SNIPS is biased in finite samples. Its advantage is greatest when fluctuations in the weights dominate the loss variance.
- The variance objective (Eq. 8) does not admit stochastic optimisation.
- Clipping M and variance regularisation λ remain necessary hyperparameters, so self-normalisation does not replace them.
- The experiments use only simulated bandit feedback with *known* propensities.

## Relevance to this project
- **§5 Exposure row:** This is the canonical SNIPS reference. For implementation, the training loss is Σ w·ℓ / Σ w within the batch or globally. Clipping is a separate, complementary knob, which supports sweeping the clip threshold for both IPS and SNIPS.
- **§6 Exposure diagnostics:** Ŝ = mean(1/P̂) over observed pairs should be about 1 (about UI/|O| in the recommender form). This gives a cheap propensity sanity check on the calibration split, alongside max weight and ESS.
- **H2:** The paper describes variance blow-up and propensity overfitting under high-capacity models. This predicts the seed variance that H2 expects for IPS. SNIPS should reduce it, so IPS versus SNIPS is a direct test of H2's mechanism.
- **§8:** SNIPS bias is O(1/n), which matters on small cells (tail and duration buckets) of the random-log test split.
