# How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility

- **Authors:** Allison J.B. Chaney, Brandon M. Stewart, Barbara E. Engelhardt (Princeton University)
- **Venue / year:** RecSys '18 (12th ACM Conference on Recommender Systems), Vancouver, 2018
- **Link:** https://arxiv.org/abs/1710.11214 (DOI https://doi.org/10.1145/3240323.3240370)
- **Local PDF:** `docs/literature/papers/chaney2018confounding.pdf`
- **Source basis:** full text (arXiv v2, 27 Nov 2018)

## Problem
Deployed recommenders are retrained on interaction data that was itself shaped by earlier recommendations — a feedback loop the authors call *algorithmic confounding*. Held-out accuracy cannot reveal its effects, and evaluating on confounded held-out data biases results toward systems similar to the confounding algorithm. The paper characterises the loop through simulation and makes three claims: (1) the loop homogenises user behaviour, and more cycles amplify this; (2) homogenisation brings no matching gain in utility, and the losses fall unequally across users; (3) the loop amplifies a recommender's effect on the item-consumption distribution, independent of homogenisation.

## Method
- **Interaction model (adapted from Schmit & Riquelme).** True utility is V_ui = P_ui + Q_ui, where P is the part known to the user and Q the part unknown to the user; the platform knows neither. V_ui ~ Beta′(ρ_u α_iᵀ), a mean/fixed-variance parameterisation with σ = 10⁻⁵. Preferences follow ρ_u ~ Dirichlet(µ_ρ) and attributes α_i ~ Dirichlet(µ_α), which gives long-tailed activity and popularity. The known share is η_ui ~ Beta′(µ_η) with µ_η = 0.98.
- **Choice rule.** A user picks i = argmax f(rank_u(i)) · P_ui with rank effect f(n) = n^−0.8. Each user interacts with exactly one item per step and with each item at most once. New items are interleaved with recommended ones.
- **Recommenders** in a common score framework s_ui = θ_u β_iᵀ: popularity, content filtering, MF (Gaussian PMF with confidence weighting, a = 1, b = 0.001), social filtering, random, and *ideal* (ranks by true V).
- **Metrics.**
  - Homogenisation: the change in Jaccard index of consumed-item sets between paired users, relative to the same users under ideal recommendations. Users are paired either by the cosine similarity of θ (local) or at random (global).
  - Utility: measured relative to ideal.
  - Gini coefficient of item consumption, computed from relative popularity rank.

## Assumptions held fixed
- **Static utility** (Assumption 1, V_ui(t) ≈ V_ui). Preferences ρ and attributes α are fixed. The loop changes *behaviour*, not true preferences, so any measured drift is a representation or consumption effect, not a preference effect.
- **Fixed user behaviour model.** One interaction per step, no repeats, and the same rank-decay f for every algorithm. The interleaving of new items is identical across systems, so differences are attributed to the algorithm alone.
- **Per-pair utility**, independent of the rest of the consumption set. The authors note that a set-level (diversity-aware) utility would probably make the effects larger.
- **The ideal and random baselines** are used to factor out artefacts of the interaction model.

## Experimental setup
- Synthetic communities of 100 users, with 10 new items per interval, K = 20 latent dimensions, and 10 random seeds ("worlds").
- Uneven user preferences: µ̃_ρ ~ Dir(1), µ_ρ = 10µ̃_ρ. Near-even item attributes: µ̃_α ~ Dir(100), µ_α = 0.1µ̃_α.
- **Two regimes:**
  - *Single training*: 50 start-up iterations, train once, then 50 confounded iterations.
  - *Repeated training*: 10 start-up iterations, then retrain every iteration for 90 iterations on all previous data.
- The text also states that each simulation runs "1,000 time intervals". This conflicts with the iteration counts above, and the paper does not reconcile them.

## Key results
- **Single training:** mild homogenisation appears right after training, then fades back to ideal levels (Fig. 3, left).
- **Repeated training:** every non-random system homogenises users beyond what ideal utility needs, and the excess grows with more loop cycles without matching utility gains (Fig. 3, right).
- **Global homogenisation** (random pairing): popularity increases it most, MF and social come next and are comparable, and content filtering increases it least but still more than ideal (Fig. 4).
- **Who loses:** users with lower utility relative to ideal tend to be *more* homogenised, across all algorithms (Fig. 5).
- **Consumption inequality:** similar homogenisation can come with different consumption Gini. MF and content homogenise comparably, but MF produces more unequal consumption (Fig. 6).
- **Weighting:** a preliminary exploration found that, in the repeated-training case, weighting (Schnabel-style IPS, De Myttenaere) "can simultaneously increase utility and decrease homogenization". No numbers are given.
- Results are shown in figures only; the paper reports no numeric tables.

## Reported limitations and trade-offs
- The study is fully synthetic: Dirichlet/Beta generative assumptions, one item per step, and no repeat consumption.
- Static utility and per-pair utility probably understate real effects.
- Weighting and causal fixes are only touched on and left to future work.
- The authors warn that public datasets (MovieLens, Douban) are confounded by their platforms' own recommenders, which favours similar algorithms in offline evaluation.

## Relevance to this project
- **§9 (feedback loops).** This is the cited justification that a single offline round cannot show loop effects. Its single-vs-repeated contrast (homogenisation fades after one training but grows under retraining) is exactly the gap §9 should fill.
- **§9 metric choice.** Jaccard-based homogenisation and consumption Gini are cheap loop-level metrics to track across rounds, alongside the §6 bias panel.
- **§5 IPS row / H on exposure correction.** Preliminary evidence that IPS-style weighting can improve utility *and* reduce homogenisation under repeated training. This gives a testable expectation for the IPS trajectory in §9.
- **§10 threats.** Supports the motivation for the random-exposure test split: confounded held-out data favours algorithms similar to the logging policy.
