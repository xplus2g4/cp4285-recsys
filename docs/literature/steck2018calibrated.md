# Calibrated Recommendations

- **Authors:** Harald Steck (Netflix)
- **Venue / year:** Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18), Vancouver, 2018
- **Link:** https://doi.org/10.1145/3240323.3240372
- **Local PDF:** `docs/literature/papers/steck2018calibrated.pdf`
- **Source basis:** full text (ACM open-access PDF)

## Problem
Recommenders trained for accuracy in the usual offline setting tend to fill the list with a user's main interests and crowd out lesser ones. A user who watched 70 romance and 30 action movies should get roughly 70%/30% romance/action, but an accuracy-optimal top-10 can be 100% romance. Steck calls the target property *calibration*: the categories in the list should match the user's past proportions. He frames it as a user-level notion of fairness across a user's *interests*, not across groups of people.

## Method
**Thought experiments (§2).** (i) Class imbalance: with only genre preferences known, always picking the majority class maximises accuracy. Predicting +1/−1 at random with 70/30 probabilities is right 0.7·70% + 0.3·30% = 58% of the time, versus 70% for always picking the majority. (ii) With item-level probabilities p(i|g)·p(g|u), the 10th romance title is still ≈2.33× more likely than the best action title (ratio 2.1 estimated from ML-20M). (iii) Under LDA, *sampling* stays calibrated but *ranking* does not. Steck concludes that calibration should be expected to cost accuracy.

**Calibration metric (§3), exact definition:**
- History distribution: p(g|u) = Σ_{i∈H} w_{u,i}·p(g|i) / Σ_{i∈H} w_{u,i} (Eq. 2). w_{u,i} is an optional per-item weight, e.g. recency.
- List distribution: q(g|u) = Σ_{i∈I} w_{r(i)}·p(g|i) / Σ_{i∈I} w_{r(i)} (Eq. 3). w_{r(i)} is a rank weight; MRR- or nDCG-style weights are suggested.
- C_KL(p,q) = KL(p‖q̃) = Σ_g p(g|u) log[p(g|u)/q̃(g|u)] (Eq. 4). The **history p is the target**, so this is KL(history ‖ smoothed list), not the reverse.
- Smoothing: q̃(g|u) = (1−α)·q(g|u) + α·p(g|u) (Eq. 5), with small α > 0 so that KL stays finite when q = 0 and p > 0. **α = 0.01 in the experiments.** The list distribution is mixed with the *user's own history*, not with a uniform distribution.
- The metric has three properties: it is 0 under perfect calibration; it is very sensitive to discrepancies in small-p genres (2%→1% costs more than 50%→49%); and it favours less extreme distributions (31% beats 29% for a 30% target).
- Alternative: Hellinger distance ‖√p − √q‖₂/2, which is defined with zeros but less sensitive to small-p genres.
- The overall C is the average of C(p,q) over users.

**Re-ranking (§4).** The method is post-processing via maximum marginal relevance: I* = argmax_{|I|=N} (1−λ)·s(I) − λ·C_KL(p, q(I)) (Eq. 6), with s(I) = Σ s(i). The list is built greedily. Through a submodular surrogate (Appendix, Eq. 9) every prefix gets a (1−1/e) guarantee. Several category types can be calibrated at once by adding separate λ-weighted C_KL terms.

**Extension (§5.1, Eq. 7).** p̄(g|u) = β·p₀(g) + (1−β)·p(g|u) mixes in a prior p₀ (uniform, or the average over users) so that genres outside the history also appear. β ∈ [0,1] trades diversity against calibration.

## Assumptions held fixed
- **The user's past distribution is the correct target.** Calibration copies history, including any distortion the logging policy put into it. The paper evaluates on random splits of observational data and never questions whether H is itself policy-biased.
- Item–category memberships p(g|i) are given and fixed. Multi-genre movies get equal weights.
- The base scores s(i) are fixed. Calibration only re-orders a fixed candidate list; the model is not retrained.
- In the experiments, no prior p₀ and no weights in Eqs. 2–3 (uniform w).
- Steck takes a user-centric view only. Item-side calibration ("is an item recommended twice as often consumed twice as often") is left for future work.

## Experimental setup
- **Data:** MovieLens 20M turned into implicit feedback (ratings ≥4 → play). About 10M plays, 21k movies, 140k users; genres from ML metadata.
- **Split:** random 99%/1% train/test, giving about 100k test plays.
- **Baseline:** weighted 50-dimensional MF tuned for recall@50.
- **Metrics:** recall@10/@50 and C_KL@10/@50. Default λ = 0.99.

## Key results
Table 2 (λ: recall@10, recall@50, C_KL@10, C_KL@50):
- λ=0: 0.209, 0.464, 0.677, 0.185
- λ=0.5: 0.199, 0.464, 0.274, 0.141
- λ=0.9: 0.146, 0.460, 0.084, 0.061
- λ=0.99: 0.090, 0.417, 0.054, 0.009
- λ=0.999: 0.082, 0.339, 0.054, 0.005

Small λ improves calibration a lot at little accuracy cost. recall@50 stays nearly flat up to λ≈0.95 and then drops; recall@10 degrades sooner.

Other findings:
- C_KL@10 is larger than C_KL@50 because 10 items constrain the distribution more.
- Baseline calibration varies widely across users (Fig. 1).
- Among the 10% least-calibrated users, the baseline nearly omits documentaries (≈3% of plays), IMAX, Musical and Western, and over-represents Action, Adventure, Crime, Mystery and Sci-Fi. After calibration Drama is slightly under-represented on average.
- Table 1 shows that C_KL meets all three properties, whereas DP [Dang & Croft] meets only property 1 and BinomDiv [Vargas et al.] fails property 1.

## Reported limitations and trade-offs
- Calibration explicitly costs accuracy, and strongly so at high λ.
- Calibrated lists can keep a user inside their filter bubble; the β-prior is the proposed fix.
- Exact hypothesis tests that would account for finite N and M are dropped for computational reasons.
- Steck suggests ramping λ from small to large during the greedy pass. He also suggests listwise learning-to-rank and settings beyond offline evaluation as future work. Item-side calibration is not addressed.

## Relevance to this project
- **§6 User groups, "calibration error (Steck RecSys '18)".** This paper supplies the exact formula. Implement C_KL = KL(p_hist ‖ q̃_list) with q̃ = 0.99·q + 0.01·p over popularity bins and over duration bins. The spec's phrase "KL between history and list" should state this direction and this α.
- **§6 Duration/Popularity panels; H1, H3.** Applying C_KL to popularity-quintile and duration-quintile "genres" turns the paper's genre metric into per-user popularity and duration miscalibration. It complements the list-level ARP and Wasserstein metrics with user-relative ones.
- **§7.2 Frontiers / RQ4.** Steck's λ re-ranker is a natural extra knob-swept method, producing a calibration–accuracy curve like Table 2. With p(g|i) defined over the 5×5 popularity×duration grid (§3), a single calibration term targets both dimensions jointly.
- **§10 Threats.** Calibration treats the logged history as the target. On KuaiRand, the history comes from the biased standard logs, so "calibrated" ≠ "unbiased". This should be noted when interpreting the calibration columns.
