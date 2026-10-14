# Causal Intervention for Leveraging Popularity Bias in Recommendation

- **Authors:** Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, Yongdong Zhang
- **Venue / year:** SIGIR 2021 (44th International ACM SIGIR Conference)
- **Link:** https://arxiv.org/abs/2105.06067 (DOI https://doi.org/10.1145/3404835.3462875)
- **Local PDF:** `docs/literature/papers/zhang2021pda.pdf`
- **Source basis:** full text

## Problem
Collaborative filtering inherits and amplifies the long-tail popularity distribution of the training data. Existing fixes (IPS, causal embeddings trained on uniform data, re-ranking/regularisation) try to *remove* popularity entirely. The authors argue that some popularity is benign (intrinsic quality, current trends) and that the goal should be to remove only the harmful part (bias amplification) and then *inject* a desired popularity signal (the forecast popularity of the next period) at inference.

## Method
- Causal graph with U (user), I (exposed item), C (interaction), Z (item popularity). Z → C models conformity (kept); Z → I models popularity-driven exposure, making Z a confounder; the backdoor path I ← Z → C is the "bad" effect.
- **PD (Popularity-bias Deconfounding):** estimate P(C | do(U,I)) = Σ_z P(C|U,I,z)P(z) via backdoor adjustment. Parameterised as P_Θ(c=1|u,i,m_i^t) = ELU'(f_Θ(u,i)) × (m_i^t)^γ, where f_Θ is MF, ELU' keeps the score positive, and m_i^t is the item's *local* popularity in time stage t (interaction share of item i in stage t). Trained with BPR. Because the popularity term factorises, P(C|do(U,I)) ∝ ELU'(f_Θ(u,i)) (E[Z^γ] is a constant), so ranking uses the matching term only.
- **PDA (… and Adjusting):** at inference, intervene do(Z = z̃): score = ELU'(f_Θ(u,i)) × (m̃_i)^γ̃, with m̃_i either the last-stage popularity (method a) or a linear forecast m̃_i = m_i^T + α(m_i^T − m_i^{T−1}) (method b, Eq. 8). γ̃ may differ from training γ; in practice γ̃ = γ; γ̃ = 0 reduces PDA to PD.
- They also show P(C|U,I) contains an extra P(I|Z) factor, which is the amplification term.

## Assumptions held fixed
- Popularity is the *only* confounder between exposure and interaction; other item attributes (e.g., duration, quality, position) are not in the graph. Users interact only with exposed items (search ignored).
- U is independent of Z; the conformity mechanism {U,I,Z}→C is unchanged under intervention, and P(Z) is the same on both graphs.
- The effect of popularity on the score is multiplicative and separable from user–item matching ((m_i^t)^γ), with one global exponent γ shared by all users — no per-user conformity.
- The test target is next-period data whose popularity is assumed predictable from recent trend (linear extrapolation); i.e., some popularity in the test set is "deserved" and should be re-injected.
- No new users/items in validation/test.

## Experimental setup
- Datasets: **Kwai** (Kuaishou User Interest Modeling Challenge, clicks, ~2 weeks, 10-core: 7,658,510 interactions, 37,663 users, 128,879 items); **Douban Movie** (after 2010, 10-core: 7,174,218 / 47,890 / 26,047); **Tencent** short video likes (Dec 27 2020–Jan 6 2021, 5-core: 1,816,046 / 80,339 / 27,070).
- Split chronologically into 10 equal-time stages; last stage for validation (30% users) / test (70% users). Test data therefore *contains* popularity bias (not random exposure).
- Metrics: Recall, Precision, HR, NDCG @20/@50, all-ranking protocol. Recommendation Rate (RR) per 10 popularity groups of equal total popularity, plus its std. dev.
- Baselines: MostPop, BPRMF, xQuAD, BPR-PC, DICE (IPS omitted because DICE beat IPS in its paper); for PDA: MostRecent, BPRMF(t)-pop, BPRMF-A, DICE-A.
- Hyper-parameters: γ searched in [0.02, 0.25] step 0.02 for PD/PDA; γ̃ for BPRMF-A/DICE-A from 0.02 to 1 step 0.02; Adam, lr 0.001, batch 2048, early stopping on Recall@20 (100 epochs patience).

## Key results
- PD beats all baselines by at least 119% (Kwai), 34% (Douban), 24% (Tencent). Kwai Top-20: PD Recall 0.0143 / NDCG 0.0177 vs BPRMF 0.0054 / 0.0067 and DICE 0.0053 / 0.0067.
- Gains scale with popularity drift (Kwai drift largest, by JSD-based Drift of Popularity).
- RR analysis: BPRMF's RR falls almost linearly from popular to unpopular groups (amplification). DICE over-suppresses groups 1–2 (below training rate on Kwai, Tencent) and over-amplifies middle groups. PD has the flattest RR curve and smallest RR std. dev., closest to training distribution; still under-serves group 10 (sparse items).
- Re-ranking (xQuAD, BPR-PC) barely changes recall.
- Local popularity matters: PD-G (global popularity) is far worse than PD.
- PDA (method b) Kwai Top-20 Recall 0.0288 / NDCG 0.0364 vs DICE-A 0.0245 / 0.0323 and BPRMF(t)-pop 0.0188 / 0.0241. Relative to BPRMF, PDA improves NDCG by 532% (Kwai), 97% (Douban), 161% (Tencent) (Table 3, averaged over K=20,50).
- Finer last-stage popularity estimates (N sub-stages) help, up to a point on Kwai.

## Reported limitations and trade-offs
- PD still gives too little exposure to the least popular group because of sparse embeddings.
- Popularity forecasting is deliberately simple; Douban's long span makes trend prediction harder.
- Global popularity fails; method depends on stage granularity.
- Authors note DICE's own evaluation used intervened (popularity-even) test data, whereas their test is temporally split and popularity-biased — so "accuracy gains" partly reflect re-injected popularity matching a biased test. No user-group or calibration analysis; no other bias dimensions examined.

## Relevance to this project
- **§5 Popularity row:** primary popularity method. Knob mapping: training exponent γ (PD) and inference exponent γ̃ (PDA). Note γ̃ = 0 is PD (pure deconfounded), larger γ̃ re-injects popularity — a natural one-parameter sweep for §7 Pareto frontiers.
- **§6 Popularity metrics / §7 mass-shift:** PDA's RR-per-group with equal-popularity-mass groups is a direct precedent for the per-cell recommendation-share analysis; reuse its group construction for popularity quantiles.
- **H1/H3, RQ2–RQ3:** PD's graph has no duration node; on KuaiRand duration can be a second confounder correlated with Z, so PD may shift mass along duration — exactly the spill-over this project audits.
- **§2/§4 evaluation:** PDA was validated on a popularity-biased, time-split test; evaluating on KuaiRand's random-exposure log tests whether its accuracy gains survive without popularity in the test target.
- Kwai dataset in PDA is a Kuaishou predecessor of KuaiRand, so PDA's setting is closely comparable.
