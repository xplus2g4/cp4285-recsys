# Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time

- **Authors:** Haiyuan Zhao, Guohao Cai (equal contribution), Jieming Zhu, Zhenhua Dong, Jun Xu, Ji-Rong Wen (Renmin University of China; Huawei Noah's Ark Lab)
- **Venue / year:** KDD 2024 (30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining), Barcelona
- **Link:** https://arxiv.org/abs/2406.07932 (DOI: https://doi.org/10.1145/3637528.3671817)
- **Local PDF:** `docs/literature/papers/zhao2024cwm.pdf`
- **Source basis:** full text (arXiv v2, incl. appendices)

## Problem
Label-correction methods (PCR = watch ratio, WTG, D2Q, D2Co) normalise watch time by duration and implicitly treat every **completely played** record as maximal interest. On KuaiRand and WeChat, completely played longer videos receive a *higher* share of explicit positive feedback (like, forward) than completely played short ones (Fig. 1), so completion means less for short videos. The paper reframes duration bias as **right-censoring**: a latent *counterfactual watch time* (CWT) — how long the user would watch if the video were long enough — is truncated at the duration.

## Method
- **CWT definition:** w_{u,v} = min(w^c_{u,v}, d_v); w^c is assumed independent of duration. Supporting evidence on KuaiRand/WeChat: repeated-play share and ratio are higher for shorter videos (Fig. 3); watch time at fixed duration (30 s) is bimodal (Fig. 4), consistent with a truncated Gaussian.
- **Theorem 1:** no transform of observed watch time alone can recover interest for completed plays (only an inequality is known) — i.e. PCR/WTG/D2Q-style label maps are provably insufficient there.
- **Economic user model:** cumulative reward ω(r)·log(w^c + 1) with diminishing marginal reward, constant marginal cost c per second, rational stop when marginal reward = cost ⇒ w^c = ω(r)/c − 1; with ω(r) = 1/(−log r): w^c = g(r; c) = 1/(−c·log r) − 1 and r = g⁻¹(w^c; c) = exp(1/(−c(w^c + 1))).
- **Counterfactual (Tobit-style) likelihood:** for incomplete plays, Gaussian density term on g′(w) = Φ⁻¹(g⁻¹(w)); for complete plays, survival term log Φ((f_θ(x) − g′(d_v))/σ). Log-likelihood = MSE part on uncensored records + "amplification" part on censored ones. Φ approximated by a sigmoid.
- **Inference:** r̂ = Φ(f_θ(x)) for ranking; ŵ = clip(1/(−c log r̂) − 1, 0, d_v) for watch-time prediction.
- Hyper-parameters (c, σ): KuaiRand (1/40, 2); WeChat (1/40, 20); Product (1/5, 5).

## Assumptions held fixed
- Duration acts **only** through truncation; CWT is duration-independent. No popularity, exposure or position modelling; logged exposure is taken as given.
- Constant per-user watch cost c (a single global hyper-parameter), Gaussian latent interest, specific functional forms for reward/cost.
- Truncation at 0 (strong dislike) is ignored — the paper focuses on liked videos.
- Evaluation label for ranking is itself watch-time-derived: r = 1 if (d ≤ w₀.₇ and w = d) or (d > w₀.₇ and w > w₀.₇), with w₀.₇ the 70th percentile of watch time — a `long_view`-style threshold (Appendix C argues it is roughly duration-independent, Fig. 10). Videos shorter than w₀.₇ are excluded from that check.

## Experimental setup
- **Datasets** (Table 1): **KuaiRand-Pure** (videos ≤ 400 s; 26,988 users, 6,598 videos, 1,266,560 interactions; 17.5% completely played; temporal split 14/7/10 days; features include `user_active_degree`, fans/follow/friend ranges, `most_popular_tag`, video_type, upload_type, tab); WeChat Channels (60-s videos excluded because they are 17.3% of data; 45.5% completely played); Product (Huawei video platform, 2,000,000 users, 1,011,007 videos, 36,366,437 interactions, records < 42 s; 32.8% complete).
- **Tasks/metrics:** watch-time prediction (MAE, XAUC) and relevance ranking (AUC, nDCG@3).
- **Baselines:** VR, PCR, WTG, D2Q (60 groups KuaiRand, 30 WeChat, 10 Product), D2Co, WLR, NDT; plus CVRDD, DCR, VLDRec (Table 4). Oracle = trained directly on the evaluation label.
- **Backbones:** FM, DCN, AutoInt (embedding 10, hidden 64, dropout 0.2, Adam 5e-4, batch 512). Code: https://github.com/hyz20/CWM.git

## Key results
- **KuaiRand ranking (Table 3, FM):** AUC Oracle 0.738; VR 0.661; PCR 0.686; D2Q 0.679; WTG 0.684; NDT 0.668; D2Co 0.688; **CWM 0.735**. nDCG@3: VR 0.442 → CWM 0.486 (Oracle 0.489).
- **KuaiRand watch-time prediction (Table 2, FM):** MAE VR 22.100, D2Q 18.271, CWM 17.738; XAUC VR 0.683, PCR 0.697, D2Q 0.646, CWM 0.714. D2Q has the second-best MAE but the worst XAUC.
- **WeChat:** several duration-debiasing labels underperform naive VR (FM AUC: VR 0.639, WTG 0.602, D2Co 0.573; CWM 0.703), attributed to the high (45.5%) complete-play share.
- **By duration (Fig. 8, KuaiRand, 10 bins up to 400 s):** baselines beat VR on short videos (< 30 s) but fall below VR on longer videos; CWM is more even across bins.
- Table 4: CWM beats CVRDD/DCR/VLDRec on all datasets (e.g. KuaiRand FM AUC 0.735 vs. DCR 0.695).
- Ablation: on KuaiRand the transform function matters most (CWM-C drops); on WeChat the censored likelihood matters most (CWM-L drops).
- Sensitivity (Fig. 11, KuaiRand FM): best XAUC at σ ∈ (1, 2), c ∈ (1/40, 1/20); best nDCG@3 at σ ∈ (2, 5), c ∈ (1/80, 1/60) — the two tasks prefer different settings.
- Online A/B (one week, 5% traffic): MWT +2.9%, VV +2.5%, CTR +0.3%.

## Reported limitations and trade-offs
- Optimal (c, σ) differ between ranking and watch-time prediction; requires tuning.
- The censored term "simply amplifies predictions for fully watched videos" and "cannot precisely model interest" (authors' own caveat).
- Dislike side (truncation at 0) not modelled.
- Ground truth for ranking is a constructed watch-time threshold, not random-exposure or explicit satisfaction.

## Relevance to this project
- **H1 (direct challenge):** CWM's KuaiRand results (Fig. 8) show PCR/WTG/D2Q/D2Co help short videos but *hurt* long videos relative to naive regression — consistent with H1's direction (normalisation shifts toward short videos) and gives a mechanism: completions of short videos are over-credited.
- **§4 Labels:** the watch-ratio cap in §4 interacts with CWM's censoring argument — completed plays (ratio = 1, or > 1 with replays) are where labels disagree. Report the share of capped records and consider a censored (Tobit) label variant as a fifth duration method.
- **§5 Duration row:** D2Q with 60 groups on KuaiRand-Pure is the reference configuration used in both D2Co and CWM; reuse for comparability. CWM supplies a KuaiRand-tuned (c, σ) = (1/40, 2).
- **§11 novelty — evaluates on KuaiRand-Pure:** with D2Co, CWM is prior work benchmarking duration labels on KuaiRand. Neither uses the random-exposure log, cross-label evaluation, or popularity/coverage metrics, which keeps the project's audit-matrix contribution distinct.
- **§6 Clickbait / §4 Satisfaction:** Fig. 1 (explicit-feedback rate among completed plays rises with duration) means the satisfaction label is itself duration-correlated on KuaiRand — relevant to treating it as "least proxy-like".
