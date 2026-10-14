# Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation

- **Authors:** Haiyuan Zhao, Lei Zhang, Jun Xu, Guohao Cai, Zhenhua Dong, Ji-Rong Wen (Renmin University of China; Huawei Noah's Ark Lab)
- **Venue / year:** RecSys 2023 (17th ACM Conference on Recommender Systems), Singapore
- **Link:** https://arxiv.org/abs/2308.08120 (DOI: https://doi.org/10.1145/3604915.3608797)
- **Local PDF:** `docs/literature/papers/zhao2023d2co.pdf`
- **Source basis:** full text (arXiv v1)

## Problem
Watch time is a biased and noisy proxy for interest: (i) **duration bias** — interested users watch longer on longer videos; (ii) **noisy watching** — users spend time on videos they dislike before deciding to skip. A pilot study **on KuaiRand** shows mean watch time of "interested" records (any of like/follow/forward/comment/profile-enter) rises with duration (Fig. 2a), and mean watch time of `is_hate` records is non-zero and also rises with duration (Fig. 2b). Prior labels (PCR = watch ratio, WTG, D2Q) address only duration bias and rely on assumptions about the interest distribution per duration.

## Method
- **Causal decomposition:** X → D, X → R (latent binary interest), {D, R} → W. Then E[W|x] = p_x·w⁺_d + (1 − p_x)·w⁻_d, where w⁺_d = E[W|d, R=1] (duration-biased watch time) and w⁻_d = E[W|d, R=0] (noisy watch time). Theorem 1 bounds the error of scaled watch time as bias error + noise error.
- **Analysis of existing labels (§3.3, App. A):** PCR (w/d) recovers interest iff w⁺_d = C₁d and w⁻_d = C₂d (linear in duration) — violated per Fig. 2. WTG recovers interest iff every duration group has the same interest mean and std — violated because interest and duration share the parent X. D2Q (quantile within equal-frequency duration bin) recovers interest iff all bins share the same interest ranking function; satisfying it needs few bins, which weakens debiasing.
- **Estimation:** for each duration value d, fit a 2-component GMM to watch times → ŵ⁺_d, ŵ⁻_d; smooth across adjacent durations with a bidirectional frequency-weighted moving average (window T).
- **Correction:** D2Co(A) = (w − w̃⁻_d)/(w̃⁺_d − w̃⁻_d) (affine; unbiased if estimates exact). D2Co(S) = (e^{αw} − e^{αw̃⁻_d})/(e^{αw̃⁺_d} − e^{αw̃⁻_d}); α < 0 reduces sensitivity to w⁺_d errors (long-watch datasets, e.g. KuaiRand), α > 0 to w⁻_d errors (short-watch, e.g. WeChat). Labels clipped to [0,1]; models trained with BCE.

## Assumptions held fixed
- Duration is the only confounder of watch time besides interest; **popularity and exposure are not modelled**. The GMM is fitted on the logged (policy-exposed) distribution per duration; duration-level mixture weights absorb the average interest at that duration.
- Watch time per (d, R) is Gaussian ("a wild assumption" in the authors' words); interest is binary.
- Ground truth for evaluation is **KuaiRand `long_view`** (r = 1 if d ≤ 18 s and w = d, or d > 18 s and w > 18 s) — itself a deterministic function of watch time and duration, so evaluation is on a duration-adjusted watch-time label, not on an independent satisfaction signal.
- Evaluation on logged standard data (temporal split), not on the random-exposure log.

## Experimental setup
- **KuaiRand-Pure** (as suggested by the KuaiRand authors), videos ≤ 4 min: 26,988 users, 6,598 videos, 1,266,560 interactions, durations [5, 240] s; temporal split first 14 days / middle 7 / last 10 → train / val / test. Features include user_id, video_id, author_id, music_id, video_type, upload_type, tab, is_like, is_follow, is_comment, is_forward, is_profile_enter, is_hate, most_popular_tag.
- **WeChat** (Big Data Challenge 2021): Table 1 lists 20,000 users, 96,418 videos, 7,310,108 interactions, durations [5, 60] s; 10/2/2-day split as in DVR.
- **Baselines:** Watch Time (w / w_max), PCR, D2Q (60 groups KuaiRand, 30 WeChat), WTG (mapped through Φ), and "-denoise" variants (labels set to 0 when watch time < 5 s). An "Oracle" row is reported but not defined in the text [INFERENCE: presumably trained on the long_view label].
- **Backbones:** FM, DeepFM, AutoInt (pytorch-fm; embedding 10; hidden 64; dropout 0.2; Adam 0.001; batch 512). Metrics: GAUC, nDCG@{1,3,5}. T ∈ [1,5]; α ∈ [−0.05, −0.01] on KuaiRand. Code: https://github.com/hyz20/D2Co.git
- **Online A/B:** Huawei browser video feed, 7 days.

## Key results
- Table 2, KuaiRand, DeepFM GAUC / nDCG@1: Watch Time 0.593 / 0.402; PCR 0.628 / 0.435; D2Q 0.635 / 0.437; WTG 0.635 / 0.436; WTG-denoise 0.647 / 0.444; D2Co(A) 0.653 / 0.447; **D2Co(S) 0.656 / 0.451**; Oracle 0.666 / 0.459. FM: Watch Time 0.584 → D2Co(S) 0.653 GAUC. All duration-normalised labels beat raw watch time by a large margin on KuaiRand.
- On KuaiRand, duration bias dominates the error (noise-caused error ≈ 0, Fig. 5a); on WeChat noise dominates in long durations.
- Table 3 (DeepFM nDCG@1 by equal-frequency duration tercile, KuaiRand): (0,32] WT 0.380 → D2Co(S) 0.397; (32,94] 0.394 → 0.411; (94,240] 0.371 → 0.389. On WeChat (42,60], PCR/D2Q/WTG fall *below* raw watch time (−20.3%, −35.4%, −4.8% relative-improvement).
- Best KuaiRand hyper-parameters: FM T ∈ {2,3,4}, α = −0.07; DeepFM T ∈ {2,3,4}, α = −0.05; AutoInt T = 2, α = −0.05 (Fig. 7 GAUC 0.649–0.657).
- Online (7-day average): Impression +5.41%, VV +8.35%, MWT +1.31% (fluctuating, −4.00% on day 4), PCR +4.81%, CTR +2.92%.

## Reported limitations and trade-offs
- Larger |α| lowers sensitivity but sacrifices unbiasedness; larger T smooths at the cost of specificity; best values differ by backbone and must be tuned.
- Gaussian component assumption is acknowledged as strong; per-duration GMMs need enough samples (hence the ≤ 4-min filter on KuaiRand).
- Heuristic denoising of baselines removes only part of the noise.
- Duration normalisations can lose accuracy on long videos when noisy watching dominates (WeChat Table 3).

## Relevance to this project
- **§11 novelty — evaluates on KuaiRand-Pure:** prior work already compares watch time, watch ratio (PCR), D2Q, WTG on KuaiRand. The project's novelty claim must rest on random-log evaluation, cross-label evaluation and off-target (popularity/exposure) bias metrics — none of which D2Co reports.
- **§4 Labels / H1:** D2Co shows PCR's linearity assumption fails on KuaiRand (Fig. 2) and that duration-normalised labels win on a `long_view` test label that is itself duration-adjusted — exactly the "change of definition" confound §4 is designed to expose. The §4 label matrix should add D2Co(S) as a candidate duration label and evaluate all of them on `is_click` and satisfaction too.
- **§5 Duration row:** D2Co implements D2Q as a *label* (within-bin quantile trained with BCE, 60 groups on KuaiRand) — matching the spec's "duration-quantile label"; group count 60 is a reference point for the bucket sweep.
- **§6 Clickbait:** the noisy-watching component (w⁻_d, watch time on `is_hate` records) is a duration-dependent analogue of the clickbait metric; Fig. 2b on KuaiRand suggests the "low watch ratio" threshold in §6 should be duration-conditional.
- **§3 coupling:** the per-tercile breakdown (Table 3) is a 1-D version of the §7.4 grid; the project extends it with the popularity axis.
