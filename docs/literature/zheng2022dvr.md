# DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias

- **Authors:** Yu Zheng, Chen Gao, Jingtao Ding, Lingling Yi, Depeng Jin, Yong Li, Meng Wang (Tsinghua University; Tencent WeChat; Hefei University of Technology)
- **Venue / year:** ACM Multimedia (MM '22), Lisboa, 2022
- **Link:** https://arxiv.org/abs/2208.05190 (DOI: https://doi.org/10.1145/3503161.3548428)
- **Local PDF:** `docs/literature/papers/zheng2022dvr.pdf`
- **Source basis:** full text (arXiv v1, incl. appendix)

## Problem
Watch time grows with duration (Fig. 1a), so watch-time-trained models over-recommend long micro-videos; the obvious fix, watch percentage, reverses the bias toward short videos (Fig. 1b). The paper shows (Table 1, WeChat Channels) that seven CTR-style regressors trained on watch time reach WatchTime@k ≈ 202–206 s, no better than a trivial **LongRec** that ranks by duration (202.92), and their number of "bad cases" (#BC: recommended video watched < 2 s) is only slightly below **RandomRec** (3,850 vs. 3,515–3,568). Producers of long videos receive > 80% of recommendation traffic vs. < 20% for short-video producers (Fig. 2b, FM). Two RQs: (1) an unbiased evaluation metric for watch time; (2) a model-agnostic debiased learner.

## Method
- **WTG (Watch Time Gain):** divide videos into equal-width duration bins (1 s per bin); WTG = (WT − μ_B(v)) / σ_B(v), i.e. watch time z-scored within the duration bin, μ and σ computed over all users on the whole dataset. Streaming update formulas given (Alg. 3).
- **Metrics:** WTG@k = mean ground-truth WTG of top-k; DCWTG@k = Σ WTG(l_i)/log₂(1+i) (no ideal-DCG normalisation); #BC@k.
- **DVR** (three components): (i) **DD** — delete duration from input features; (ii) **WTG target** — regress WTG with MSE and rank by predicted WTG; (iii) **ADV** — a duration regressor Ψ (1×1 dense layer) predicts duration from predicted WTG, connected via a gradient reversal layer; Φ minimises L_WTG − α·L_D.
- **DVR-** variant: train on watch time, convert prediction to WTG at ranking time.

## Assumptions held fixed
- Duration is the only bias; within a 1-s bin, watch-time differences are assumed to be pure preference. Popularity is **kept as an input feature** (`VideoPopularity`, `UserActiveness` on both datasets) and is not debiased; any duration↔popularity correlation can still leak duration information through popularity — the adversary only targets duration in the *output*, not popularity.
- Exposure is taken as given; evaluation is on the logged (policy-biased) test interactions, ranked within each user's test items. No random exposure, no propensity correction.
- Bin statistics (μ, σ) are global, not per user; extreme durations are dropped (WeChat kept 5–60 s, Kuaishou 5–120 s; < 0.1% of records removed) so every bin has > 10,000 points.
- Evaluation uses WTG itself, i.e. the training target equals the evaluation target — the "change of definition" issue flagged in spec §4.

## Experimental setup
- **Datasets** (Table 3): WeChat Channels (WeChat Big Data Challenge 2021): 10,000 users, 639,557 videos, 2,672,809 records; temporal split 10/2/2 days. Kuaishou (China MM 2018 competition dataset, not KuaiRand): 20,000 users, 96,418 videos, 7,310,108 records; temporal 8:1:1.
- **Backbones:** LibFM, Wide&Deep, DeepFM, NFM, AFM, AutoInt, AFN; 3 hidden layers × 64 units, Adam lr 0.001, batch 512.
- **Metrics:** WTG@10, DCWTG@10, #BC@10; also duration histograms of recommended videos (Fig. 8).
- Code: https://github.com/tsinghua-fib-lab/WTG-DVR

## Key results
- **Mass shift (Fig. 8, WeChat):** watch-time-trained LibFM/AutoInt give 52.3% / 55.6% of traffic to 60-s videos (dataset maximum) and < 1.46% to videos < 20 s; with WTG target, 60-s share drops to 18.47% / 18.95% and < 20 s rises to about 8.59%.
- **Table 2, WTG@10 / DCWTG@10 / #BC@10:** WeChat FM 0.0209 / 0.2985 / 6381 → DVR 0.1332 / 1.5100 / 5947; WeChat AFN 0.0157 / 0.2599 / 6358 → DVR 0.1408 / 1.5858 / 5917. Kuaishou FM 0.0571 / 0.4178 / 5854 → DVR 0.2094 / 1.6137 / 5240; Kuaishou AFN 0.0536 / 0.4037 / 5832 → DVR 0.2015 / 1.5551 / 5229. WTG@10 gains described as > 300% in most cases, > 500% for AFN; #BC@10 about 21.65% lower than AFM on Kuaishou.
- DVR- (post-hoc conversion) already captures most of the gain (e.g. WeChat DeepFM DVR- 0.1372 vs. DVR 0.1469 WTG@10).
- **Ablation (Table 4, DCWTG@10):** WeChat NFM 0.3334: +DD +8.59%, +WTG +354.28%, +ADV +386.68%; Kuaishou AFN 0.4037: +49.08%, +267.39%, +285.21%. Most gain comes from the label change.
- **α:** Fig. 9 shows best DCWTG/#BC at α ≈ 0.3–0.4 on WeChat; the implementation section states "the optimal α is 0.1" (internal inconsistency).

## Reported limitations and trade-offs
- Watch percentage is explicitly shown to swap long-video bias for short-video bias; WTG is proposed as neutral.
- Too-small α leaves predictions correlated with duration; too-large α causes underfitting of the main task.
- Bins need enough data; sparse extreme durations are discarded.
- No online A/B test (listed as future work). Only WTG-family metrics are reported, so a gain is partly by construction [INFERENCE: models trained on watch time are scored on a z-scored target they were not trained for].

## Relevance to this project
- **§4 Labels:** WTG is a fourth duration-normalised label alongside watch ratio and D2Q quantile; adding it (z-score per duration bin) costs one column and tests whether "neutral" normalisation behaves differently from watch ratio.
- **H1 / §7.3 mass-shift:** the paper's own evidence (Fig. 1b) supports H1's premise that watch ratio favours short videos, and Fig. 8 gives a reference mass shift for WTG; the project can test whether these shifts are popularity-coupled — DVR never measures popularity effects although `VideoPopularity` is a feature.
- **§4 "change of definition" critique:** DVR's headline gains come from evaluating on its own training target (ablation: +WTG dominates); the project's method × evaluation-label table is precisely the control this paper lacks.
- **§6 Duration metrics:** the duration histogram of top-k and the "bad case" (< 2 s watch) count are cheap additions/analogues to the clickbait metric.
- **§11 novelty:** evaluated on WeChat Channels and the 2018 Kuaishou competition data, not KuaiRand, and not on random exposure.
