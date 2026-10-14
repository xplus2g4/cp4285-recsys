# Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation

- **Authors:** Ruohan Zhan, Changhua Pei, Qiang Su, Jianfeng Wen, Xueliang Wang, Guanyu Mu, Dong Zheng, Peng Jiang (Kuaishou Technology; Zhan at HKUST)
- **Venue / year:** KDD 2022 (28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining), Washington DC
- **Link:** https://arxiv.org/abs/2206.06003 (DOI: https://doi.org/10.1145/3534678.3539092)
- **Local PDF:** `docs/literature/papers/zhan2022d2q.pdf`
- **Source basis:** full text (arXiv v1)

## Problem
Watch-time prediction for full-screen short-video feeds (Kuaishou). Watch time rises with video duration (Fig. 2: 60th-percentile watch time vs. duration on >20 billion samples), and a platform maximising watch time progressively over-exposes long videos (Fig. 3: impression change over 11 months skewed toward long-duration bins). Models trained on such data risk amplifying the bias through the feedback loop; long videos also dominate sample counts and (in WLR) sample weights. The paper claims to be the first study of duration bias in watch-time prediction.

## Method
- **Causal graph** over U (user), V (video), D (duration), W (watch time): {U,V}→W is interest; D→W is a *legitimate* duration effect to keep; D→V is the *bias* (duration affects which videos get exposed) to remove.
- **Backdoor adjustment:** E[W | do(U,V)] = Σ_d P(D=d) E[W | U,V,D=d]. Approximated by splitting training data into M equal-frequency duration groups (duration quantiles) and fitting per-group models f_k.
- **Quantile labels for parameter sharing (D2Q):** within each group k compute the empirical CDF Φ̂_k of watch time; the label for sample i is Φ̂_{k_i}(w_i) ∈ [0,1]. A single regressor h(u,v) is trained with MSE on these quantile labels across all groups; duration is also an input feature so the model can distinguish groups.
- **Inference:** predicted watch time ŵ = Φ̂_k⁻¹(h(u,v)) — i.e. the quantile prediction is mapped *back to the raw watch-time scale* of the video's duration group, and candidates are ranked by that value.
- **Res-D2Q:** adds a residual "duration adjustment" MLP tower in the last layer.
- Naive-D2Q (separate model per group) is described but rejected for scalability.

## Assumptions held fixed
- Duration is the **only** confounder treated; popularity, exposure/selection bias, position and clickbait are not modelled. The causal graph has no popularity node, so any duration–popularity coupling (RQ3) is outside the model.
- Within a duration group, the watch-time distribution is assumed to reflect interest only (duration is held approximately constant inside a bucket); bucket granularity trades bias removal against estimation error of Φ̂_k.
- The D→W effect is assumed legitimate and is **preserved**: the output is still raw watch time, so long videos still score higher for equal within-group quantile. D2Q removes the exposure-imbalance path, it does not normalise toward watch ratio.
- Exposure in the training log is taken as given (no propensity weighting); the logging policy is the production system.
- Groups have equal sample counts (quantiles of the logged duration distribution, not of the catalogue).

## Experimental setup
- **Data:** proprietary Kuaishou production logs — 47,298,353 users, 15,187,170 videos ("photos"), 5,942 categories, 1,346,539,657 instances; 1,211,885,691 train / 134,653,965 test samples. Not KuaiRand; no randomly exposed test data.
- **Baselines:** VR (value regression, MSE on raw watch time); WLR (Covington et al. 2016 weighted logistic regression; binary label = watch time > q60 of training watch time, positives weighted by watch time).
- **Metrics:** MAE, XAUC (pairwise ordering agreement on sampled pairs), XGAUC (per-user XAUC weighted by user sample size). No bias/diversity/duration-distribution metrics of the recommended lists.
- **Knob:** number of duration groups M ∈ {1, 10, 20, 30, 50, 100}.
- **Online A/B:** 5% user buckets, 24 h, VR baseline vs. WLR vs. Res-D2Q (M=30).
- Code: https://github.com/MorganSQ/Ks-D2Q

## Key results
- Table 2 (XAUC / XGAUC / MAE): VR 0.6843 / 0.6380 / 28.2413; WLR 0.6940 / 0.6436 / 65.6535; D2Q M=1 0.7084 / 0.6562 / 36.4871; D2Q M=30 0.7148 / 0.6660 / 26.5227; Res-D2Q M=30 0.7145 / 0.6693 / 26.5073 (best XGAUC and MAE). M=100: D2Q 0.7119 / 0.6608 / 26.8277.
- Performance is inverted-U in M: rises to ~20–30 groups then falls as per-group sample size shrinks (Fig. 6).
- Online: Res-D2Q +0.746% watch time vs. VR (CI ±0.15%); WLR +0.184%. Like/follow/share/comment changes for Res-D2Q not significant. Deployed on Kuaishou.

## Reported limitations and trade-offs
- Explicit trade-off: more duration buckets → better deconfounding but noisier empirical CDFs.
- Without the duration feature in the input, samples in different groups sharing a quantile label become indistinguishable.
- Evaluation measures prediction accuracy only; the paper does not measure the duration distribution of recommendations, so the claim that bias is removed is argued causally, not measured. [INFERENCE from absence of such metrics in §5.]

## Relevance to this project
- **§5 Duration row:** D2Q is the primary duration method; its knob "number of duration buckets" maps directly to M ∈ {1,10,20,30,50,100} from Table 2 — reuse that grid (KuaiRand is much smaller, so the optimum will shift lower).
- **§4 / H1:** D2Q ranks by Φ̂_k⁻¹(h), a *raw watch-time* prediction; only the intermediate quantile h(u,v) is duration-normalised. Whether D2Q moves mass to short videos depends on which score is ranked. The project must state which output is used and ideally report both (quantile-ranked ≈ duration-quantile label; value-ranked = D2Q as published).
- **§6 Duration metrics / §7.3 mass-shift:** the paper provides no list-level duration metrics, so the project's mean/median duration, Wasserstein and DurRSP/DurREO measurements are new evidence about D2Q.
- **§3 coupling / RQ3:** the causal graph omits popularity; the project's popularity × duration grid tests an interaction D2Q assumes away.
- **§11 novelty:** D2Q is evaluated on proprietary Kuaishou data, not KuaiRand, and has no random-exposure evaluation.
