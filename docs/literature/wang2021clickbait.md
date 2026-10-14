# Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue

- **Authors:** Wenjie Wang, Fuli Feng, Xiangnan He, Hanwang Zhang, Tat-Seng Chua (NUS; USTC; NTU)
- **Venue / year:** SIGIR 2021 (44th International ACM SIGIR Conference), virtual
- **Link:** https://arxiv.org/abs/2009.09945 (DOI: https://doi.org/10.1145/3404835.3462962)
- **Local PDF:** `docs/literature/papers/wang2021clickbait.pdf`
- **Source basis:** full text (arXiv v4); the extracted text ends partway through §6.3 (in-depth analysis, Fig. 7/Table 5 body), so details there are not summarised.

## Problem
CTR-trained recommenders learn that attractive *exposure features* (title, cover, caption — visible before the click) cause clicks, so items whose content disappoints still rank high and the problem self-reinforces (Matthew effect). Post-click feedback (likes) would fix it but is sparse. Goal: mitigate clickbait using **click data only** at training time.

## Method
- **Causal graph:** U (user), E (exposure features), T (content features), I = f_I(E,T) (item representation), Y (score). Conventional models have only E→I→Y; the proposed graph adds a **direct edge E→Y**, representing clicks driven purely by exposure features.
- **Counterfactual inference:** the clickbait component is the *natural direct effect* (NDE) of E on Y: Y(u, i*, e) − Y(u, i*, e*), "what the score would be if the item had only its exposure features". Ranking uses Y_CR = Y_{u,i,e} − Y_{u,i*,e} (TE − NDE = TIE). NDE is estimated under the reference content situation T = t* to avoid leakage of exposure information through content (e.g. a cover image is a video frame).
- **Model (late fusion):** two branches Y_{u,i} (full item features) and Y_{u,e} (exposure features only), fused by MUL-sigmoid: Y_{u,i,e} = Y_{u,i} · σ(Y_{u,e}). Training is multi-task: l(Y_{u,i,e}, Ȳ) + α · l(Y_{u,e}, Ȳ) over clicks.
- **CR inference:** Y_CR = Y_{u,i,e} − c_u · σ(Y_{u,e}), where c_u = mean of Y_{u,i} over all items for user u (reference value).
- Backbone: MMGCN (multi-modal GCN), BPR loss, lr 0.001, 64-d latent features. α tuned over {0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5}; best at 1 or 2.

## Assumptions held fixed
- Item features split cleanly into pre-click exposure features and post-click content features; the split is chosen by the authors (TikTok: caption = exposure; audio/visual = content. Adressa: title + description = exposure; body = content).
- Exposure/selection bias and popularity are not modelled; which items were shown is taken as given. Negatives are uniformly sampled non-interacted items.
- "Satisfaction" is operationalised as a like proxy (TikTok: thumbs-up, favourite or finish; Adressa: dwell time > 30 s), used only for testing.
- Test set is drawn from the logged, policy-biased data; the authors explicitly acknowledge items with attractive exposure features are over-represented in it, and that a fully unbiased test set would require external intervention, "left to future work".
- No duration modelling; dwell time is only used to define the Adressa like label.

## Experimental setup
- **Datasets** (Table 1): TikTok (ICME Challenge 2019 micro-video, multi-modal features): 18,855 users, 34,756 items, 1,493,532 clicks, 589,008 likes. Adressa (news): 31,123 users, 4,895 items, 1,437,540 clicks, 998,612 likes. Clicks ending in likes: 39.44% (TikTok) vs. 69.47% (Adressa). Over 60% of TikTok items have like/click ratio < 0.5 (Fig. 4).
- **Split:** per user, 10% of liked clicks → test; 10% of the rest → validation. All-ranking evaluation, P/R/NDCG@{10,20}.
- **Baselines:** NT (normal training), CFT (content features only), IPW (Liang et al. exposure model), and post-click-aware CT (train on liked clicks only), NR (negative-feedback reweighting, Wen et al.), RR (re-rank NT top-20 by like/click ratio).
- Code: https://github.com/WenjieWWJ/Clickbait/

## Key results
- Table 2 (NDCG@10): TikTok NT 0.0333 → CR 0.0370 (+11.11%); Adressa 0.0817 → 0.0878 (+7.47%). R@20 TikTok 0.0635 → 0.0683; Adressa 0.1612 → 0.1712. CR beats all baselines, including RR which uses post-click data.
- IPW is worse than NT on both datasets (TikTok N@10 0.0314; Adressa 0.0663); the authors attribute this to propensity estimation difficulty and argue clickbait is not solved by discouraging high-click items.
- CT (training on liked clicks only) hurts TikTok (N@10 −11.71%) due to sparse likes.
- Table 3: NDE under T = t* beats T = t (TikTok N@20 0.0476 vs. 0.0465).
- Table 4: training on the new graph alone (CR-TE) already improves (TikTok N@20 0.0461), CR inference adds more (0.0476).
- Fig. 6: CR recommends items with low like/click ratio less often than NT (qualitative histogram).

## Reported limitations and trade-offs
- Sparse post-click feedback limits test-set scale; test set is biased toward attractive-exposure items; no unbiased test set.
- Choice of reference situation matters (leakage via content features).
- Requires item-level exposure vs. content features; the method is not applicable to ID-only models without them [INFERENCE from the framework's requirement of e as a separate input].
- Fusion function matters: non-linearity is "essential" (Table 5, compared SUM-linear/sigmoid/tanh, MUL-tanh).

## Relevance to this project
- **§6 Clickbait (secondary):** supplies the conceptual definition — high click but low post-click satisfaction — and the like/click-ratio item statistic; the spec's "high click rate but low watch ratio or high `is_hate`" metric is a KuaiRand analogue of the paper's like/click ratio (Fig. 4/Fig. 6).
- **§4 Labels:** confirms the click ↔ satisfaction gap motivating the label matrix (click vs. satisfaction rows); CR's own evaluation is on a like-proxy label while training on clicks — the cross-label design in §4 generalises this.
- **§10 / §2:** the paper explicitly names the lack of an intervention-based unbiased test set as a limitation; KuaiRand's random log is exactly that intervention, strengthening the project's evaluation rationale.
- **H2 / §5 Exposure row:** IPW underperforming naive on a satisfaction label is prior evidence that exposure correction can lose accuracy on off-target labels.
- Not evaluated on KuaiRand. CR is not in the §5 method table; adding it would need KuaiRand's video features split into "exposure" (cover/caption not provided as raw content) vs. "content" — likely infeasible [INFERENCE].
