# Support the Underground: Characteristics of Beyond-Mainstream Music Listeners

- **Authors:** Dominik Kowald, Peter Muellner, Eva Zangerle, Christine Bauer, Markus Schedl, Elisabeth Lex
- **Venue / year:** EPJ Data Science 10, 14 (2021)
- **Link:** https://doi.org/10.1140/epjds/s13688-021-00268-9 (preprint: https://arxiv.org/abs/2102.12188)
- **Local PDF:** `docs/literature/papers/kowald2021underground.pdf` (arXiv v1)
- **Source basis:** full text (arXiv preprint)

## Problem
Collaborative filtering serves mainstream music listeners better than listeners of non-popular music. The paper characterises beyond-mainstream (BeyMS) listeners and their music, splits them into subgroups, and relates subgroup properties (openness, diversity) to recommendation accuracy. It takes a user-side view of popularity bias rather than the usual item-side long-tail view.

## Method
- **Data.** Starts from LFM-1b (120,322 Last.fm users, >1.1B listening events, 2005–2014) and its CultMRS subset. Tracks are matched to the Spotify API, which supplies 8 acoustic features: danceability, energy, speechiness, acousticness, instrumentalness, tempo, valence, liveness. This yields 3,478,399 tracks and about 394M listening events from 55,149 users.
- **Mainstreaminess definition (§3.3), the item this project needs.** Mainstreaminess is "generally defined as the overlap between a user's individual listening history and the aggregated listening history of all Last.fm users". The paper uses the **M-global-R-APC** measure from Bauer & Schedl 2019 (ref. [50], PLOS ONE). It is **Kendall's τ rank correlation between the user's vector of artist play counts and the global artist play-count vector** (aggregated over all users). M = mainstreaminess, global = global perspective, R = rank correlation, APC = artist play counts. Kowald et al. **do not define a new measure**. They pick this one out of several defined in [50] because it worked well for LFM-1b recommendation in [50].
- **User filtering.** A Gaussian KDE over listening events per user keeps users with 4,688–14,787 events (gradient ≈ 0), leaving 12,814 users.
- **BeyMS threshold.** The cutoff is set at the first inflection point of the mainstreaminess distribution, by Gaussian analogy the 15.9th percentile: M ≤ 0.097732. This gives 2,074 BeyMS users (16.19%). A matched MS group is 2,074 random users above the threshold.
- **Genre filtering.** Last.fm tags are filtered by the EveryNoise taxonomy (3,034 genres). Six coarse genres with IDF < 0.90 are dropped, leaving 157,444 BeyMS tracks.
- **Clustering.** Tracks are embedded with UMAP (Euclidean, D=2, d_min=0.1, N=15) and clustered with HDBSCAN* (s_min = N_min = 1,375). This gives 4 clusters (C_folk, C_hard, C_ambi, C_elec). Users are assigned to clusters by IDF-weighted listening counts w_{u,c} = IDF(c)·LE_{u,c}.
- **Recommendation.** Rating prediction via Surprise on per-user min-max-scaled play counts in [1, 1000]. Algorithms: UserItemAvg, UserKNN (k=40, cosine), UserKNNAvg, NMF (15 factors). Default hyperparameters, 5-fold 80/20 CV, MAE.

## Assumptions held fixed
- **Mainstreaminess is a property of the user's own history relative to the global aggregate, at the artist level.** It is rank-based, so it ignores play-count magnitudes. It is computed on the platform's logged consumption, which is itself shaped by past recommendation.
- The global reference is the whole dataset (the "average" Last.fm listener). Country-specific variants in [50] are not used.
- Users with extreme activity are excluded, so activity is held roughly constant while mainstreaminess varies.
- Hyperparameters are not tuned, so the comparison is between groups, not algorithms.
- Play counts stand in for preference (implicit → pseudo-rating).

## Experimental setup
- **LFM-BeyMS:** BeyMS has 2,074 users, 157,444 tracks and 4,916,174 LEs. The Recommendation subset has 4,148 users, 1,084,922 tracks and 16,687,363 LEs.
- **Metric:** MAE. Tests: one-tailed Mann-Whitney U (α=0.0001) for BeyMS vs MS, and ANOVA (α=0.01) plus Tukey-HSD (α=0.05) across subgroups.

## Key results
- **Table 2 MAE (BeyMS / MS):** UserItemAvg 63.46/61.26; UserKNN 71.67/68.49; UserKNNAvg 67.58/63.40; NMF 57.77/54.82. All algorithms are significantly worse for BeyMS. NMF is best overall.
- **Subgroups (Table 3):** U_hard 919 users, U_elec 642, U_folk 369, U_ambi 143.
- **NMF MAE (Table 4):** U_ambi 54.21, U_elec 56.62, U_folk 57.23, U_hard 59.69. U_hard is significantly worst for all algorithms. U_ambi is better than MS (54.82), but not significantly.
- U_ambi is the most open but least diverse subgroup, and U_hard the least open but most diverse. Openness relates more to accuracy than within-group diversity does, consistent with Tintarev et al.
- Beyond-mainstream users are heterogeneous: "low mainstreaminess" is not a single disadvantaged group.

## Reported limitations and trade-offs
- The sample comes from Last.fm only; generalisation to Spotify or other platforms is untested.
- Only traditional CF is used, no deep models.
- Openness and diversity lack formal definitions.
- Genre filtering keeps on average 61% of each listening history.
- The authors warn of a specialisation–generalisation trade-off: subgroup-specific models could treat other users unfairly.

## Relevance to this project
- **§6 User groups (mainstreaminess).** The measure the spec attributes to "Kowald et al." originates in Bauer & Schedl 2019 (M-global-R-APC). The KuaiRand analogue: Kendall τ between a user's per-item (or per-author/category) play-count vector and the global play-count vector on the training logs. Then threshold, either at the 15.9th percentile as here or by quantiles.
- **H3 (mainstream users lose accuracy under popularity debiasing).** This paper documents the *baseline* gap: low-mainstream users get worse accuracy. H3 predicts popularity debiasing narrows this gap by hurting MS users. Kowald's MS/BeyMS split and per-group error is the template for §7.4 ("whose accuracy was traded").
- **§10 Threats.** Kowald finds large heterogeneity *within* the low-mainstream group. Reporting only a two-group NDCG gap may hide subgroup effects. Consider tertiles/quintiles of mainstreaminess.
- **Metric caveat.** Their accuracy measure is rating MAE, not ranking metrics, so the magnitudes do not transfer to NDCG@K on KuaiRand.
