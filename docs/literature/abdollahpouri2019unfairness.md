# The Unfairness of Popularity Bias in Recommendation

- **Authors:** Himan Abdollahpouri, Masoud Mansoury, Robin Burke, Bamshad Mobasher
- **Venue / year:** RMSE workshop (Recommendation in Multi-stakeholder Environments) at ACM RecSys 2019, CEUR-WS; arXiv 1907.13286
- **Link:** https://arxiv.org/abs/1907.13286
- **Local PDF:** `docs/literature/papers/abdollahpouri2019unfairness.pdf` (arXiv v3, CC BY 4.0)
- **Source basis:** full text

## Problem
Most popularity-bias work takes the item side (coverage, long-tail exposure). This short workshop paper takes the **user** side. It asks how far recommendations drift from the popular/non-popular mix a user expects, given their own profile, and whether some user groups are hurt more than others.

## Method
- An item is "popular" if it is in the top 20% of item popularity. Popularity φ(i) is the number of ratings of i divided by the number of users.
- Users are grouped by the share of popular items in their profile:
  - **Niche (N):** bottom 20%. More than half of their profile is long-tail.
  - **Blockbuster-focused (B):** top 20%, with on average more than 85% popular items.
  - **Diverse (D):** everyone else.
- **Group Average Popularity:** GAP(g) = (1/|g|) Σ_{u∈g} (Σ_{i∈p_u} φ(i)) / |p_u|. It is computed on profiles (GAP_p) and on recommendation lists (GAP_r).
- **ΔGAP(g) = (GAP_r − GAP_p) / GAP_p.** 0 means a fair representation of the user's popularity preference. Positive means the algorithm imposes popularity.
- This is a purely descriptive audit of existing algorithms. **No debiasing method is proposed or evaluated.**

## Assumptions held fixed
- A user's historical profile ratio of popular items is the "expected" ratio, i.e. calibration relative to the profile. The ideal is ΔGAP = 0 regardless of whether the profile itself is shaped by past exposure.
- The popular/non-popular cut is fixed at the top 20%, and the user-group cuts at the 20/60/20 percentiles.
- Algorithms are tuned to similar precision (≈0.1), so popularity propagation is compared at roughly fixed accuracy.
- Only one dimension (item popularity) is studied. Genre, content length and other attributes are ignored.

## Experimental setup
- MovieLens 1M: 1,000,209 ratings, ~3,900 movies, 6,040 users. Random 80/20 split, top-10 lists.
- Algorithms: User KNN, Item KNN, SVD++, Biased MF, Most popular, Random.
- Analyses: profile size vs. number/ratio/average popularity of rated items. Item rating count vs. recommendation count (scatter plots). Profile vs. recommendation popular-item ratio per group. ΔGAP per group.

## Key results
- More than 4,000 users have at least 20% non-popular items in their profile.
- Profile size correlates negatively with the ratio and with the average popularity of rated items. Niche users have the **largest** profiles, then diverse users, and blockbuster-focused users the smallest. Groups N and D together are about 75% of users (D has more than 3,500 users).
- All algorithms except Random raise the share of popular items in recommendations for every group. Most popular has the highest ΔGAP, then Item KNN. Random has negative ΔGAP. SVD++ comes closest to users' expected ratio.
- **Niche users have the largest ΔGAP for every algorithm.** They are the most under-served by popularity bias.

## Reported limitations and trade-offs
- A single dataset (MovieLens 1M). The authors plan more datasets and other satisfaction measures (relevance, diversity) per group.
- Exact ΔGAP numbers appear only in figures, not in the text.
- No accuracy-by-group analysis and no mitigation. It says nothing about what happens to blockbuster/mainstream users when bias is reduced.

## Relevance to this project
- **§6 User-side metrics / H3:** this is the source of the niche / diverse / blockbuster grouping by profile popularity share and of the GAP/ΔGAP metric. ΔGAP per group is a cheap user-side popularity calibration to report next to the spec's KL calibration (Steck).
- **H3 attribution:** the paper shows the *biased baseline* hurts **niche** users most. It does **not** test debiasing and so gives no evidence that debiasing harms mainstream users. H3's "mainstream users losing accuracy" is a hypothesis the project would establish, not a prior finding.
- **§6 grouping by activity:** the finding that niche users are the most active means groups by `user_active_degree` and by mainstreaminess are confounded on KuaiRand too. Check their overlap before reading gaps causally.
- **§6 Duration analogue:** a "ΔGAP over duration" (average duration of recommended items vs. the user's history) is the natural duration counterpart. It complements DurRSP/DurREO.
