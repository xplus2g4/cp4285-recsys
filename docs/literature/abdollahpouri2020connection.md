# The Connection Between Popularity Bias, Calibration, and Fairness in Recommendation

- **Authors:** Himan Abdollahpouri, Masoud Mansoury, Robin Burke, Bamshad Mobasher
- **Venue / year:** ACM RecSys 2020 (Late-Breaking Results track), 6 pages
- **Link:** https://arxiv.org/abs/2008.09273 (DOI https://doi.org/10.1145/3383313.3418487)
- **Local PDF:** `docs/literature/papers/abdollahpouri2020connection.pdf`
- **Source basis:** full text

## Problem
The paper asks whether algorithmic popularity bias explains why some user groups get less calibrated recommendations. In its framing, unfairness means unequal miscalibration across user groups (following Yao & Huang's notion of consistent performance across groups).

## Method
- **Miscalibration (genre-based):** p(c|u) is the share of the user's rated items that have genre c. q(c|u) is the same share in the recommended list. MC(P_u, Q_u) is the **Hellinger distance** H = ‖√P_u − √Q_u‖₂ / √2. Group MC is the mean over users in the group. This is Steck's calibration idea, but measured with Hellinger rather than KL.
- **Popularity lift:** θ(i) is the fraction of users who rated i. GAP_p(G) and GAP_q(G) are the group means of each user's average item popularity in their profile and in their recommendation list. **PL(G) = (GAP_q − GAP_p) / GAP_p.** PL > 0 means amplification, PL < 0 means the list is less popular than the profile, and PL = 0 means no amplification. PL is the same quantity as ΔGAP in the 2019 paper.
- **User groups:** users are sorted by the average popularity of their rated items and split into 10 groups, G1 (most niche) to G10 (most blockbuster).
- The paper is a descriptive audit of standard algorithms. **No debiasing method is evaluated.**

## Assumptions held fixed
- The user's profile distribution (genre mix, average popularity) is the target. Deviation from it counts as miscalibration or lift, with no adjustment for exposure that shaped the profile.
- Genre is the only feature used for calibration. Popularity reaches miscalibration only through the genres attached to popular items.
- The authors only conjecture that popularity is the cause. The link is shown by association (the two extreme groups), not by intervention. They name causal tests as future work.
- Each algorithm is tuned for precision, and accuracy is not equalised across methods.

## Experimental setup
- MovieLens 1M (1,000,209 ratings, ~3,900 movies, 6,040 users, 18 genres). Yahoo Movies core-10 (173,676 ratings, 2,131 movies, 7,012 users, 24 genres).
- Random 80/20 split and top-10 lists, run with librec-auto / LibRec 2.0.
- Algorithms: UserKNN, ItemKNN, SVD++, BMF, Most-popular. Precision on MovieLens: 0.214 / 0.223 / 0.122 / 0.107 / 0.182. On Yahoo: 0.13 / 0.127 / 0.2 / 0.047 / 0.1.

## Key results
- PL falls as group average popularity rises (Fig. 1). Niche groups are amplified the most.
- Table 1 compares G10 and G1:
  - MovieLens ItemKNN: G10 PL 0.458, MC 0.250; G1 PL 12.19, MC 0.418.
  - MovieLens Most-popular: G10 0.563 / 0.277; G1 15.7 / 0.501.
  - MovieLens SVD++: G10 −0.09 / 0.380; G1 7.063 / 0.556.
  - MovieLens BMF: G10 0.086 / 0.396; G1 10.60 / 0.635.
  - Yahoo: G10 PL is negative for every algorithm (−0.25 to −0.871). G1 PL ranges from 1.96 (SVD++) to 15.13 (Most-popular).
- For every algorithm, G1 has both higher PL and higher MC than G10.
- Genre analysis: the most frequent genre (Comedy) is *not* amplified. The genres of the most popular movies are amplified instead: Action, Thriller, Sci-Fi and Adventure on MovieLens, and Action, Adventure, Sci-Fi, Fantasy, Crime and Gangster on Yahoo. Item popularity propagates into feature miscalibration.

## Reported limitations and trade-offs
- The evidence is correlational and limited to the two extreme groups. Causality is left as future work (sampling to control popularity, testing whether mitigation lowers miscalibration).
- The definition only partly separates miscalibration from plain lack of personalisation.
- No accuracy-by-group results and no debiasing experiment. The paper is silent on whether reducing popularity bias would hurt blockbuster/mainstream groups.

## Relevance to this project
- **§6 User-side calibration:** this is direct precedent for measuring calibration per user group along a popularity spectrum. The spec uses KL (Steck) over popularity/duration profiles, while this paper uses Hellinger over genres. Report the chosen divergence explicitly, and consider Hellinger too because it is bounded and defined when q has zeros.
- **§6 Popularity metrics:** popularity lift PL(G) per user group is the group-level form of ARP relative to the profile. It is ready to use with 10 mainstreaminess deciles on KuaiRand.
- **§7 mass-shift / H1 / RQ3:** the "popular item → popular feature" mechanism (item popularity drags its correlated attributes into lists) is the same mechanism this project hypothesises for popularity ↔ duration. Popularity correction should drag duration mass the other way if duration and popularity are correlated.
- **H3:** the paper supports "the biased baseline under-serves niche users". It does **not** support "popularity debiasing makes mainstream users lose accuracy". Note that G10 already has negative PL on Yahoo, so blockbuster users are not always over-served even by the baseline.
