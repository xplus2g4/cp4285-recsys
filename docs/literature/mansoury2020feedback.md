# Feedback Loop and Bias Amplification in Recommender Systems

- **Authors:** Masoud Mansoury, Himan Abdollahpouri, Mykola Pechenizkiy, Bamshad Mobasher, Robin Burke
- **Venue / year:** CIKM '20 (short paper), 2020
- **Link:** https://arxiv.org/abs/2007.13019
- **Local PDF:** `docs/literature/papers/mansoury2020feedback.pdf`
- **Source basis:** full text (arXiv v1, 25 Jul 2020)

## Problem
Collaborative filtering propagates, and can intensify, the popularity bias already present in rating data. Users then consume the recommendations and those interactions are logged back, so bias can compound over time. The paper simulates this loop offline and asks two questions:
1. How does popularity bias amplify over iterations?
2. What follows downstream: lower aggregate diversity, drift in how users' tastes are represented, homogenisation, and disparate impact on a minority user group?

## Method
**Iterative offline simulation.** For T iterations:
1. Split the current data D_t into 80% train and 20% test.
2. Train the recommender and produce lists R_t.
3. Each user picks items with acceptance probability prob(i | R_u^t) = e^{α · rank_i}, with α < 0, so higher-ranked items are more likely to be chosen.
4. A rating for the chosen item is produced with an Item Response Theory model, ω = s̄_u + sd(s_u) · s̄_i + η with η ~ N(0,1). It is rounded and clipped to [a, b].
5. New (item, rating) pairs are added only if the item is not already in the profile, giving D_{t+1}.

**Formal argument.** Assume the average popularity of recommendations satisfies PR_t ∝ PD_t + θ_t with θ_t > 0. Adding K interactions raises data popularity by K·θ_t / (|D_t| + K). Hence PR_{t+1} > PR_t, i.e. monotone amplification.

**Metrics:**
- Average popularity of recommended items.
- Aggregate diversity (catalogue coverage).
- Per-user KL divergence of the genre distribution at iteration t from iteration 1 (taste-representation drift).
- KL(G_M ‖ G_F) between the male and female genre distributions (homogenisation).
- KL(G ‖ G_M) and KL(G ‖ G_F) against the initial population distribution.

## Assumptions held fixed
- **User intrinsic preferences are not modelled as changing.** The KLD drift measures the *system's representation* of users, not their true taste.
- **Acceptance depends only on rank position** (exponential decay). Item relevance and quality do not enter the choice rule. Ratings come from a fixed IRT model built from profile and item means.
- **No repeat consumption.** Items already in a profile are never re-added.
- **Recommender hyperparameters are fixed across iterations:** 50 factors for BPR and 50 neighbours for UserKNN, both tuned for precision.
- The amplification proof assumes a constant positive popularity lift θ_t at every iteration. The authors use this to hypothesise that popularity-debiasing methods, which shrink θ_t, would amplify less. They do not test it.

## Experimental setup
- **Data:** MovieLens 1M, with 6,040 users, 1,000,209 ratings, 3,706 movies, density 4.468%, and 18 genres.
- **User groups:** 4,331 males provide 753,769 ratings and 1,709 females provide 246,440. Females are treated as the minority group.
- **Algorithms:** UserKNN, BPR and MostPopular.
- **Iterations:** T = 20.

## Key results
All results are shown in figures (Figs. 2–4); the paper reports no numeric tables.
- **Popularity:** average popularity of recommendations rises over iterations for all three algorithms. BPR has the steepest slope.
- **Coverage:** aggregate diversity falls as popularity concentrates, most visibly for BPR, which starts with relatively high coverage.
- **Taste drift:** each user's deviation from their initial genre distribution grows over time for every algorithm.
- **Homogenisation:** KL(G_M ‖ G_F) "dramatically decreases" over iterations, showing strong homogenisation of the two groups.
- **Which group dominates:** female profiles move toward the initial population distribution, which was dominated by males, while male profiles drift slightly away from it.
- **Disparate impact:** per-user deviation from initial preferences is "significantly higher" for females than males under all algorithms. The loop hurts the minority group more.

## Reported limitations and trade-offs
- The work is a short paper on one dataset, with three simple algorithms and no popularity-debiasing method in the loop (listed as future work).
- The cause of BPR's stronger amplification is left unexplained.
- Grouping is fixed by gender label. Other groupings (profile size, average popularity) are suggested.
- The selection policy is rank-only. Top-1 and random selection are suggested alternatives.
- Repeat consumption is disallowed, which the authors note is unrealistic for music (and, implicitly, short video).

## Relevance to this project
- **§9 (feedback loops).** This is the second cited reference. Its recommend → simulate acceptance → retrain protocol over T = 20 rounds is a direct template for §9's 5–10 rounds. The rank-based acceptance rule e^{α · rank} is a simple fallback if KuaiSim is unusable.
- **§5 Popularity row / H-level claims.** The authors' untested hypothesis, that popularity-debiasing reduces loop amplification, is a concrete question §9 can answer for PDA and the naive baseline.
- **§6 bias panel.** Average recommendation popularity and catalogue coverage map onto the popularity metrics. User-group disparity suggests reporting per-group (e.g. user-activity) breakdowns.
- **Caveat for §9.** Mansoury's loop has no duration dimension. Whether duration or watch-ratio debiasing (D2Q) also compounds over rounds is untested here.
