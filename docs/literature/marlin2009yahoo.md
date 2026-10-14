# Collaborative Prediction and Ranking with Non-Random Missing Data

- **Authors:** Benjamin M. Marlin (UBC), Richard S. Zemel (University of Toronto)
- **Venue / year:** ACM RecSys 2009, pp. 5–12
- **Link:** https://doi.org/10.1145/1639714.1639717
- **Local PDF:** `docs/literature/papers/marlin2009yahoo.pdf` (author-hosted copy, cs.toronto.edu/~zemel)
- **Source basis:** full text

## Problem
Collaborative filtering almost always assumes ratings are Missing At Random (MAR). Under Little & Rubin's theory, a wrong missing-data assumption biases parameter estimates and predictions. An earlier study by the authors (UAI 2007, 35,000+ Yahoo! participants) found evidence of MAR violations. This paper gives the first study of non-random missing data for *ranking*, and extends rating-prediction results with matrix factorisation and kNN baselines.

## Method
- **Factorisation:** P(R,X,Z|μ,θ) = P(R|X,Z,μ) · P(X,Z|θ), i.e. a missing-data model times a data model.
- **Data model:** a multinomial mixture (MM) with K user clusters and Dirichlet priors.
- **Missing-data models**, combined with MM and fit jointly by generalised EM:
  - **MM/MAR:** the missing-data model is ignored.
  - **CPT-v:** P(r=1|x=v) = μ_v. Observation depends only on the rating value, and is the same for all items. Beta priors, closed-form M-step.
  - **Logit-vd:** P(r=1|x=v) = σ(σ_v + ω_d). It adds a per-item effect ω_d explicitly meant for unequal item exposure or promotion. Gaussian priors, gradient M-step with line search.
- **Baselines:**
  - Probabilistic MF with a global mean, ranks K ∈ {1,5,10,20} and λ ∈ {0.1,1,5,10}.
  - Item kNN with adjusted cosine, positive similarities only.

## Assumptions held fixed
- Response indicators are independent across items given the ratings. Missingness depends only on the item's own rating value (CPT-v), or on the rating value plus a static item offset (Logit-vd).
- No user or item features are used, and there is no dependence on other users or on time.
- Users fall into discrete latent clusters (MM), and ratings are discrete on a 1–5 scale.
- The survey's "randomly selected" songs are treated as an approximately random sample of unrated items, justified by their small overlap with user-selected items.

## Experimental setup
- **Data: the Yahoo! Music "Ratings for User-Selected and Randomly Selected Songs, v1.0"** (Webscope; later called Yahoo! R3).
  - Ratings come from LaunchCast radio (user-selected, X^u) and from an online survey (randomly selected, X^r).
  - 15,400 users, each with ≥ 10 user-selected ratings. 5,400 of them have exactly 10 random-song ratings.
  - 1,000 songs sampled at random from the catalogue.
  - About 250,000 user-selected ratings and exactly 54,000 random ratings.
- **Protocol:**
  - Each user is filtered to ≥ 11 user-selected ratings. 10 user-selected items are held out (X^hu), and all random ratings are held out.
  - Five-fold cross-validation over survey users gives **weak** generalisation (training users) and **strong** generalisation (new users).
  - Models are trained only on user-selected ratings and tested on both held-out user-selected and random items.
- **Metrics:** NMAE (MAE / 1.6) and NDCG@L over held-out lists of 10 items.

## Key results
- **Rating distributions (Fig. 2):** random-song ratings contain far fewer high ratings than user-selected ones, which is strong evidence of MAR violation.
- **Rating prediction on random items (Fig. 3):** MM/Logit-vd is slightly better than MM/CPT-v. Both are significantly better than iKNN, MM and MF, which assume MAR.
- **Rating prediction on user-selected items:** plain MM significantly *outperforms* CPT-v and Logit-vd. The method ranking reverses depending on the test distribution.
- **Ranking (Fig. 4):** on random-item lists, MM/Logit-vd and MM/CPT-v are significantly better than MM and MF. On user-selected lists, MM ties the MNAR models and all three beat MF and iKNN.
- Exact numbers are shown only in figures, so none are quoted here. Schnabel et al. 2016 cite the CPT-v Yahoo results as MAE 0.770 / MSE 1.115.

## Reported limitations and trade-offs
- The missing-data models are deliberately simplistic. They ignore features and other influences on observation.
- Hyperparameters were not updated, and the effect of K is limited because the data are sparse.
- Ranking theory under MNAR is not developed; the paper only argues that score-and-sort passes on the bias.
- The evaluation conclusion depends on the test distribution: MNAR-aware models lose on user-selected test data.

## Relevance to this project
- **§2 / §10:** This is the methodological precedent for our core design: train on policy-biased logs and evaluate on a randomly exposed sample. It also shows that conclusions *reverse* between biased and random test sets. That justifies reporting the KuaiRand standard-log versus random-log contrast and never tuning on the standard-log test.
- **§10 Generalisation:** This is the primary citation for Yahoo! R3, the optional replication dataset. It has 1,000 songs and 10 random ratings per survey user, with no duration, so only the popularity and exposure rows transfer.
- **§5 Exposure (popularity propensity):** Logit-vd's item offset ω_d is an early form of item-level exposure propensity, and CPT-v is the value-based propensity that Schnabel's Naive Bayes estimator reuses. The "popularity propensity" versus "feature-based propensity" contrast has this lineage.
- **§10 Random log is only approximately unbiased:** the paper's caveat (random items ≈ random unrated items, given small overlap) mirrors the restricted KuaiRand random pool. Claims hold relative to that pool.
