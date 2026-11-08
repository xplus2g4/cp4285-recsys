// One definition per term. Every page uses these names; hover any dotted term to see its definition.
// Keys are the ids used in <span class="term" data-term="KEY">…</span>.
window.GLOSSARY = {
  // ---------- Data ----------
  exposure: { term: "Exposure", group: "Data", def: "One video shown to one user. Each log row is one exposure, whatever the user did with it." },
  standard_log: { term: "Standard log", group: "Data", def: "KuaiRand's normal logs. The platform's own recommender chose which videos to show, so the data inherit that recommender's preferences. All models train on the standard log only." },
  random_log: { term: "Random-exposure log", group: "Data", def: "KuaiRand's log in which videos were inserted at random into users' feeds. Because exposure does not depend on any recommender, it is the closest thing we have to unbiased ground truth. Used only for selection (calibration) and evaluation (test)." },
  calibration: { term: "Calibration split", group: "Data", def: "30% of random-log users. Used for early stopping and to pick each method's hyperparameters." },
  test_split: { term: "Test split", group: "Data", def: "The other 70% of random-log users (19,099 users). Each final model is evaluated on it exactly once." },
  guarded: { term: "Guarded split", group: "Data", def: "Training uses only standard-log rows from 8–21 April, i.e. before the random log starts on 22 April, so training never overlaps the evaluation window." },
  label: { term: "Label", group: "Data", def: "The training or evaluation target of a row. Five labels: click, long view, watch ratio, raw watch time and satisfaction." },
  click: { term: "Click", group: "Data", def: "Label: the user clicked (started) the video." },
  long_view: { term: "Long view", group: "Data", def: "Label: KuaiRand's flag that the user watched the video for a long time relative to its length." },
  watch_ratio: { term: "Watch ratio", group: "Data", def: "Label: play time ÷ video length, capped at 1. Relevant = played to the end. Favours short videos, which are easier to finish." },
  raw_watch_time: { term: "Raw watch time", group: "Data", def: "Label: play time in milliseconds. Favours long videos, which can collect more seconds. Has no yes/no relevance, so only NDCG uses it." },
  satisfaction: { term: "Satisfaction", group: "Data", def: "Label: the user liked, followed, forwarded or commented, and did not mark 'hate'. Very sparse: only about 10% of test users have a positive, so these numbers are noisy." },
  head: { term: "Head item", group: "Data", def: "One of the 20% most-exposed items in the training log (1,469 items). The head receives 76% of all training exposures." },
  tail: { term: "Long-tail item", group: "Data", def: "Any item outside the head (80% of items, 24% of exposures)." },
  pop: { term: "Popularity", group: "Data", def: "Unless stated otherwise: an item's number of training exposures." },
  duration: { term: "Duration", group: "Data", def: "Video length in seconds. The median pool video is 81 s." },

  // ---------- Biases ----------
  bias: { term: "Bias (in recommendation)", group: "Bias", def: "A systematic gap between what the logged data say and what users would do if every item had an equal chance to be seen and judged." },
  exposure_bias: { term: "Exposure bias", group: "Bias", def: "Users can only react to what they were shown. Items the old recommender showed often collect more positives regardless of how good they are. Also called selection or policy bias." },
  popularity_bias: { term: "Popularity bias", group: "Bias", def: "Popular items are over-represented in the data and in recommendations, more than their quality alone would justify. Here: lists concentrate on head items." },
  duration_bias: { term: "Duration bias", group: "Bias", def: "Watch-time targets depend on video length: raw watch time rewards long videos, watch ratio rewards short ones, independent of interest." },
  conformity: { term: "Conformity effect", group: "Bias", def: "Users click an item partly because it is popular (others clicked it), not only because it matches their interest." },
  coupling: { term: "Bias coupling", group: "Bias", def: "Two biases are coupled when correcting one moves the other. In KuaiRand, the most popular items are shorter than average (median 67 s for the top 200 vs 81 s overall), so popularity and duration corrections can interact." },
  off_target: { term: "Off-target effect", group: "Bias", def: "A change in a bias that a method was not designed to fix. The research is mostly about these." },

  // ---------- Modelling ----------
  recommender_family: { term: "Recommender family", group: "Model", def: "The model class a method plugs into: matrix factorisation (MF), feature-based / CTR models (FM, DeepFM), graph collaborative filtering (e.g. LightGCN), sequential models, or LLM-based recommenders." },
  cf: { term: "Collaborative filtering (CF)", group: "Model", def: "Recommending from the user-item interaction matrix alone: users who behaved alike get similar recommendations. MF and LightGCN are CF models." },
  mf: { term: "Matrix factorisation (MF)", group: "Model", def: "A CF model that gives every user and item a vector (embedding). Score = global bias + user bias + item bias + dot product of the two vectors." },
  backbone: { term: "Backbone", group: "Model", def: "The base model that produces the score f(u, i). Here every method uses the same MF backbone (64-dimensional embeddings), so only the debiasing correction differs." },
  logit: { term: "Logit", group: "Model", def: "The raw score before the sigmoid; σ(logit) is the predicted probability." },
  bce: { term: "BCE", group: "Model", def: "Binary cross-entropy: the standard loss for predicting a 0/1 label (or a target in [0, 1]) from a logit." },
  mse: { term: "MSE", group: "Model", def: "Mean squared error: the standard regression loss." },
  fm: { term: "Factorization machine (FM)", group: "Model", def: "A model over categorical features: a weight per feature value plus a dot product for every pair of features. Used here only to estimate the IPS-feat propensity." },
  naive_mf: { term: "Naive MF", group: "Method", def: "The baseline: MF trained with BCE on clicks, with no debiasing. Every click-trained method is compared against it." },
  vr: { term: "VR (value regression)", group: "Method", def: "The baseline for D2Q: MF trained with MSE to predict raw watch time directly. This is the naive watch-time model the D2Q paper compares against." },
  knob: { term: "Knob", group: "Protocol", def: "The one method-specific hyperparameter we sweep to trace a trade-off curve (M for D2Q, c for MACR, clip for IPS-feat). It is not tuned for accuracy; every value is reported." },
  sweep: { term: "Sweep", group: "Protocol", def: "One knob value: tune lr × weight decay on a 9-point grid using the calibration split, then retrain the best point with 5 seeds and evaluate each on the test split once." },
  seed_std: { term: "Seed std", group: "Statistics", def: "Standard deviation of a metric over the 5 final seeds. It measures training instability." },
  delta: { term: "Δ (delta)", group: "Statistics", def: "Method mean minus baseline mean, both averaged over 5 seeds and computed on the same test users." },
  boot_ci: { term: "Paired bootstrap CI", group: "Statistics", def: "95% interval for Δ from 1,000 resamples of test users, using the same resampled users for method and baseline. It covers user sampling only, not seed variance." },
  sig: { term: "Significance marker (*)", group: "Statistics", def: "The paired bootstrap CI of Δ excludes zero. No correction for multiple comparisons is applied, and seed variance is not included, so read it together with the seed std." },
  pareto: { term: "Pareto frontier", group: "Statistics", def: "The set of settings that no other setting beats on every axis at once. A setting off the frontier is 'dominated'." },

  // ---------- Metrics: accuracy ----------
  gauc: { term: "GAUC", group: "Metric · accuracy", def: "Group AUC. For each test user, the AUC of the model's scores over that user's random-log candidates (probability a positive is ranked above a negative); then averaged over users, weighted by candidate count. 0.5 = random." },
  ndcg: { term: "NDCG@10", group: "Metric · accuracy", def: "Normalised discounted cumulative gain of the top 10 when re-ranking each user's random-log candidates. 1 = ideal order. Gains are the label values." },

  // ---------- Metrics: popularity / exposure / duration ----------
  topk: { term: "Top-10 list", group: "Metric · lists", def: "The 10 highest-scoring items for a user over the whole 7,344-item pool, excluding nothing. All list metrics below use it." },
  arp: { term: "ARP@10", group: "Metric · popularity", def: "Average recommendation popularity: mean training-exposure count of the items in a top-10 list. Lower = less popular items. Naive MF: 1,998." },
  gini: { term: "Gini@10", group: "Metric · exposure", def: "Gini coefficient of how often each item appears in any top-10 list. 1 = all recommendations on one item. Naive MF: 0.996." },
  w1: { term: "W1 (duration)", group: "Metric · duration", def: "Wasserstein distance (in seconds) between the duration distribution of recommended items and of items the users actually liked in the random log. 0 = same shape." },

  // ---------- Causal / IPS vocabulary ----------
  confounder: { term: "Confounder", group: "Causal", def: "A variable that affects both the cause and the outcome. For D2Q: video duration affects both which videos the model prefers and how long users watch them." },
  counterfactual: { term: "Counterfactual inference", group: "Causal", def: "Asking 'what would the score be if one cause were switched off?' and ranking by the difference." },
  total_effect: { term: "Total effect", group: "Causal", def: "The full prediction, including every path from user and item to the outcome." },
  direct_effect: { term: "Natural direct effect (NDE)", group: "Causal", def: "In MACR: the part of the prediction that comes from the item (or user) alone, not from user-item matching. Mostly popularity." },
  propensity: { term: "Propensity", group: "IPS", def: "The probability that a row was observed (here: that the item was exposed to the user). IPS-feat estimates it from user and item features." },
  ipw: { term: "Inverse propensity weight", group: "IPS", def: "Weight 1 ÷ propensity on each training row. Rarely exposed items get large weights, so the weighted loss approximates the loss under uniform exposure." },
  clip: { term: "Clip", group: "IPS", def: "Cap on the weights, written as the largest allowed ratio between any two weights (the smallest weight is 1). Smaller clip = less correction, less variance. 'inf' = no cap." },
  ess: { term: "ESS share", group: "IPS", def: "Effective sample size share: (Σw)² ÷ (N·Σw²). 1 = all rows count equally; 0.1 = the weighted data carry the information of about 10% of the rows." },
  ecdf: { term: "ECDF / quantile", group: "Duration", def: "Empirical cumulative distribution: the share of values ≤ a given value. D2Q replaces watch time by its quantile within videos of similar length (0 = least watched, 1 = most)." },
  duration_group: { term: "Duration group", group: "Duration", def: "D2Q splits training rows into M equal-size groups by video duration. M is D2Q's knob; M = 1 means no duration adjustment." },
};

// Standard method names, colours and families, shared by every page.
window.METHODS = {
  naive_mf: { name: "Naive MF", color: "#6b7280", family: "Baseline" },
  vr: { name: "VR", color: "#374151", family: "Baseline (watch time)" },
  d2q: { name: "D2Q", color: "#c2410c", family: "Duration · label" },
  d2q_q: { name: "D2Q-Q", color: "#f59e0b", family: "Duration · label" },
  ips_feat: { name: "IPS-feat", color: "#e11d48", family: "Exposure · reweighting" },
  macr: { name: "MACR", color: "#9333ea", family: "Popularity · causal" },
};
