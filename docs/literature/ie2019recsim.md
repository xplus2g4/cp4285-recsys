# RecSim: A Configurable Simulation Platform for Recommender Systems

- **Authors:** Eugene Ie, Chih-wei Hsu, Martin Mladenov, Vihan Jain, Sanmit Narvekar, Jing Wang, Rui Wu, Craig Boutilier (Google Research; Narvekar at UT Austin)
- **Venue / year:** arXiv preprint (cs.LG), 2019
- **Link:** https://arxiv.org/abs/1909.04847 (code: https://github.com/google-research/recsim)
- **Local PDF:** `docs/literature/papers/ie2019recsim.pdf`
- **Source basis:** full text (arXiv v2, 26 Sep 2019)

## Problem
Static datasets such as MovieLens 1M cannot answer counterfactual questions: they cannot say how a user would have responded to a different action. Logs from myopic recommenders also have action distributions unlike those of long-horizon policies, which limits IPS and other importance weighting. RL for recommendation therefore lacks environments with:
- sequential interaction,
- latent and evolving user state,
- slate actions,
- configurable choice behaviour.

RecSim provides a platform for *authoring* stylised simulation environments. The authors explicitly do not aim for a high-fidelity "sim-to-real" simulator.

## Method
**Environment components:**
- **User model:** a prior over observable, latent and behavioural features such as interests, satisfaction and time budget.
- **Document model:** a prior over document features, both latent (e.g. quality) and observable (e.g. topic, length, popularity).
- **User-choice model:** multinomial logit or exponentiated cascade. Choice depends on observable document features and all user features. Responses such as watch time can depend on latent document features.
- **User transition model:** interest drift, budget decay and satisfaction change.

The whole system is a dynamic Bayesian network over trajectories of user states, slates, choices and observations, wrapped as an OpenAI Gym environment.

**Reference environment (SlateQ).**
- Documents have a topic vector, length ℓ(d) and quality L_d ~ N(µ_T(d), σ²).
- User interest is I(u,d) = u·d, with u ∈ [−1,1]^|T|.
- Satisfaction is S = f(I, L_d), e.g. (1−α)I + αL_d.
- Interest is nudged after consumption.
- Each session has a time budget B_u that falls by ℓ(d) minus a bonus that grows with appeal.

**Agents and tooling.**
- Hierarchical agent layers: ClusterClickStatsLayer, FixedLengthHistoryLayer and TemporalAggregationLayer.
- Baselines: TabularQ, FullSlateQ (DQN), Random, a Dopamine adapter, and bandits (UCB1, KL-UCB, Thompson sampling).
- Episodes are logged as TF SequenceExamples for batch RL.

## Assumptions held fixed
- **Users are independent.** One user's interaction does not affect another's state, so there is no cross-user or population-level feedback in the default design. Users are simulated sequentially; interleaving is planned.
- **Choice model is fixed per environment.** Robustness to choice-model misspecification can be tested by training under one choice model and evaluating under another.
- Document quality is unobservable before consumption but drives transitions. Topic is observable.
- Included environments are illustrations and are "not advocated as benchmarks".

## Experimental setup
Three case studies:
1. **Latent-state bandits.** Single-item recommendation, one-hot topics, log-normal quality, static user interests. Metric: CTR over a session, in low- and high-topic-affinity configurations.
2. **SlateQ slate decomposition** (Ie et al. 2019). Results are referenced, not reproduced.
3. **Advantage amplification** (Mladenov et al. 2019). A "chocolate vs kale" satisfaction dynamic with low signal-to-noise ratio. Results are referenced.

## Key results
Average CTR (%) in the latent-state bandit study. Parentheses give improvement over Random.

| Agent | Low affinity | High affinity |
|---|---|---|
| Random | 7.86 | 14.97 |
| Greedy (omniscient-prior) | 9.59 (+22.01%) | 17.56 (+17.30%) |
| TabularQ | 8.24 (+4.83%) | 20.16 (+34.67%) |
| FullSlateQ | 9.64 (+22.60%) | 23.28 (+55.51%) |
| UCB1 | 9.76 (+24.17%) | 25.17 (+68.14%) |

Qualitative results from the other case studies:
- **SlateQ:** RL with slate decomposition beats myopic policies, and the gain survives choice-model shift.
- **Advantage amplification:** temporal aggregation brings policy quality close to the case where satisfaction is fully observed.

## Reported limitations and trade-offs
- The simulator is not meant to be faithful; stylised models capture only targeted phenomena.
- Users are simulated serially (no concurrent or multi-user interaction).
- Fitting stylised user models to production logs is listed as future work.
- Stylised rule-based dynamics mean conclusions depend on the authored choice and transition models.

## Relevance to this project
- **§9 (simulator choice).** A fallback or contrast to KuaiSim. RecSim's configurable choice model could encode a user-response model fitted on the KuaiRand random log, as §9 allows. Document *length* and *time budget* map naturally onto video duration, which is relevant to the duration (D2Q) arm.
- **§9 limitation to flag.** RecSim's default design has no cross-user coupling. Population-level popularity feedback, the mechanism behind §9's popularity metrics, only appears if retraining on pooled logs is added outside the per-user loop.
- **§10 threats.** Train-under-one-choice-model / evaluate-under-another is a template for testing how sensitive loop results are to simulator misspecification.
- **IPS footnote.** Logs from myopic policies limit importance weighting. This is relevant context for the §5 IPS row when propensities come from a non-random logging policy.
