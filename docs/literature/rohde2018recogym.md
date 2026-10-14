# RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising

- **Authors:** David Rohde, Stephen Bonner, Travis Dunlop, Flavian Vasile, Alexandros Karatzoglou (Criteo AI Labs; Universitat Pompeu Fabra; Telefonica Research)
- **Venue / year:** arXiv preprint, 2018. The PDF uses the RecSys '18 (Vancouver) template. [INFERENCE] It was presented at the RecSys 2018 REVEAL workshop; the PDF does not name the workshop.
- **Link:** https://arxiv.org/abs/1808.00720 (code: https://github.com/criteo-research/reco-gym)
- **Local PDF:** `docs/literature/papers/rohde2018recogym.pdf`
- **Source basis:** full text (arXiv v2, 14 Sep 2018)

## Problem
Practitioners report poor correlation between offline supervised metrics and online performance. In online advertising, a user generates two kinds of interaction:
- **organic:** product views on the retailer site;
- **bandit:** ad impressions and clicks on publisher sites.

IPS-based offline evaluation of new policies suffers from huge propensity weights, and capping those weights introduces bias, when the target policy is far from the logging policy. RecoGym is a simulator that can evaluate arbitrary policies without that variance.

## Method
**User model.** A Markov chain alternates organic e-commerce sessions and bandit publisher sessions, then stops. Ads can only be shown in bandit sessions. A click returns the user to the organic session. Conversion probability is assumed constant once the user is back on the shop.

**Link between organic and bandit behaviour.**
- Organic views: v_{u,p,t} ~ Bernoulli(σ(Λ_{u,p,t})).
- Click-through: Φ_{u,a,t} = f(Λ_{u,p,t} + ε_{u,a,t}), where ε has mean 0 and variance σ_Φ², and f is an increasing calibration function.
- With σ_Φ = 0, organic data alone orders actions correctly. As σ_Φ grows, organic and bandit orderings diverge.

**Configurable properties:**
- the correlation between organic and bandit behaviour;
- the latent cluster dimension of users and items;
- the impact of past ad exposure on CTR (ad fatigue, through the t-index on Φ).

**Gym API.** `reset` creates a synthetic user. `step` takes a recommended product and returns:
- observation: the last organic session, or None during bandit events;
- reward: click or no click;
- done;
- info.

**Baseline agents:**
- Random.
- Logistic: per-(last organic product, action) CTR counts.
- Supervised-Prod2Vec: a sigmoid over a linear user × product representation, with the user represented by their last organic product.

## Assumptions held fixed
- **User preferences are static apart from ad fatigue.** RecSim's authors note that RecoGym "does not allow user state transitions". Sequentiality lives in the agent (exploration, CTR estimation), not in the user.
- Conversion is a constant once the user is on the shop; the agent's goal reduces to raising CTR.
- Organic and bandit preferences share latent cluster structure. The correlation is set by σ_Φ.
- Users are generated independently on `reset`; there is no cross-user dynamic.

## Experimental setup
The paper does not run benchmark experiments. It gives **sanity checks** as expected-performance sketches (Fig. 2) for three agent families:
- Pure Organic: should be invariant to bandit-data volume and degrade as σ_Φ grows.
- Pure Bandit: should be invariant to σ_Φ and improve with bandit volume.
- Combined: should beat both at moderate bandit volume and converge with Pure Bandit in the limit.

No datasets or measured numbers are reported.

## Key results
- The contribution is the environment itself, released as an OpenAI Gym environment, plus the formal organic ↔ bandit link and its expected-behaviour checks.
- The paper reports **no quantitative results**. The figures are conceptual sketches of expected trends.

## Reported limitations and trade-offs
- This is a first release with a heavily simplified user model (constant conversion, a Markov session chain).
- Conclusions depend entirely on the stylised generative model.
- IPS evaluation is motivated as high-variance and biased under capping, but the simulator's realism is not validated against real logs.
- In KuaiSim's comparison (zhao2023kuaisim, Table 6), RecoGym rebuilt on KuaiRand had the lowest click AUC (0.6729) and the lowest DDPG agent reward among the five simulators.

## Relevance to this project
- **§9 (simulator choice).** RecoGym is mainly a negative reference. It is ad/CTR-focused, has no item duration or watch-time signal, and has no user-state transitions, so it cannot carry the duration (D2Q) arm of §9.
- **§5 IPS row / §10 propensity misspecification.** The paper's motivation (exploding propensity weights, and bias from capping when the target policy is far from the logging policy) matches the IPS trade-off the audit should report. Record the clipping threshold and weight variance alongside IPS results.
- **§2 test set rationale.** The organic-vs-bandit split mirrors KuaiRand's standard-vs-random exposure. Using the random log as the test set is the empirical analogue of evaluating on "bandit" data, rather than relying on IPS-reweighted organic data.
