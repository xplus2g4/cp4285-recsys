# Multistakeholder Recommendation: Survey and Research Directions

- **Authors:** Himan Abdollahpouri, Gediminas Adomavicius, Robin Burke, Ido Guy, Dietmar Jannach, Toshihiro Kamishima, Jan Krasnodebski, Luiz Pizzato
- **Venue / year:** User Modeling and User-Adapted Interaction 30(1):127–158, 2020
- **Link:** https://doi.org/10.1007/s11257-019-09256-1 (preprint: https://arxiv.org/abs/1905.01986, titled "Beyond Personalization: Research Directions in Multistakeholder Recommendation")
- **Local PDF:** `docs/literature/papers/abdollahpouri2020multistakeholder.pdf` (arXiv v2, 15 Dec 2019; the journal version is behind a paywall)
- **Source basis:** full text of the arXiv preprint (64 pages). The published UMUAI version may differ in wording and numbering.

## Problem
Academic recommender research optimises for the end user alone. Real systems serve several parties: users, item providers and the platform. Properties such as fairness, balance, profitability and reciprocity are invisible to standard accuracy metrics. The survey offers multistakeholder recommendation as a unifying framework. It traces its origins (economics of multisided platforms, multi-objective recommendation, matching), gives a design taxonomy, works through three examples, and lists methodological issues and research directions.

## Method
**Definitions (§2.1).** A recommender is a scoring function f(u,i,R) → ℝ. In a *user-oriented* system f preserves the order of user preferences: r_ui ≥ r_uj ⇒ f(u,i,R) ≥ f(u,j,R) (Eq. 2). In a *multistakeholder* system this order-preservation need not hold, and f is read as a balance of several parties' interests.

**Taxonomy.** Each design is a triple ⟨C,P,S⟩:
- Consumers can be passive C⁻ or active C⁺ (query-driven).
- Providers are passive/active and neutral (P_n) or personalized (P_p). Table 1 covers standard recommendation, reciprocal, paid placement, display and search advertising.
- The system is neutral (S_n), aggregate (S_a, utility = sum of the other parties' utilities, e.g. commission) or targeted (S_t, its own objective).

**Fairness is presented as a quintessential S_t concern.** Most fairness research has the form ⟨C_p, P_n, S_t⟩.

**Examples.**
- **(§5) People/reciprocal recommendation:** dating, jobs, social networks.
- **(§6) Value-aware recommendation:** profit-aware re-ranking. Expedia case: a learning-to-re-rank model that optimises margin subject to staying close to the relevance ranking, measured with Kendall τ.
- **(§7, Kamishima) Fairness-aware recommendation.** Two senses: fairness in the *process*, i.e. sensitive information S is not exploited, with conditions such as Ŷ ⊥ S | X, Ŷ ⊥ S and Ŷ ⊥ S | Y; and fairness of *resource allocation*.

**Evaluation (§8.5, Table 4).** Provider metrics:
- Exposure(p) = Σ_L Σ_j 1(j∈I_p)
- Hits(p) counts hits in the test set
- Reach(p) = number of users with at least one of p's items
- TargetReach(p)
- PAccuracy(p,m) = average of metric m over p's test items

A system-level proportional-impact metric Pr[Ŷ=1|S=0]/Pr[Ŷ=1|S=1] → 1. The authors recommend reporting **mean, variance and skewness** of each stakeholder's utility distribution, not just the mean.

## Assumptions held fixed
- In the ⟨C,P,S⟩ framing, each stakeholder's utility is assumed measurable from offline data (hits, exposure, reach). The survey itself notes this requires business-critical data that is rarely shareable.
- Fairness-in-process definitions assume a fixed, known sensitive attribute S. Independence conditions are imposed on predictions while the rest of the data distribution is held fixed.
- The re-ranking designs (value-aware LRR, MMR, xQuAD) hold the base relevance model fixed and adjust only the final list. Multi-objective designs change the loss but keep the accuracy term as the C_p anchor.
- Offline evaluation with random fixed-size partitions of user profiles is the default; the authors question it (§8.3).

## Experimental setup
Survey with no unified experiments. The case study in §6.3 uses Expedia hotel searches from 2016, a matrix-factorization base model, NDCG@10 on clicks+bookings, and commission %. It compares a production value-aware model (tuned by A/B tests) with LRR and a relevance-only baseline.

## Key results
- The taxonomy (Table 1, Fig. 2) and stakeholder notation.
- Expedia (Table 3): both value-aware approaches gave about +16.7% commission. LRR lost less relevance than the production value-aware model (−5.9% vs −8.7% NDCG@10).
- The profit-aware studies reviewed suggest substantially higher short-term business value without a considerable loss in recommendation quality or trust. The authors flag that long-term effects are less clear.
- Provider metric definitions (Table 4) and the moments-of-utility reporting practice.

## Reported limitations and trade-offs
- Little agreement on evaluation methods.
- Real multistakeholder data (margins, commissions) is proprietary, so research leans on simulation (e.g. randomly assigning items to providers, with normal or power-law utilities).
- Utility models across stakeholders are hard to compare.
- Standard offline splits may not reflect multistakeholder impact.
- User studies are rare.
- Multi-objective constraint approaches scale exponentially with the number of constraints.
- Explanations become harder when user preferences are traded off against other parties.

## Relevance to this project
- **§6 Exposure/coverage and Popularity panels.** Table 4's provider metrics give formal definitions to borrow. Exposure(p) and Reach(p) at the item level (or KuaiRand author level) underpin catalogue coverage@K, exposure entropy and Gini.
- **§8 Statistical protocol.** The suggestion to report mean, variance and skewness of per-stakeholder utility fits the per-user and per-item distributions already produced for the bootstrap CIs. It is cheap to add for the user-group NDCG and item-exposure distributions.
- **H3 / §7.4.** The survey frames fairness as an S_t objective layered on a C_p accuracy anchor. That matches the project's view of popularity debiasing as a system objective whose cost is paid by particular consumer groups. Use it as framing for "whose accuracy was traded".
- **§5 interventions.** The survey's split into in-loss multi-objective methods and post-hoc re-ranking maps onto the project's methods. In-training: IPS, PDA, D2Q labels. Post-processing: e.g. calibration re-ranking (Steck). The two should be reported as separate classes.
