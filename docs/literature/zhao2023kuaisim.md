# KuaiSim: A Comprehensive Simulator for Recommender Systems

- **Authors:** Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai (City University of Hong Kong; Kuaishou Technology)
- **Venue / year:** NeurIPS 2023 (37th Conference on Neural Information Processing Systems), Datasets & Benchmarks track (as stated in the assignment). The PDF footer says only "NeurIPS 2023", and the track is not stated in the text read.
- **Link:** https://arxiv.org/abs/2309.12645 (code: https://github.com/Applied-Machine-Learning-Lab/KuaiSim)
- **Local PDF:** `docs/literature/papers/zhao2023kuaisim.pdf`
- **Source basis:** full text (arXiv v2, 19 Oct 2023), including appendix A.1

## Problem
Existing RL-recommender simulators have several gaps:
- They simulate a single immediate signal (click).
- They ignore leave and retention behaviour.
- Rule-based simulators are scenario-specific.
- Learned simulators (RL4RS, Virtual-Taobao) sample users from a pretrained generator, adding a further distribution mismatch.
- There is little work on evaluating simulators themselves.

KuaiSim targets three task levels: request-level (list-wise), whole-session (standard MDP) and cross-session (retention).

## Method
The simulator is a response function S: (O_t, A_t) → Y_t with three modules.

**User Immediate Response Module (UIRM).**
- A Transformer encodes the user profile and history to give state s_t.
- A DNN produces per-behaviour attention weights w_t.
- Behaviour likelihood is p(y | s_t) = w_t ⊙ A_t − ρ · item_correlation(A_t). The correlation penalty rewards diverse lists.
- The b-dimensional multi-behaviour feedback Y^(I) is sampled from this likelihood.
- The module is pretrained on log data with per-behaviour BCE.

**Leave module.** A "temper" starts at the maximum session length and falls with the immediate reward. The user leaves when temper ≤ threshold T. The initial temper, decay rate and threshold are hyperparameters, not learned.

**Retention module.**
- Next-day return probability is p_ret = b_u + λ₁ r_t + λ₂ b, combining a personal bias (DNN), a response bias (λ₁ × reward) and a global bias.
- Return time ~ Geometric(p_ret), capped at D = 10 days.

**User sampling.** Initial users and histories are sampled directly from the dataset rather than from a learned generator.

## Assumptions held fixed
- **The pretrained UIRM is frozen during simulation.** Within-session dynamics come only from the appended history H_t. True user preferences (the network weights) do not evolve across rounds, and there is no population-level re-fitting.
- **Leave and retention are hand-specified functional forms** (linear temper decay, a geometric return law) driven by a scalar reward. The geometric form is justified by data analysis (Fig. 2d).
- **"Good recommendations" means low intra-list item correlation plus high UIRM likelihood.** Item quality, duration and popularity are not explicit inputs to the response model as described.
- Each calendar day is treated as one session.

## Experimental setup
- **Data:** KuaiRand-Pure, with 50-core video filtering, giving 27,077 users, 7,551 items, 1,436,609 interactions, 246,738 sessions and 0.70% density. Also ML-1m, with 6,400 users, 3,706 items, 1,000,208 interactions and 16,629 sessions; ratings > 3 count as "like" and the rest as "hate".
- **Signals used from KuaiRand.** Six positive: click, "view time" (appearing as `long_view`, a binary completeness flag, in Fig. 2 and appendix Table 1), like, comment, follow and forward. Two negative: hate and leave.
- **Benchmarked algorithms:**
  - request level: CF, ListCVAE, PRM;
  - whole session: A2C, SA2C, DDPG, TD3, HAC;
  - cross session: CEM, TD3, RLUR.
- **Metrics:** L-reward, coverage, ILD, depth, average and total reward, return day, user retention.
- **Simulator comparison:** RecoGym, RecSim, RL4RS and Virtual-Taobao were rebuilt on KuaiRand, and a DDPG agent was trained in each. Every configuration was averaged over 5 runs.

## Key results
**Request level** (Table 3). ListCVAE has the best max L-reward (4.042) and ILD (0.565). CF has the best average L-reward (2.253). PRM is worst on coverage (27.52).

**Whole session** (Table 4). HAC is best on every metric:

| Metric | HAC |
|---|---|
| Depth | 14.98 |
| Average reward | 0.6895 |
| Total reward | 10.1742 |
| Coverage | 35.70 |
| ILD | 0.9874 |

DDPG is second (total reward 10.0873). A2C is worst (8.3905).

**Cross session** (Table 5):

| Algorithm | Return day ↓ | Retention ↑ |
|---|---|---|
| RLUR | 3.481 | 0.607 |
| TD3 | 3.556 | 0.581 |
| CEM | 3.573 | 0.572 |

**Simulator comparison** (Table 6, DDPG agent):

| Simulator | Depth | Total reward | Click AUC |
|---|---|---|---|
| KuaiSim | 14.86 | 10.081 | 0.7234 |
| VirtualTaobao | 14.45 | 9.570 | 0.6866 |
| RecSim | 14.05 | 9.347 | 0.6842 |
| RL4RS | 14.39 | 9.235 | 0.6929 |
| RecoGym | 13.55 | 7.194 | 0.6729 |

KuaiSim's improvements are significant at p < 0.05 (two-sided t-test).

**Sensitivity.** Best results come at max time step 20 and slate size 20.

**Answers to the assignment's questions:**
- **Is KuaiSim built on KuaiRand?** Yes: "We construct our simulator KuaiSim on the KuaiRand dataset". It uses the KuaiRand-Pure version, and the appendix describes the random-exposure collection (7,583 items, 1,186,059 random interventions, 27,285 users).
- **Does it use the random or the standard log?** The paper does **not** say which KuaiRand-Pure log it trains on. The reported 1,436,609 interactions exceed the 1,186,059 random interventions, so [INFERENCE] it most likely trains on the standard (recommender-exposed) log, or a mix of both.
- **Which user-response signals does it simulate?** Multi-behaviour immediate feedback (click, long_view, like, comment, follow, forward, hate), a binary leave signal, and return time in days. It does **not** simulate continuous play time, watch ratio, or anything conditioned on video duration.

## Reported limitations and trade-offs
- It does not cover explainability, diversity or fairness objectives in the simulator modelling.
- It has been shown only on video-style datasets (KuaiRand, ML-1m).
- The leave and retention modules are heuristic, with hyperparameters tuned to fit marginals.
- Simulator fidelity is measured only by click AUC and agent reward, not by distributional checks on popularity or duration.

## Relevance to this project
- **§9 claim check.** "KuaiSim, which is built on KuaiRand" is confirmed (KuaiRand-Pure). But the paper does not say the UIRM is fitted on the *random* log. [INFERENCE] If it is fitted on the standard log, the simulator inherits the same exposure and popularity bias the project is auditing, so loop results would be confounded by the simulator itself. §9's alternative, "a user-response model fitted on the random log", avoids this.
- **§9 duration arm.** KuaiSim's response space has only a binary `long_view` and no duration-conditioned watch time. To track D2Q and watch-ratio label effects over rounds, it must be extended with a play-time/duration head, or the random-log response model must be used.
- **§6 panel compatibility.** KuaiSim already reports coverage and ILD. Popularity concentration, duration-bucket mass and the exposure metrics from §6 must be computed externally on the recommended lists each round.
- **§10 threats.** The 50-core filter drops long-tail videos (7,583 → 7,551 items). Simulated popularity metrics will therefore understate tail effects relative to the full random-log pool.
