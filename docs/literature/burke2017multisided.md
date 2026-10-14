# Multisided Fairness for Recommendation

- **Authors:** Robin Burke (DePaul University)
- **Venue / year:** Poster, Workshop on Fairness, Accountability, and Transparency in Machine Learning (FAT/ML) 2017
- **Link:** https://arxiv.org/abs/1707.00093
- **Local PDF:** `docs/literature/papers/burke2017multisided.pdf` (arXiv v2)
- **Source basis:** full text

## Problem
Fairness work in ML usually removes bias with respect to a protected attribute in classification. Recommendation differs in two ways. (1) Personalization: there is no shared global ranking, because the best items differ per user. (2) Recommenders mediate transactions among several parties (consumers, providers, platform), so fairness may matter on more than one side. The paper gives a taxonomy and sketches architectures. It is a 4-page position paper with no experiments.

## Method
**Stakeholders.**
- Consumers C receive the recommendations.
- Providers P supply or stand behind the items.
- The platform/system S matches them.

The system's utility is what is ultimately maximised. It is a function of the other stakeholders' utilities plus domain-specific terms: transaction share, customer satisfaction, legal compliance and social mission.

**Taxonomy of fairness.**
- **C-fairness:** disparate impact on protected classes of *consumers*. Example: credit-card offers from a single bank. Suggested design: map users to a prototype space, e.g. latent factors, engineered for statistical parity w.r.t. the protected class (after Zemel et al. 2013), "with bounded loss with respect to ranking accuracy".
- **P-fairness:** fairness for *providers* only. Examples: Kiva.org loans, and new Etsy sellers getting a "reasonable share" of recommendations. Providers are passive and must wait for recommendation opportunities. Two levels are distinguished:
  - *List diversity* (group-level). Achievable by re-purposing diversity-aware re-ranking, but satisfiable by showing the same "diverse" items to everyone.
  - *Catalog coverage* (individual-level fairness). Needs a dynamic mechanism. Burke proposes budgeted second-price auctions over recommendation opportunities with a *purchasing parity* rule: give each of p protected providers B/(2p) and each of q other providers B/(2q) of a total budget B (p ≪ q). Bids are proportional to remaining budget and candidate quality and inversely proportional to future opportunities, similar to the BALANCE algorithm.
- **CP-fairness:** both sides, e.g. reciprocal recommendation, employment, rental housing. Proposed solutions are decoupled: a C-fair ranking feeds the P-fair bidding mechanism. How combining them affects each stakeholder is left open.

## Assumptions held fixed
- Fairness is defined relative to **protected classes** (race, gender, minority ownership) chosen by the operator/regulator. Popularity, duration and exposure are not discussed as fairness attributes.
- The relevance/personalization model is taken as given. Fairness interventions bound the loss in personalization accuracy rather than changing what "relevance" means.
- In the decoupled CP design, each side's correction assumes the other side's mechanism stays fixed. The interaction between them is explicitly not analysed.
- Stakeholder utilities are assumed formalizable and comparable enough to fold into one system utility.

## Experimental setup
None. Conceptual/position paper with illustrative domains: job recommendation, consumer credit, Kiva, Etsy, rental housing, online dating.

## Key results
- The C / P / CP taxonomy, with P-fairness split into list-level (group) and catalog-level (individual) variants.
- The observation that list diversity ≠ catalog coverage: group fairness at the list level does not guarantee individual provider fairness.
- An auction/budget architecture with purchasing parity B/(2p) vs B/(2q) for individual P-fairness.
- No numerical results.

## Reported limitations and trade-offs
- There is a tension between the personalization view (obey user preference) and the regulatory view (value particular outcomes).
- Every fairness intervention costs personalization; the loss "must be bounded".
- Multistakeholder settings are domain-specific, so suitable datasets are hard to find and results hard to generalise.
- How combined C-fair and P-fair solutions affect each stakeholder and overall system performance is an open question.

## Relevance to this project
- **H3 / §6 User groups and Popularity panels.** Burke's split maps onto the project's two-sided reading of popularity debiasing. Item-side popularity metrics (ARP, Gini, APLT, PopRSP/REO) are P-side (provider/item exposure) outcomes. The user-group NDCG gaps (mainstreaminess, activity) are C-side outcomes. H3 is in effect a claim that a P-fairness intervention creates a C-fairness cost, which Burke flags as the open question for CP settings.
- **§6 Exposure/coverage.** Burke's list-diversity vs catalog-coverage distinction supports reporting both list-level APLT and catalogue-level coverage@K and exposure entropy. Improving one does not imply the other.
- **RQ2 framing.** The decoupled-solution caveat ("how outcomes for each stakeholder … are affected by combining such solutions") is a conceptual precursor of the project's off-diagonal audit.
- Background/framing citation only. It supplies no metric or method to implement.
