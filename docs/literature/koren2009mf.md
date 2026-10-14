# Matrix Factorization Techniques for Recommender Systems

- **Authors:** Yehuda Koren, Robert Bell, Chris Volinsky
- **Venue / year:** IEEE Computer, vol. 42, no. 8, pp. 30–37, August 2009
- **Link:** https://doi.org/10.1109/MC.2009.263
- **Local PDF:** not available. The paper is paywalled at IEEE Xplore, Semantic Scholar lists `openAccessPdf.status = CLOSED`, and no author-hosted or licensed open copy was found. Third-party mirrors (e.g. SciSpace) were not used.
- **Source basis:** abstract + IEEE Xplore page (abstract, section outline, opening paragraph), plus Crossref/Semantic Scholar metadata. The model description below follows the section titles and the standard published formulation of this article. Statements marked [INFERENCE] were not checked against the full text in this session.

## Problem
The paper asks how to predict user–item preferences (e.g. Netflix movie ratings) from sparse rating data. The IEEE page frames recommenders as either content filtering, which needs external profiles that are often unavailable, or collaborative filtering, which uses only past behaviour. Abstract: "As the Netflix Prize competition has demonstrated, matrix factorization models are superior to classic nearest neighbor techniques for producing product recommendations, allowing the incorporation of additional information such as implicit feedback, temporal effects, and confidence levels."

## Method
Section outline (IEEE page): Recommender System Strategies; Matrix Factorization Methods; A Basic Matrix Factorization Model; Learning Algorithms; Adding Biases; and further sections, which the outline truncates.

[INFERENCE, from the standard formulation of this article:]
- **Basic model:** r̂_ui = q_iᵀ p_u, with user and item vectors in a joint latent space of dimension f. It is fit only on observed ratings with L2 regularisation, λ(‖q_i‖² + ‖p_u‖²).
- **Learning:** SGD or alternating least squares.
- **Biases:** "Adding Biases" extends the model to r̂_ui = μ + b_u + b_i + q_iᵀ p_u. The global mean μ and the user/item offsets b_u, b_i absorb rating variation that has nothing to do with the interaction, including popular items being rated higher.
- **Extensions:** implicit feedback, temporal dynamics (time-varying biases and user factors), and confidence weights.

## Assumptions held fixed
- The model is fit only on **observed** entries. Missing ratings are implicitly treated as ignorable (MAR), so no exposure or selection mechanism is modelled [INFERENCE].
- Item-bias terms absorb popularity as a *signal*, not a bias to remove. Popularity correction (PDA, MACR) or IPS has to be layered on top.
- Preference is low-rank and stationary unless the temporal extension is used.

## Experimental setup
Netflix Prize data (per the abstract framing). The detailed RMSE comparisons in the article were not read in this session, so no numbers are reported here.

## Key results
From the abstract: MF outperforms classic nearest-neighbour methods and can absorb implicit feedback, temporal effects and confidence levels. No numerical results are reproduced because the full text was not available.

## Reported limitations and trade-offs
Not verifiable without the full text. [INFERENCE] Plain MF fit on observed data inherits the logging policy's exposure and selection bias. That is why the debiasing literature (IPS, DR, PDA) wraps MF rather than replacing it.

## Relevance to this project
- **§5 Backbone:** MF is the primary backbone. Use the biased form (μ + b_u + b_i + q_iᵀp_u) for every method, with one fixed embedding size f shared with DeepFM.
- **§5 Popularity row:** the item bias b_i absorbs popularity. PDA's popularity term and MACR's item branch act alongside or instead of b_i, so whether b_i is kept should be fixed across methods and reported.
- **§5 Exposure row:** IPS, SNIPS and DR reweight MF's per-observation loss. Because MF is a pointwise squared or BCE loss, the weighting is straightforward.
- **§10 Threats:** fitting only on observed entries assumes MAR. Evaluating on the KuaiRand random log is exactly what tests that assumption.
