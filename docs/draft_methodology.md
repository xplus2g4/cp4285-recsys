# Methodology: trade-offs between debiasing methods on KuaiRand

> Dataset row counts and some citation details were written from memory. Check them against the KuaiRand release and the original papers before submission.

## 1. Research questions

- **RQ1.** When a single-bias correction (popularity, duration or exposure) is measured on randomly exposed data, how much does it reduce the bias it targets?
- **RQ2.** How does each correction change the biases it does *not* target? These are the off-diagonal effects.
- **RQ3.** Is there a duration ↔ popularity coupling in KuaiRand that makes one correction move error into the other dimension?
- **RQ4.** Which methods or settings sit on the Pareto frontier of accuracy against the bias panel, and which are dominated?

**Hypotheses** (each states a measurable direction):

- **H1.** Duration normalisation (D2Q) moves recommendations toward short videos. Because duration and popularity are correlated in the logs, it also shifts popularity concentration. The direction is measured in §3.
- **H2.** IPS-style exposure correction spreads recommendations more evenly over the catalogue (lower Gini). It also raises variance across seeds.
- **H3.** Popularity debiasing lowers ARP at a cost in accuracy, and the cost is larger under click than under long view and watch ratio, because clicks reward popular items most.

## 2. Data and splits

- **Dataset:** KuaiRand-Pure. Use the standard logs (`log_standard_*`), the random-exposure log (`log_random_*`, `is_rand = 1`), and the user and video feature tables.
- **Training:** standard logs only, which carry the policy bias.
- **Random-exposure log:** split by user, about 30% calibration and 70% test.
  - *Calibration* is used for hyperparameter selection and propensity sanity checks.
  - *Test* is used only for final reporting and is never touched during tuning.
- **Temporal guard:** the random log overlaps in time with the second standard window. Use standard data on or before the start of the random window as the main training set, and report a sensitivity run that uses all standard data.
- **Sanity checks to report:**
  - The item-popularity distribution in the random log is close to uniform over the random item pool.
  - The duration distribution is compared across the standard log, the random log and the full catalogue.

## 3. Coupling analysis (before any modelling)

This analysis tests whether the RQ3 coupling exists, so it comes before any model.

1. Compute the Spearman correlation between item duration and item popularity (exposure count and play count) on the standard logs and on the random log.
2. Build a 2-D grid: 5 popularity quantiles × 5 duration quantiles. For each cell, report:
   - its share of exposures (standard log);
   - its positive rate under each label (random log);
   - its share of positives (standard log).
3. Read the gap between cells. Comparing logged exposure with random-log relevance shows where the policy over- or under-exposes items compared with their unbiased appeal.

The same grid is reused in §7 to show where each method moves recommendation mass.

## 4. Labels: hold the target fixed

Duration correction changes the target as well as the weighting. To keep that from being confounded with the model, every method is trained on its own label and evaluated under **all** of these labels:

| Label | Definition | Bias it carries |
|---|---|---|
| Click | `is_click` | exposure; rewards attractive covers over content |
| Long view | `long_view` | partly duration-normalised |
| Watch ratio | `play_time_ms / duration_ms` (capped) | favours short videos |
| Raw watch time | `play_time_ms` | favours long videos |
| Satisfaction | any of like/follow/forward/comment, excluding `is_hate` | sparse, least proxy-like |

The result is a method × evaluation-label table. A method that "wins" only on its own training label is reporting a change of definition, not a debiasing gain.

## 5. Models and interventions

- **Backbone:** fixed across methods so that only the correction varies. Use matrix factorisation as the primary model and one feature-based model (e.g. DeepFM with user and video features) as a robustness check. Use the same embedding size and tuning budget for both.

| Family | Methods | Knob swept to trace a frontier |
|---|---|---|
| None | Naive MF on click; value regression on raw watch time (the D2Q paper's baseline) | — |
| Popularity | MACR (Wei et al. KDD '21) | MACR c |
| Duration | D2Q (Zhan et al. KDD '22), ranked by predicted watch time or by duration quantile | number of duration groups |
| Exposure | IPS with a feature-based propensity (Schnabel et al. ICML '16) | weight-clipping threshold |

- Every knob gets 6–9 values, so each method produces a curve rather than a single point.
- Optional extension: DICE (Zheng et al. WWW '21), to test whether removing the conformity embedding reproduces the popularity trade-off.

## 6. Bias panel

All metrics are computed on the random-exposure test split. Top-K lists are produced over the random item pool, with K ∈ {10, 20}.

**Accuracy** (one value per evaluation label):

- GAUC and NDCG@K within each user's randomly exposed candidate set.

**Popularity:**

- Average recommendation popularity (ARP).

**Exposure:**

- Gini coefficient of item recommendation frequency.
- For IPS: maximum weight, effective sample size, and standard deviation of every metric across seeds.

**Duration:**

- Wasserstein distance between the duration distributions of recommended items and of relevant items.

## 7. Analysis

1. **Audit matrix.**
   - Rows are methods at their tuned setting; columns are panel metrics.
   - Each cell is the change from the naive baseline trained on the same label. Signs are flipped so that positive always means improvement.
   - Each cell gets a bootstrap 95% CI over test users.
   - The diagonal answers RQ1 and the off-diagonal cells answer RQ2.
2. **Frontiers.**
   - Plot every knob sweep in accuracy × popularity, accuracy × exposure and accuracy × duration space.
   - Mark the Pareto-nondominated points.
   - Report hypervolume against a fixed reference point so that method families can be compared with one number.
3. **Mass-shift heatmaps.**
   - For each method, plot (share of recommendations per popularity × duration cell) minus the naive model's share.
   - This is the direct test of RQ3 and H1: it shows whether D2Q's mass lands in the short, popular cells or the short, niche cells.
4. **Where accuracy comes from.**
   - Split each method's NDCG gain or loss across the same grid cells and across evaluation labels. This shows *where* accuracy was traded, which tests H3.

## 8. Statistical protocol

- 5 seeds per configuration.
- Report the mean ± standard deviation across seeds, plus a user-level bootstrap CI (1,000 resamples).
- Use paired tests against the naive baseline on per-user metrics, with Holm correction across the panel.
- All selection happens on the calibration split. The test split is evaluated once per final configuration.

## 9. Optional extension: feedback loops

A single offline round cannot show loop effects (Chaney et al. RecSys '18; Mansoury et al. CIKM '20). If time allows:

- Use KuaiSim, which is built on KuaiRand, or a user-response model fitted on the random log as the simulator.
- Run 5–10 rounds: recommend, simulate feedback, retrain.
- Track how the panel metrics change over rounds for naive, MACR, D2Q and IPS.

Report this as an exploratory result, not a main claim.

## 10. Threats to validity

- **The random log is only approximately unbiased.** Exposure is randomised, but the random items were still inserted into a feed, the item pool is restricted, and engagement still depends on the user's own selection. Claims hold relative to this pool.
- **The random-log test split is small.** Metrics on tail items and duration buckets will be noisy, which is why CIs are reported per cell and sparse cells are merged.
- **Proxy labels.** No label is ground-truth satisfaction. The label-matrix design in §4 reduces this problem but does not remove it.
- **Propensity misspecification.** IPS results depend on the propensity model. The feature-based propensity is checked by its AUC against uniform negatives and its correlation with item popularity, and its effective sample size is reported at every clipping threshold.
- **Generalisation.** Results come from one platform and one period. Optionally replicate the popularity and exposure rows on Yahoo!R3 or Coat for comparison with prior multi-bias work. Those datasets have no duration, so that row cannot be replicated.

## 11. Expected contribution

- A per-label audit matrix of debiasing methods on KuaiRand's random-exposure data.
- The first explicit measurement of the duration ↔ popularity interaction on this dataset.
- A frontier view showing which single-bias corrections are dominated once the full bias panel is considered.
