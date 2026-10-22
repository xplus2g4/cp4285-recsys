# Prompt: write one method page for the debiasing trade-off presentation

Fill in `{{METHOD_PAGE}}`, `{{METHOD_KEYS}}` and the method brief at the end, then hand this whole file to one sub-agent per method.

## Context

- **Research.** We measure the *trade-offs* of recommendation debiasing methods. Chen et al. (*Bias and Debias in Recommender System: A Survey and Future Directions*, ACM TOIS 2023, §4.5 and §5.2) note that "existing methods are usually designed for just addressing one or two specific biases". Our question is what each single-bias method does to the biases it does **not** target.
- **Setup.**
  - Dataset: KuaiRand-Pure.
  - Training: the standard (policy-biased) log only.
  - Evaluation: the random-exposure log, split by user into a 30% calibration part and a 70% test part (19,099 users).
  - Backbone: every method uses the same 64-dimensional MF backbone, trained on clicks. D2Q and VR are the exceptions: they train on raw watch time.
  - Knob values: each method has one knob, and every value is its own sweep. A sweep tunes lr × weight decay on a 9-point grid using calibration, then trains 5 seeds and evaluates each one on test exactly once.
- **Audience.** A professor in a 1:1 progress meeting. He knows recommender systems. He may not know each paper's private vocabulary.
- **Deliverable.** `.presentation/{{METHOD_PAGE}}`, one static HTML page. It is linked from `.presentation/index.html`, which another author owns.

## Sources (read before writing; never invent numbers)

| What | Where |
|---|---|
| All results (seed mean, seed std, Δ vs baseline, paired-bootstrap CI of Δ, significance, tuned lr/wd, IPS diagnostics) | `.presentation/assets/data.js` (`window.RESULTS.sweeps[]`, access it through `RB.*`) |
| Findings text and tables already reviewed for this method | `/tmp/pres/src/pr-<method>.md` (the merged PR description) |
| Paper summary | `docs/literature/<paper>.md`; the PDF is in `docs/literature/papers/` if you need details |
| Our implementation and its departures from the paper | `docs/spec_pipeline.md` §13 and `src/cp4285_proj/recommenders/<module>.py` |
| Metric definitions | `docs/spec_pipeline.md` §7.3 |

**Units in `data.js`:**
- `list_duration` is stored in **ms**. `RB.val` converts it to seconds.
- Wasserstein (W1) is in seconds.
- ARP is in training exposures.
- Diagnostics are in `rec.diag` (`ips_ess_share`, `ips_max_weight`, `ips_clipped_share`, `propensity_auc`, `propensity_pop_spearman`, `d2q_groups`).

## Shared assets (read them; do not edit them)

`assets/style.css`, `assets/glossary.js`, `assets/charts.js` and `assets/data.js` are owned by the index author. If you need a change, report it; do not patch them.

Load the assets in this order:
```html
<link rel="stylesheet" href="../assets/style.css">
<script src="../assets/data.js"></script>
<script src="../assets/glossary.js"></script>
<script src="../assets/charts.js"></script>
```

`window.RB` API (see `assets/charts.js`):

| Function | Use |
|---|---|
| `RB.lineChart(el, {methods, metric, height?, xlabel?, tipMetrics?, baseline?, errorBars?})` | Metric against knob, with the baseline as a dashed line, ± seed std bars, and filled dots where Δ is significant. |
| `RB.scatter(el, {methods, x, y, baselines?, labels?, height?, tipMetrics?})` | Trade-off plot, one curve per method in knob order. |
| `RB.deltaTable(el, {methods, metrics, includeBaseline?})` | Seed means with coloured Δ and `*`. |
| `RB.metricSelect(el, keys, initialKey, render)` | Dropdown that re-renders a chart. |
| `RB.sweeps(m)`, `RB.find(m, knob)`, `RB.baselineOf(m)`, `RB.val/std/delta/dci/sig(rec, key)`, `RB.fmt(key, v, {signed})` | Use these when a number appears in prose that you want to compute rather than hard-code. |

- **Metric keys:** use the keys of `RB.METRIC_INFO`.
- **Method keys:** `naive_mf vr pd pda d2q d2q_q ips_pop snips macr ips_feat`. Knobs are numbers, the string `"inf"`, or `null` for baselines.

## Terminology rules (mandatory)

1. **Use the standard names in `window.METHODS`:** Naive MF, VR, PD, PDA, D2Q, D2Q-Q, IPS-pop, SNIPS, IPS-feat, MACR. Use the knob symbols γ, M, clip, c. Do not introduce synonyms such as "IPS-popularity", "feature IPS" or "MF-IPS".
2. **Define every term before using it.** At its first use on the page, wrap each jargon word in `<span data-term="KEY">…</span>`, where KEY is a key of `window.GLOSSARY`; hovering then shows the definition.
   - If a term you need is not in the glossary, define it inline in plain words at its first use, and list it in your report.
   - A paper-specific symbol must be explained in words where it first appears. Example: "ŷ_k, the user-item matching score".
3. **Write in simple technical English.** Use short sentences, one idea per sentence, and active voice. No marketing words ("novel", "powerful", "significantly" unless you mean statistical significance). No filler.
4. **Report statistics precisely.**
   - "Significant" means the paired-bootstrap CI of Δ excludes 0.
   - Always state seed std next to small effects.
   - Say "about" for rounded numbers.
   - Do not claim causality beyond what the mechanism and the data support. Mark interpretations as interpretation ("Likely because …").

## Required page structure

Copy this skeleton and keep the section ids, because the index links to them.

```html
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{NAME}} — debiasing trade-offs</title>
<link rel="stylesheet" href="../assets/style.css">
<script src="../assets/data.js"></script><script src="../assets/glossary.js"></script><script src="../assets/charts.js"></script>
</head><body><div class="page">
<nav class="toc"><a class="home" href="../index.html">← Overview</a>
  <a href="#summary">Summary</a><a href="#problem">Problem</a><a href="#mechanism">How it works</a>
  <a href="#demo">Interactive demo</a><a href="#family">Recommender family</a><a href="#impl">Our implementation</a>
  <a href="#results">Results</a><a href="#findings">Findings</a><a href="#discussion">Discussion</a><a href="#caveats">Caveats</a></nav>
<main>
  <h1>{{NAME}}</h1><p class="subtitle">{{full paper title}} — {{authors short}}, {{venue year}}</p>
  <section id="summary"> pills (bias targeted / mechanism / pipeline stage / recommender family) + 3–4 sentence summary + "Headline result" card </section>
  <section id="problem"><h2>The problem it targets</h2> … </section>
  <section id="mechanism"><h2>How it works</h2> training vs inference, equations in <div class="eq">, every symbol explained </section>
  <section id="demo"><h2>Interactive demo</h2><div class="widget"> … </div></section>
  <section id="family"><h2>Recommender family</h2> … </section>
  <section id="impl"><h2>Our implementation</h2> knob grid, departures from the paper, diagnostics </section>
  <section id="results"><h2>Results</h2> charts + table </section>
  <section id="findings"><h2>Findings</h2> … </section>
  <section id="discussion"><h2>Discussion</h2> … </section>
  <section id="caveats"><h2>Caveats</h2> … </section>
  <footer>Data: test split, 19,099 users, 5 seeds per setting. Generated from outputs/sweeps via assets/data.js.</footer>
</main></div>
<script> /* page charts and the demo */ </script>
</body></html>
```

**Pills in the summary.** Use the classes `pill pop|dur|exp` for the bias, `pill stage` for the mechanism and pipeline stage, and `pill fam` for the family.
- Mechanism vocabulary: *loss reweighting*, *causal adjustment of the score*, *label transformation*, *counterfactual inference*.
- Pipeline-stage vocabulary: *training loss*, *training label*, *inference score*.

### Section contents

- **Problem.** Which bias the method targets, why it matters, and what goes wrong without the method. Use a concrete KuaiRand example where possible.
- **How it works.**
  - Describe training and inference separately, as numbered steps.
  - Include the key equation(s). Explain every symbol in words.
  - Say what the knob controls and what its extreme values mean (for example: knob = 0 is identical to Naive MF).
- **Interactive demo** (required).
  - Build a small, self-contained toy that shows the *mechanism*, not the results. It should use 5–12 toy items or rows, with sliders or toggles for the knob, and show how scores, weights, targets or the ranking change live.
  - Use inline SVG or HTML only. No external libraries or CDNs. It must work from `file://`.
  - The toy must be deterministic (fixed numbers or a seeded generator).
  - Put a one-paragraph "What to look at" caption under it.
  - Label it as a toy example with made-up numbers.
- **Recommender family.**
  1. The family of models the method was proposed and evaluated on in its paper. Name the backbones the paper tested, e.g. MF, LightGCN, a feature-based / CTR model, or a non-recsys learner.
  2. Whether it is model-agnostic, and what it requires from the backbone (a score, embeddings, item popularity, side features, timestamps).
  3. What we used: the MF backbone, a CF model.
  4. Whether it would carry over to graph CF, sequential or LLM-based recommenders. Mark this as reasoning, not evidence, unless the paper or a source shows it.
- **Our implementation.**
  - The knob grid.
  - Every departure from the paper, and why.
  - The diagnostics we logged.
  - Whether the tuned lr/wd changes with the knob (`rec.lr`, `rec.wd`). This matters because those rows mix the method's effect with an optimisation change.
- **Results.**
  - At least one `lineChart` with a `metricSelect` over the relevant metrics.
  - At least one `scatter` showing the main trade-off.
  - A `deltaTable` over a curated set of 8–12 metrics, covering accuracy, popularity, exposure, duration, clickbait and the user gap.
  - For each chart, a one-line caption saying what to look at.
- **Findings.** 4–7 bullets. Each bullet is one claim with its numbers (mean, Δ, seed std where relevant), and each says whether the effect is on the targeted bias or off-target. Ground every number in `data.js` or the PR text.
- **Discussion.** A short discussion of 2–4 paragraphs:
  - why the observed effects happen (mechanism → result);
  - which trade-off this method exposes;
  - how it relates to the research questions:
    - RQ1: does it reduce its own bias?
    - RQ2: what happens to the other biases?
    - RQ3: duration ↔ popularity coupling;
    - RQ4: Pareto position;
  - the hypotheses H1 / H2 / H3 where relevant:
    - H1: duration correction moves lists to short videos and shifts popularity;
    - H2: IPS raises tail coverage but also seed variance and tail precision loss;
    - H3: popularity debiasing narrows the item gap but widens the user gap, with mainstream users losing.
  - End with 1–2 open questions this method raises.
- **Caveats.**
  - The bootstrap CIs of set metrics (coverage, entropy, Gini) are biased.
  - There is no multiple-comparison correction.
  - `*` covers user sampling, not seed variance.
  - Satisfaction cells rest on about 10% of users.
  - Add any caveats specific to the method.

## Verification (required before you finish)

1. The server is already running at `http://127.0.0.1:8765/`. Open `http://127.0.0.1:8765/{{METHOD_PAGE}}` in the browser tool.
2. There must be no console errors.
3. There must be no element with class `term-missing`.
4. Every chart must render.
5. Move every demo control at least once and confirm the display updates.
6. Take a full-page screenshot and look at it: no overlapping text and no empty chart boxes.
7. Check that every hard-coded number in your prose matches `data.js` or the PR text after rounding.

## Report back (keep it short)

- The file written.
- Glossary keys you needed but that were missing, with the definition you used inline.
- Any requested changes to the shared assets.
- Anything in the sources that looked inconsistent.
- The structured fields requested in the output schema.

## Method brief

{{METHOD_BRIEF}}
