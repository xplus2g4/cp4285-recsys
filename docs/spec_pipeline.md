# Spec: bias-panel pipeline on KuaiRand-Pure (iteration 1)

This spec turns `docs/draft_methodology.md` into a pipeline:

1. preprocess and split KuaiRand-Pure once, into a frozen split;
2. feed that split to any recommender through one interface;
3. train it with torch;
4. evaluate it with an editable metric panel.

Iteration 1 implements one recommender, Naive MF. Everything else is built so that the §5 methods can be added without restructuring.

Section references (§N) point to `docs/draft_methodology.md`.

## 1. Scope

**In scope (iteration 1):**
- `preprocess` for the `guarded` split;
- the `Recommender` / `TorchRecommender` / `Backbone` abstraction;
- Naive MF;
- the GPU training loop with early stopping on calibration;
- `sweep` (tuning, then seed reruns);
- saved predictions;
- the metric registry and the whole §6 panel except the IPS-only diagnostics;
- the user-level bootstrap and the seed mean ± std;
- MLflow tracking;
- a soft guard on the test split.

**Out of scope (extension points, see §12):**
- the `full` split (sensitivity run);
- paired tests and Holm correction;
- training on raw watch time;
- IPS diagnostics (maximum weight, ESS);
- every recommender other than Naive MF;
- the §3 coupling analysis, which goes in a notebook that reads the preprocess outputs;
- §9 feedback loops.

## 2. Tech stack

| Concern | Choice | Note |
|---|---|---|
| Environment | `uv`, Python 3.12 | already set up in `pyproject.toml` / `.python-version` |
| Training | `torch>=2.7` from the `pytorch-cu128` index | the RTX 5080 (sm_120) needs CUDA ≥ 12.8 wheels; same `[tool.uv.sources]` block as `~/cp4285` |
| Preprocessing | `polars` | lazy CSV scan, Parquet output |
| Metrics / statistics | `numpy`, `scipy` | `scipy.stats.wasserstein_distance`, `spearmanr` |
| Config | TOML (`tomllib`) → frozen dataclasses; `--field value` CLI overrides | the convention in `~/cp4285` |
| Tracking | `mlflow>=3` on `sqlite:///mlflow.db` | one experiment: `kuairand-bias-panel` |
| Progress | `tqdm` | |
| Tests | `pytest` (dev group) | |

Dataset size makes the stack simple. After the item drop, train is about 1.12M rows, the random log about 1.15M rows, and there are 27,285 users and 7,344 items. All training tensors fit on the GPU, so there is no `DataLoader` and no out-of-core processing.

Entry points (`[project.scripts]`) are listed below. They replace the placeholder `cp4285-proj = "cp4285_proj:main"`.

```toml
kr-preprocess = "cp4285_proj.cli:preprocess_main"
kr-train      = "cp4285_proj.cli:train_main"
kr-sweep      = "cp4285_proj.cli:sweep_main"
kr-evaluate   = "cp4285_proj.cli:evaluate_main"
kr-rescore    = "cp4285_proj.cli:rescore_main"
```

## 3. Layout

```
src/cp4285_proj/
  config.py            # dataclasses, TOML loading, CLI overrides
  cli.py
  tracking.py          # MLflow helpers, RunLogger
  data/
    preprocess.py      # raw CSV → processed split
    labels.py          # LabelSpec registry
    schema.py          # Parquet schemas + loaders (incl. test guard)
  recommenders/
    base.py            # Recommender protocol, TorchRecommender, Backbone
    registry.py        # @register_recommender, lookup by name
    naive_mf.py        # MFBackbone, NaiveMF
  training/
    loop.py            # GPU-resident epoch loop, early stopping, seeding
    sweep.py           # budgeted search → final seed reruns
    runner.py          # one training run: MLflow run, best.pt, config.toml, fit.json, calibration predictions
  evaluation/
    bundle.py          # EvalBundle construction + resampling
    predict.py         # candidate scores, top-K, caching
    panel.py           # panel.toml parsing/expansion, runner, panel hash
    bootstrap.py       # user-level bootstrap, seed aggregation
    guard.py           # test-split access control
    evaluate.py        # kr-evaluate / kr-rescore orchestration (per run, per sweep)
  metrics/
    registry.py        # @register_metric, MetricKind
    accuracy.py        # gauc, ndcg
    popularity.py      # arp, gini, aplt, rsp, reo (pop groups)
    duration.py        # list_duration, wasserstein_duration, rsp/reo (dur groups)
    coverage.py        # coverage, exposure_entropy
    groups.py          # ndcg_by_group, calibration_kl
    clickbait.py       # clickbait_share
    common.py          # shared vectorised helpers (top-k frequencies, bin counts, weighted median)
configs/
  data.toml            # preprocess parameters
  pipeline.toml        # pipeline-fixed settings (tuning budget, seeds, K_max, selection rule)
  panel.toml           # the editable metric panel
  recommenders/mf.toml # MF hyperparameters + search space
  experiments/*.toml   # recommender × training label × split
data/raw/KuaiRand-Pure/data/   # downloaded (MD5 0820331067a3784d9691136f772b35a7), git-ignored
data/processed/<split>/        # preprocess output, git-ignored
outputs/runs/<run_id>/         # run artifacts, git-ignored
notebooks/                     # §3 coupling analysis (outside the pipeline)
```

## 4. Step 1: preprocess (`kr-preprocess --split guarded`)

### 4.1 Inputs

`data/raw/KuaiRand-Pure/data/`, with the row counts verified at load. The counts below are measured; the CSVs have one more line each for the header.

| File | Rows |
|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | 1,141,112 |
| `log_standard_4_22_to_5_08_pure.csv` | 295,497 (not used by `guarded`) |
| `log_random_4_22_to_5_08_pure.csv` | 1,186,059 |
| `user_features_pure.csv` | 27,285 |
| `video_features_basic_pure.csv` | 7,583 |
| `video_features_statistic_pure.csv` | 7,583 (**never read**: averaged over both windows, so it leaks the test window) |

If any file is missing or has the wrong row count, preprocessing fails before it writes anything.

### 4.2 Steps

1. **Drop items with no duration.** Drop the 239 items with null `video_duration` together with all their log rows: 24,076 standard rows and 36,920 random rows. These are exactly the items whose log rows all have `duration_ms == 0`. The pool becomes **7,344 items**. The drop counts go in the manifest.
2. **Index.**
   - `user_idx` is contiguous over `user_features_pure.csv`.
   - `item_idx` is contiguous over the kept pool items, ordered by `video_id`.
   - Both raw IDs are kept as columns.
3. **Labels** (`data/labels.py`). These are computed on every log row:

   | Label | Definition | Kind | Binary relevance used by GAUC / REO / Wasserstein |
   |---|---|---|---|
   | `click` | `is_click` | binary | `== 1` |
   | `long_view` | `long_view` | binary | `== 1` |
   | `satisfaction` | `(is_like ∨ is_follow ∨ is_forward ∨ is_comment) ∧ ¬is_hate` | binary | `== 1` |
   | `watch_ratio` | `min(play_time_ms / duration_ms, cap)`, with `cap = 1.0` | graded in [0,1] | `>= 1.0` (full play) |
   | `raw_watch_time` | `play_time_ms` | graded, unbounded | none (NDCG only) |

4. **Train split (`guarded`).** All rows of `log_standard_4_08_to_4_21`. Every exposure is its own row: repeated (user, item) exposures are kept. The raw file has 48,362 repeat exposures (rows beyond the first of a pair; 45,136 distinct pairs repeat); after the item drop, 47,585 (both counts are in the manifest as `train_repeats`).
5. **Calibration and test split.** Random-log users are shuffled with `split_seed` and split 30% calibration / 70% test **by user**. Every random-log row of a user goes to that user's part. The split seed is fixed, does not depend on the model seeds, and is recorded in the manifest.
6. **Item metadata.** Computed on the train split only.
   - `pop_exposure`: number of train rows for the item.
   - `pop_play`: number of train rows with `play_time_ms > 0`.
   - `duration_ms`: from `video_duration`.
   - `pop_bin` / `dur_bin`: 5 equal-size quantile bins over the pool, ranked by `(pop_exposure, video_id)` and by `(duration_ms, video_id)`. Breaking ties by `video_id` keeps the bins equal in size even though popularity has many tied counts. These bins are the §3 grid and are reused by RSP/REO and calibration KL.
   - `is_head`: one of the top `ceil(0.2 · 7,344)` items by the same popularity ranking. The head holds about 75% of train exposures. Every other item, cold items included, is long tail.
   - `is_cold`: `pop_exposure == 0`.
   - `is_clickbait`: computed from **calibration** rows only, so test labels never define item attributes. An item is clickbait if it has at least `clickbait.min_support = 20` calibration exposures, its click rate is ≥ the 75th percentile, **and** either its mean `watch_ratio` is ≤ the 25th percentile or its `is_hate` rate is ≥ the 90th percentile **and > 0**. Percentiles are taken over the supported items (`np.quantile`, linear interpolation). The thresholds live in `data.toml` and are recorded in the manifest.
     - *Implementation note (`> 0`).* `is_hate` is rare (0.096% of calibration rows; 316 of 7,077 supported items have any), so its 90th percentile is 0.0 and "rate ≥ p90" alone holds for every item. Read literally, the hate arm would make every high-click item clickbait (1,772 items). With the `> 0` guard there are 163 (92 from the watch-ratio arm). The manifest records `hate_rate_requires_positive = true`.
7. **User metadata.** History is the user's train rows. A user with fewer than `min_history = 5` distinct train items gets the group `no_history` in every history-based grouping (`mainstream`, `dur_pref`; `activity` comes from the user table and has no `no_history` group). This covers, for example, the random-log users who have no train rows at all: 1,075 in the raw logs (checked at runtime), 1,113 after the item drop, because 38 users had train rows only on dropped items.
   - `activity_group`: `user_active_degree` collapsed to `high` (`high_active`), `full` (`full_active`), `middle` (`middle_active`) and `other` (everything else: low, single_low, 2_14_day_new, day_new, 30day_retention, UNKNOWN; 1,035 users).
   - `mainstream_score`: Spearman correlation between the user's per-item train exposure counts (a vector over all pool items, zeros included) and the global `pop_exposure` vector. Kowald et al. take this measure from Bauer & Schedl and use Kendall τ; Spearman is chosen here. Ties take average ranks, as in `scipy.stats.spearmanr`. `mainstream_group`: tertiles `low` / `mid` / `high` over users with enough history.
   - `dur_pref_ms`: median `duration_ms` over the user's train rows. `dur_pref_group`: tertiles `short` / `mid` / `long`.
   - Tertiles are equal-size and rank-based: users are ranked by `(value, user_idx)` and split into thirds, the same way as the item bins.
   - `hist_pop_dist[5]` / `hist_dur_dist[5]`: share of the user's train rows in each `pop_bin` / `dur_bin`, for calibration KL.
   - `role`: `calibration` | `test` | `train_only`. In this release every one of the 27,285 users is in the random log, so `train_only` is empty.
8. **Manifest and sanity checks** (`manifest.json`):
   - hashes of the input files;
   - `split_seed` and the config snapshot;
   - row counts per part;
   - user counts per part and per group;
   - drop counts;
   - cold-item and cold-user counts per part;
   - per-label positive rates per part;
   - the §2 checks:
     - how uniform item frequency is over the pool in the random log (coefficient of variation and the χ² statistic against uniform);
     - duration quantiles (p10/25/50/75/90) for train, random and catalogue.

   It also records hashes of the output files.

Preprocessing is deterministic: running it twice gives the same output hashes.

### 4.3 Output schemas (`data/processed/guarded/`)

```python
# train.parquet, calibration.parquet, test.parquet (one row per log row)
row_id: int64            # 0-based row index in the raw CSV; stable within the part, unique across calibration + test
user_idx: int32
item_idx: int32
time_ms: int64
tab: int8
click: int8
long_view: int8
satisfaction: int8
watch_ratio: float32     # capped
raw_watch_time: int64    # play_time_ms
is_hate: int8

# items.parquet (one row per pool item, sorted by item_idx)
item_idx: int32
video_id: int32
duration_ms: int64
pop_exposure: int32
pop_play: int32
pop_bin: int8            # 0..4, 4 = most popular
dur_bin: int8            # 0..4, 4 = longest
is_head: bool
is_cold: bool
is_clickbait: bool

# users.parquet (one row per user, sorted by user_idx)
user_idx: int32
user_id: int32
role: str                # calibration | test | train_only
n_train_rows: int32
n_train_items: int32
activity_group: str
mainstream_score: float32   # NaN if no_history
mainstream_group: str
dur_pref_ms: float32        # NaN if no_history
dur_pref_group: str
hist_pop_dist: list[float32]  # length 5; null if no_history
hist_dur_dist: list[float32]  # length 5; null if no_history

# manifest.json (see §4.2 step 8)
```

`test.parquet` is read only through `data.schema.load_test(token)` (see §9).

## 5. Steps 2–3: the recommender abstraction and training

### 5.1 Ownership

| Owned by the **recommender** (`configs/recommenders/<name>.toml`) | Owned by the **pipeline** (`configs/pipeline.toml`, the same for every method) |
|---|---|
| model hyperparameters, **including embedding size** | training label (experiment config, applied outside the recommender) |
| optimizer, batch size, max epochs, patience | tuning budget (number of trials) |
| search space (`[search]`) | selection rule: calibration GAUC under the training label's binary relevance |
| inference (`score`, `recommend`) | seeds for the final runs, `K_max`, masking policy |

### 5.2 Interfaces

```python
# recommenders/base.py
from typing import ClassVar, Protocol, Self
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class TrainData:
    """Train split, resident on `device`. Row-aligned tensors."""
    user: torch.Tensor          # int64 [N]
    item: torch.Tensor          # int64 [N]
    target: torch.Tensor        # float32 [N]; the training label (soft target for watch_ratio)
    extras: dict[str, torch.Tensor]  # other label columns, time, tab: for future weighting hooks
    items: "ItemTensors"        # pop_exposure, pop_bin, dur_bin, duration_ms on device, indexed by item_idx
    n_users: int
    n_items: int

@dataclass(frozen=True)
class FitContext:
    label: "LabelSpec"
    device: torch.device
    seed: int
    selection: "SelectionFn"    # pipeline-owned: (Recommender) -> float, calibration GAUC
    calibration: "CandidateSet" # for selection only; the model never sees calibration labels as training data
    log: "RunLogger"            # MLflow + file logging

@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_selection_score: float
    epochs_run: int

class Recommender(Protocol):
    """The only surface the pipeline uses. Implementations register with @register_recommender(name)."""
    name: ClassVar[str]

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult: ...
    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Row-aligned float32 scores for (user, item) pairs. Used for candidate sets."""
    def recommend(self, user_idx: np.ndarray, k: int, exclude: "CSRMask | None") -> tuple[np.ndarray, np.ndarray]:
        """Top-k over the whole pool per user: (item_idx int32 [U,k], score float32 [U,k]), sorted descending.
        Items in `exclude` get score -inf before top-k."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self: ...

class Backbone(nn.Module):
    """User/item → logit. Shared across methods so that only the correction varies."""
    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # [B] logits
        raise NotImplementedError
    def score_all(self, user: torch.Tensor) -> torch.Tensor:                   # [B, n_items] logits
        """Default: chunked forward over all items. Backbones override when cheaper (MF: one matmul)."""

class TorchRecommender(Recommender):
    """Generic torch implementation. Subclasses provide `build_backbone` and may override hooks.
    Iteration 1 uses only the defaults."""
    backbone: Backbone

    def build_backbone(self, n_users: int, n_items: int) -> Backbone: ...
    def configure_optimizer(self) -> torch.optim.Optimizer: ...       # default: Adam(lr, weight_decay)

    # hooks (defaults are the naive behaviour)
    def sample_weight(self, batch: "Batch") -> torch.Tensor | None:   # default None → unweighted (IPS/SNIPS/DR later)
        return None
    def loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        ...                                                            # default: BCEWithLogits, mean (weighted if weight)
    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        return logits                                                  # inference-time correction (PDA/MACR later)

    # fit → training.loop.run(self, train, ctx); score/recommend → backbone + adjust_scores + masking
```

The default `recommend` works through users in chunks:

1. `backbone.score_all`;
2. `adjust_scores`;
3. set excluded items to `-inf` (from a CSR mask built from the user's train items);
4. `torch.topk(k)`.

No pipeline code reads embeddings or other model internals.

**Additions made during implementation** (they extend the contract above; nothing above was removed or changed):
- `Batch` is a frozen dataclass `(idx, user, item, target, train)`; `batch.extra(name)` gathers `train.extras[name][idx]` on demand, so unused extras are never sliced.
- `CandidateSet` is `(row_id, user_idx, item_idx, labels)` as numpy arrays; `CSRMask` (in `data/schema.py`) is `(indptr, indices)` over all users; `RunLogger` is a protocol with `log_metrics(metrics, step)`.
- Construction is not part of the pipeline surface. A `TorchRecommender` is built as `cls(model_cfg, train_cfg)` and declares two class attributes: `model_config_cls` (the frozen dataclass for its `[model]` section) and `supported_labels` (checked at config load, which is how MF rejects `raw_watch_time`).
- `Backbone` declares `n_users` and `n_items` (used by `save` and the default `score_all`).

### 5.3 Naive MF (`recommenders/naive_mf.py`)

- **Backbone:** `logit(u,i) = ⟨p_u, q_i⟩ + b_u + b_i + b`.
  - Embeddings are initialised with N(0, 0.01²) and biases with zero.
  - `score_all` is `P[u] @ Qᵀ + b_u + b_I + b`.
- **Loss:** pointwise BCE on logged exposures. There is no negative sampling, because the logs already contain the negatives.
  - For `watch_ratio`, the capped value is the soft target.
  - Supported training labels: `click`, `long_view`, `satisfaction`, `watch_ratio`. `raw_watch_time` is rejected at config load with a clear error.
- **Users and items with no train rows** keep their starting embeddings and are still scored. Torch's `Adam(weight_decay=…)` decays every row at every step, so `NaiveMF.configure_optimizer` keeps the default Adam and registers a step post-hook that restores those rows (`P`, `b_u` for users; `Q`, `b_i` for items) to their initial values.
- **`configs/recommenders/mf.toml` defaults:**

```toml
[model]
embedding_dim = 64

[train]
optimizer = "adam"
lr = 3e-3
weight_decay = 1e-6
batch_size = 8192
max_epochs = 50
patience = 5

[search]           # recommender-owned search space
lr = [1e-3, 3e-3, 1e-2]
weight_decay = [0.0, 1e-6, 1e-5]
```

### 5.4 Training loop (`training/loop.py`)

- **Seeding:** `random`, `numpy` and `torch` (including CUDA). `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` when `pipeline.deterministic = true`, which is the default.
- **Data on the GPU:** `TrainData` is built once on the device. Each epoch draws `torch.randperm(N, device=…)` and slices batches by index.
- **Per batch:** forward → `sample_weight` → `loss` → step.
- **After each epoch:** `selection = ctx.selection(model)`. This is calibration GAUC under the training label's binary relevance, computed from `score` over the calibration candidate rows. The best `state_dict` is kept in memory. Training stops when `patience` epochs pass without improvement or `max_epochs` is reached.
- **Logged to MLflow per epoch:** `train_loss`, `cal_selection`, `epoch_time_s`.

### 5.5 Sweep (`kr-sweep configs/experiments/mf_click.toml`)

1. **Tuning stage.**
   - Expand the recommender's `[search]` grid. If it holds more than `pipeline.tuning_budget` points (default 9), sample `tuning_budget` points with `pipeline.sweep_seed`.
   - Train each point with seed 0.
   - Choose the point with the highest `best_selection_score`.
2. **Final stage.** Retrain the chosen point with `pipeline.final_seeds = [0,1,2,3,4]`. These runs are tagged `final=true` and `sweep_id=<id>`.

`kr-train` runs a single configuration. `--final` tags it as final.

Every run is one MLflow run. It records:
- **params:** the flattened experiment, recommender and pipeline config, the split name and the manifest hash;
- **tags:** `final`, `sweep_id`, `stage`, `recommender`, `train_label`.

## 6. Step 3b: saved predictions

Every run writes these files to `outputs/runs/<run_id>/` (MLflow artifacts mirror them):

| File | Written by | Contents |
|---|---|---|
| `config.toml` | train | resolved config |
| `best.pt` | train | `state_dict` + recommender name + hyperparameters; loaded with `weights_only=True` |
| `fit.json` | train | `FitResult` |
| `pred/calibration/candidates.parquet` | train | `row_id, user_idx, item_idx, score` for every calibration row |
| `pred/calibration/topk.parquet` | train | `user_idx, rank, item_idx, score` for calibration users, `rank < K_max` |
| `pred/test/candidates.parquet` | evaluate (first test evaluation) | the same for test rows |
| `pred/test/topk.parquet` | evaluate (first test evaluation) | the same for test users |

- `K_max = 100` (pipeline config).
- Top-K is computed over the full pool. Each user's train items are masked when `pipeline.mask_train_seen = true`, the default. Only 624 random-log pairs also occur in train (609 after the item drop), so masking barely touches accuracy, but it keeps already-seen popular items out of the distribution metrics.
- Test predictions are produced lazily by `kr-evaluate`, so `train` and `sweep` never load `test.parquet`. The first test evaluation writes the test top-K with the run's current calibration top-K width.
- `kr-rescore <run_id> --k <K>` reloads `best.pt` and rewrites `topk.parquet` for both parts with a larger `K_max`. It is needed only for panel entries with `k > K_max`. The panel runner refuses such entries and names the `rescore` command. The test users are taken from the existing test `topk.parquet`, so rescoring never reads `test.parquet`; if no test top-K exists yet, only calibration is rewritten.

## 7. Step 4: the metric panel

### 7.1 Editing the panel

`configs/panel.toml` is the only place where the panel is defined.

- **To add a metric:** write one function decorated with `@register_metric` and add one TOML entry.
- **To drop a metric:** delete its entry.
- **No retraining is needed** for either: `kr-evaluate` reruns the panel on saved predictions.

```toml
# configs/panel.toml
eval_labels = ["click", "long_view", "watch_ratio", "raw_watch_time", "satisfaction"]
bootstrap = { resamples = 1000, seed = 0, ci = 0.95 }

[[metric]]
name = "gauc"
labels = "all"            # expands to eval_labels; skipped for labels with no binary relevance

[[metric]]
name = "ndcg"
k = [10, 20]
labels = "all"

[[metric]]
name = "arp"
k = [10, 20]
# … one [[metric]] table per entry in §7.3
```

Rules:
- **Expansion.** Each list-valued parameter, plus `labels = "all"`, is expanded into a cross product. Each instance gets a canonical id, e.g. `ndcg@10[click]`, `rsp_pop@20`, `ndcg_by_group@10[click|mainstream]`.
- **Validation before computing anything.** An unknown metric name, a missing or unknown parameter, `k > K_max`, or an unknown label fails before any metric is computed. `K_max` here is the width of the run's saved `topk.parquet`, so it grows after `kr-rescore`. A label without a binary relevance given explicitly to a metric that needs one also fails; under `labels = "all"` such labels are skipped.
- **Panel hash.** The SHA-256 of the expanded, canonicalised panel is stored with every result. Results are written to `outputs/runs/<run_id>/eval/<split>/<panel_hash>/`. The canonical form is the JSON (sorted keys) of `eval_labels`, the `bootstrap` settings and the instances sorted by id. Reordering `[[metric]]` tables therefore keeps the hash; changing entries or bootstrap settings changes it.

### 7.2 Metric contract

```python
# metrics/registry.py
MetricKind = Literal["per_user", "global", "per_group"]

def register_metric(name: str, *, kind: MetricKind, params: dict[str, type], needs_label: bool,
                    needs_relevance: bool = False, weight: Literal["uniform", "candidates"] = "uniform") -> Callable: ...
# params: every parameter except `label` (e.g. {"k": int}); a Literal[...] type lists allowed values.
# needs_label adds the `labels` parameter. needs_relevance: the label must have a binary relevance, so
#   labels = "all" can skip labels without one at validation time (gauc, reo_*, list_duration, wasserstein_duration).
# weight: how a per_user metric is aggregated; "candidates" weights users by |C_u| (gauc).
# These two keywords were added because expansion/validation and GAUC's aggregation need them and §7.3 implies them.

# Return types by kind:
#   per_user  → np.ndarray float64 [U] aligned with bundle.users; NaN = user excluded for this metric
#   global    → dict[str, float]           (one or more named values, e.g. {"value": g} or {"mean": …, "delta_mean": …})
#   per_group → dict[str, float]           group name → value, plus "gap" (max − min over groups other than no_history)

@dataclass(frozen=True)
class EvalBundle:
    split: Literal["calibration", "test"]
    users: np.ndarray                 # user_idx [U]; may repeat under bootstrap
    cand_user_pos: np.ndarray         # [R] index into `users` for each candidate row
    cand_item: np.ndarray             # [R] item_idx
    cand_score: np.ndarray            # [R]
    cand_labels: dict[str, np.ndarray]    # label name → [R] values
    labels: dict[str, "LabelSpec"]        # kind, binary relevance fn
    topk: np.ndarray                  # [U, K_max] item_idx
    seen: "CSRMask"                   # per user: train items (for "un-interacted" in RSP)
    items: "ItemMeta"                 # numpy arrays by item_idx: duration_ms, pop_exposure, pop_bin, dur_bin, is_head, is_clickbait
    user_groups: dict[str, np.ndarray]    # "activity" | "mainstream" | "dur_pref" → [U] group labels
    hist_dist: dict[str, np.ndarray]      # "pop" | "dur" → [U, 5]; NaN rows for no_history
    n_items: int

    def resample(self, positions: np.ndarray) -> "EvalBundle":
        """Bundle over users[positions] (with repeats), keeping their candidate rows and top-K rows."""

    # helpers added during implementation:
    n_candidates: np.ndarray          # property: |C_u| per bundle user
    def user_cache(self, key, fn) -> np.ndarray: ...  # per-user quantity computed once on the original bundle,
                                                      # indexed by the resample positions (keeps bootstrap fast)
    def memo(self, key, fn): ...                      # per-bundle memo (e.g. item frequencies shared by metrics)

def metric(b: EvalBundle, *, k: int | None = None, label: str | None = None, **params) -> np.ndarray | dict[str, float]: ...
```

Per-user metrics report `n_excluded` next to their value. Training diagnostics, such as the future IPS maximum weight and ESS, are **not** panel metrics. They are logged by the training loop.

### 7.3 Iteration-1 panel

Notation:
- `L_u`: user u's top-k list from the full pool.
- `C_u`: u's candidate rows (random exposures in the split).
- `rel`: the label's binary relevance.
- `pop(i)` = `pop_exposure`.

| Id | Kind | Params | Definition |
|---|---|---|---|
| `gauc` | per_user → aggregated with candidate-count weights | label | AUC of `score` against `rel` within `C_u`. NaN if `C_u` has only one class. The aggregate is GAUC = Σ_u \|C_u\|·AUC_u / Σ_u \|C_u\| over non-NaN users. Labels with no binary relevance (`raw_watch_time`) are skipped. |
| `ndcg` | per_user | k, label | NDCG@k ranking `C_u` by score. The gain is the label value: 0/1, capped watch ratio, or `raw_watch_time` min-max scaled within the user. The ideal is taken over `C_u`. Lists are truncated at min(k, \|C_u\|). NaN if every gain is 0 or every gain is equal. |
| `arp` | per_user | k | mean over i ∈ `L_u` of pop(i) |
| `aplt` | per_user | k | share of `L_u` with `¬is_head` |
| `gini` | global | k | Gini coefficient of item recommendation frequency over all `n_items` pool items, zeros included |
| `coverage` | global | k | \|∪_u L_u\| / n_items |
| `exposure_entropy` | global | k | Shannon entropy of item recommendation frequency / log(n_items) |
| `rsp_pop`, `rsp_dur` | global | k | Ranking-based statistical parity (Zhu, Wang & Caverlee, **SIGIR '20**; the draft's "Zhu et al. WSDM '21" is wrong, see §11). For each bin a: P_a = Σ_u \|L_u ∩ G_a\| / Σ_u \|G_a \ seen_u\|. RSP = std_a(P_a) / mean_a(P_a), population std. The groups G_a are the 5 `pop_bin` or `dur_bin` groups. |
| `reo_pop`, `reo_dur` | global | k, label | Ranking-based equal opportunity: the same as RSP with P_a = Σ_u \|L_u ∩ G_a ∩ Pos_u\| / Σ_u \|G_a ∩ Pos_u\|, where Pos_u = distinct items in `C_u` with `rel`. NaN if some bin has no positives. |
| `list_duration` | global | k, label | `{mean, median}` of `duration_ms` over all items in all `L_u` (pooled), `{ref_mean, ref_median}` over relevant candidate rows, and `{delta_mean, delta_median}` |
| `wasserstein_duration` | global | k, label | `scipy.stats.wasserstein_distance` between the durations (seconds) of pooled `L_u` items and of relevant candidate rows |
| `ndcg_by_group` | per_group | k, label, grouping ∈ {activity, mainstream, dur_pref} | mean `ndcg` per group, plus `gap` |
| `calibration_kl_pop`, `calibration_kl_dur` | per_user | k | Steck's C_KL = KL(p ‖ q̃), with p = `hist_dist`, q = the share of `L_u` in each bin, q̃ = (1−α)q + αp, α = 0.01. NaN for `no_history` users. |
| `clickbait_share` | per_user | k | share of `L_u` with `is_clickbait` |

Per-user values are averaged over non-NaN users unless the table says otherwise (as for `gauc`). Group breakdowns of other per-user metrics are added as new `per_group` metrics, the same way as `ndcg_by_group`.

### 7.4 Statistics

- **Per run** (`kr-evaluate <run_id> --split calibration|test`), output to `eval/<split>/<panel_hash>/`:
  - `per_user.parquet`: `user_idx`, `n_candidates`, then one column per per-user metric instance;
  - `summary.json`: point estimates and exclusion counts;
  - `bootstrap.json`: a 95% percentile CI from 1,000 user resamples, with a fixed seed. Per-user metrics resample the saved per-user vectors. Global and per-group metrics are recomputed on `bundle.resample(positions)`. The raw draws are kept in `bootstrap_samples.npy` (`[B, S]`, columns named in `bootstrap.json`).
  - Scalar names: a per-user or single-valued global instance is named by its id; a multi-valued global instance becomes `id.key` (e.g. `list_duration@10[click].delta_mean`); a per-group instance becomes `id.group` and `id.gap`.
  - *Caveat.* Resampling users with replacement leaves about 63% distinct users, so set-based global metrics (`coverage`, `gini`, `exposure_entropy`) are biased in the resamples, and their percentile CI can exclude the point estimate (seen for `coverage`). Read those CIs as variability, not as coverage intervals.
- **Per sweep** (`kr-evaluate --sweep <sweep_id> --split test`): evaluates every `final` run, then writes `aggregate.json` with the mean ± std (ddof = 1) over seeds of every point estimate. The aggregate is logged to the sweep's parent MLflow run (`stage=sweep`). The CI in `aggregate.json` is the percentile CI of the seed-mean: all runs share the same users and bootstrap seed, so resample b uses the same positions in every run, and the CI is taken over the per-resample mean across runs.
- **Extension point:** paired per-user tests against the naive baseline, with Holm correction across the panel. The per-user Parquet files already hold the input those tests need.

## 8. Configuration

The precedence chain is: dataclass defaults < TOML < CLI `--section.field value`.

```toml
# configs/data.toml
split_seed = 4285
calibration_frac = 0.3
watch_ratio_cap = 1.0
n_bins = 5
head_frac = 0.2
min_history = 5
[clickbait]
min_support = 20
click_rate_q = 0.75
watch_ratio_q = 0.25
hate_rate_q = 0.90

# configs/pipeline.toml (pipeline-owned; identical across methods)
tuning_budget = 9
sweep_seed = 0
final_seeds = [0, 1, 2, 3, 4]
k_max = 100
mask_train_seen = true
deterministic = true
tracking_uri = "sqlite:///mlflow.db"
experiment = "kuairand-bias-panel"

# configs/experiments/mf_click.toml
recommender = "mf"
train_label = "click"
split = "guarded"
```

`pipeline.toml` has no per-experiment override path. The CLI override is accepted only for debugging and tags the run with `pipeline_override=true`.

Fields beyond the TOML above, with dataclass defaults: `PipelineConfig.device = "cuda"`, `runs_dir = "outputs/runs"`, `sweeps_dir = "outputs/sweeps"`; `ExperimentConfig.pipeline_config = "configs/pipeline.toml"` and `recommender_config` (empty means `configs/recommenders/<recommender>.toml`); `DataConfig.raw_dir` and `processed_dir`. `kr-preprocess` takes `--config configs/data.toml` and `--data.*` overrides. The run directory and the MLflow run share one id (the MLflow run id).

## 9. Test-split guard (soft)

- `data.schema.load_test(token)` requires a `TestAccess` token. Only `evaluation.guard.authorize(run)` creates that token, and it requires the run tag `final=true` and MLflow status `FINISHED`; otherwise it raises an error. The status check means that a final run cannot authorise itself while it is still training. `training/` holds no token, so `kr-train` and `kr-sweep` have no way to get test rows.
- Each test evaluation increments the MLflow tag `test_eval_count` on the run and appends `{timestamp, panel_hash, reason}` to `eval/test/log.jsonl`. The count is recorded as soon as `test.parquet` has been read, so an evaluation that later crashes still counts.
- When `test_eval_count > 0`, `kr-evaluate` prints a warning listing the earlier evaluations and continues. `--reason "<text>"` is recorded when given. Nothing is hard-blocked, so you can add metrics later and rerun after fixing a metric bug.

## 10. Acceptance criteria (iteration 1)

1. `uv sync` installs the stack. `torch.cuda.is_available()` is true on the RTX 5080, and an MF forward and backward pass runs on it.
2. `kr-preprocess --split guarded` checks the raw row counts in §4.1 and writes the files in §4.3. Then:
   - The item pool has 7,344 items, and no dropped `video_id` appears in any part.
   - Calibration and test users are disjoint, their union is the random-log user set, and calibration is 30% ± 1 user.
   - Every random-log row (after the drop) is in exactly one of calibration or test.
   - `train.parquet` has 1,141,112 − 24,076 rows.
3. Running `kr-preprocess` twice gives the same output hashes in `manifest.json`.
4. The manifest contains every count and check in §4.2 step 8.
5. Popularity, bins, head, user groups and history distributions depend only on train rows: changing any calibration or test label leaves `items.parquet`, except `is_clickbait`, and `users.parquet` unchanged. `is_clickbait` depends on calibration rows only: changing test labels leaves it unchanged.
6. `kr-train configs/experiments/mf_click.toml` trains Naive MF on the GPU and writes everything in §6 for calibration. The calibration GAUC at the best epoch is above 0.5. With `deterministic = true`, two runs with the same seed give the same `best_selection_score`.
7. `kr-train` with `train_label = "raw_watch_time"` fails at config load with an error naming the unsupported label.
8. `kr-sweep` runs at most `tuning_budget` tuning runs and then 5 final runs, all visible in MLflow with the tags in §5.5.
9. `kr-evaluate <final_run> --split test` writes the test predictions and the panel output. Evaluating a non-final run on test raises an error. A second test evaluation warns and records `test_eval_count = 2`.
10. Adding a `[[metric]]` entry for a newly registered function, or deleting an entry, then rerunning `kr-evaluate`, gives results under a new panel hash. No retraining and no rescoring happen, because the existing `pred/` files are reused.
11. An unknown metric name, an unknown parameter, `k > K_max` or an unknown label in `panel.toml` fails before any metric is computed.
12. Metric correctness: unit tests with small hand-computed fixtures cover `gauc` (including single-class exclusion and candidate-count weighting), `ndcg` (binary, graded and truncation), `gini`, `rsp`/`reo`, `calibration_kl` (including smoothing) and `wasserstein_duration`.
13. `kr-evaluate --sweep <id> --split test` writes the mean ± std over the 5 seeds and a 95% bootstrap CI for every panel instance.
14. `kr-rescore <run_id> --k 200` rewrites `topk.parquet` with 200 columns, and a panel entry with `k = 200` then passes validation.

## 11. Departures from `draft_methodology.md`, and corrections

- **Embedding size is chosen by each recommender (§5).** The draft asks for "the same embedding size and tuning budget". Only the tuning budget is held fixed. Fix the paper text, or set the same `embedding_dim` in every recommender config.
- **No sensitivity run on all standard data (§2).** Only `guarded` is built. `kr-preprocess` takes the split name as a parameter so that `full` can be added later.
- **239 items dropped.** These items have no duration at all. The random log is therefore uniform over 7,344 items, not 7,583. List this in §10 threats.
- **`user_active_degree` has 9 levels, not 4** (the 4 levels are stated in `docs/literature/gao2022kuairand.md`). Collapsed as in §4.2 step 7.
- **PopRSP/PopREO citation.** The draft cites Zhu et al. WSDM '21, which defines PRU/PRI. RSP/REO come from Zhu, Wang & Caverlee, SIGIR '20 (see `docs/literature/zhu2021popopp.md`).
- **Mainstreaminess** uses Spearman; the source (Bauer & Schedl, via Kowald et al.) uses Kendall τ.
- **Calibration split usage.** Calibration is used for early stopping as well as hyperparameter selection, the same way for every method.
- **Found during implementation** (not in the draft):
  - *The random log is not statistically uniform over the pool.* Item frequency has CV 0.196 and χ² = 43,944 on 7,343 degrees of freedom (p ≈ 0). Every pool item appears, but the draft's sanity check ("close to uniform") holds only loosely. Report the CV and χ² rather than claim uniformity.
  - *`is_hate` is too rare for a percentile threshold.* See §4.2 step 6: the draft's clickbait rule needs the `> 0` guard to mean anything.
  - *Satisfaction is too sparse for per-user accuracy.* Most candidate sets hold no satisfaction positive. On calibration, `gauc[satisfaction]` and `ndcg@k[satisfaction]` exclude 7,357 of 8,186 users, so these cells rest on about 10% of users.
  - *Bootstrap CIs of set-based metrics are biased* (§7.4 caveat). The draft asks for a user-level bootstrap CI on every cell. For `coverage`, and potentially `gini` and `exposure_entropy`, that CI can fail to cover the point estimate. This was observed for `coverage`.
  - *The default search grid is too narrow for MF on click.* The tuning winner (`lr = 1e-2`, `weight_decay = 1e-5`) sits on a corner of the grid. Two other points (`weight_decay = 1e-5` with `lr = 1e-3` and with `lr = 3e-3`) reach `max_epochs = 50` with their best epoch at 47–49. A fixed tuning budget only treats methods equally if each method's grid contains its optimum.

## 12. Extension points

| Addition | Where it plugs in |
|---|---|
| `full` split | `data/preprocess.py` split definition; `experiments/*.toml` `split = "full"` |
| PDA / MACR | `TorchRecommender.adjust_scores` (+ training-time popularity term in `loss`) |
| IPS / SNIPS / DR | `sample_weight` / `loss`; diagnostics (max weight, ESS) logged from `training/loop.py` |
| D2Q / duration-quantile label | new `LabelSpec` in `data/labels.py` (+ `extras` in `TrainData`) |
| AutoDebias | a `Recommender` whose `fit` also consumes calibration rows; flagged `uses_calibration_for_training` in MLflow |
| DeepFM | new `Backbone` reading user/video feature tensors |
| Raw watch time training | regression `loss` + selection on NDCG (needs a pipeline decision because the selection rule is GAUC) |
| Paired tests + Holm | `evaluation/bootstrap.py` over `per_user.parquet` of two sweeps |
| New metrics | one `@register_metric` function + one `panel.toml` entry |
