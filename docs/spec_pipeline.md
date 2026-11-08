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
- the metric registry and the §6 panel (accuracy plus one metric per bias, §7.3) except the IPS-only diagnostics;
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
    naive_mf.py        # MFBackbone, NaiveMF (§5.3)
    macr.py            # MACR (§13.1)
    d2q.py             # D2Q (§13.2)
    ips_feature.py     # IPS with a feature-based propensity (§13.3)
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
    popularity.py      # arp
    exposure.py        # gini
    duration.py        # wasserstein_duration, jsd_dur, dur_bin_share (duration quintiles)
    common.py          # shared vectorised helpers (top-k lists, recommendation frequencies, binary relevance)
configs/
  data.toml            # preprocess parameters
  pipeline.toml        # pipeline-fixed settings (tuning budget, seeds, K_max, selection rule)
  panel.toml           # the editable metric panel
  recommenders/*.toml  # per-recommender hyperparameters + search space
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

   | Label | Definition | Kind | Binary relevance used by GAUC / Wasserstein |
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
   - `is_cold`: `pop_exposure == 0`.
7. **User metadata.** Computed on the train split only.
   - `n_train_rows` / `n_train_items`: the user's train rows and distinct train items. Users with no train rows are still scored (§5.3). Among the random-log users, 1,075 have no train rows in the raw logs (checked at runtime) and 1,113 after the item drop, because 38 users had train rows only on dropped items.
   - `role`: `calibration` | `test` | `train_only`. In this release every one of the 27,285 users is in the random log, so `train_only` is empty.
8. **Categorical side features** (static metadata, no log rows; used by the feature propensity of §13.3 and a later DeepFM backbone). `user_features.parquet` and `item_features.parquet` hold one Int32 code per field, where 0 = missing and 1..K are the distinct raw values sorted as strings.
   - User fields (`schema.USER_FEATURES`): `user_active_degree`, `is_live_streamer`, `is_video_author`, the four `*_range` columns (follow, fans, friend, register days) and `onehot_feat0`–`onehot_feat17`. Raw counts are left out because their `*_range` bins are kept, and `is_lowactive_period` is constant in this release.
   - Item fields (`schema.ITEM_FEATURES`): `video_type`, `upload_type`, `upload_dt`, `music_type`, `tag`, `orientation` (portrait / landscape / square, from `server_height` against `server_width`) and `duration_decile`. The deciles are 10 equal-size bins of the pool ranked by `(duration_ms, video_id)`, i.e. catalogue bins that use no log rows. Left out: `author_id` and `music_id`, which are near-identifiers, and `visible_status`, which is constant.
9. **Manifest and sanity checks** (`manifest.json`):
   - hashes of the input files;
   - `split_seed` and the config snapshot;
   - row counts per part;
   - user counts per part;
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
is_cold: bool

# users.parquet (one row per user, sorted by user_idx)
user_idx: int32
user_id: int32
role: str                # calibration | test | train_only
n_train_rows: int32
n_train_items: int32

# user_features.parquet / item_features.parquet (one row per user / pool item, sorted by user_idx / item_idx)
user_idx | item_idx: int32
<field>: int32               # one column per schema.USER_FEATURES / schema.ITEM_FEATURES field; 0 = missing

# manifest.json (see §4.2 step 9)
```

`test.parquet` is read only through `data.schema.load_test(token)` (see §9).

## 5. Steps 2–3: the recommender abstraction and training

### 5.1 Ownership

| Owned by the **recommender** (`configs/recommenders/<name>.toml`) | Owned by the **pipeline** (`configs/pipeline.toml`, the same for every method) |
|---|---|
| model hyperparameters, **including embedding size** | training label (experiment config, applied outside the recommender) |
| optimizer, batch size, max epochs, patience | tuning budget (number of trials) |
| search space (`[search]`) | selection rule: calibration GAUC under the training label's binary relevance; for a label without one (`raw_watch_time`), calibration NDCG@10 under that label (§13.2) |
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
    items: "ItemTensors"        # pop_exposure, duration_ms on device, indexed by item_idx
    user_features: torch.Tensor # int64 [n_users, F_u] categorical codes (§4.2 step 8)
    item_features: torch.Tensor # int64 [n_items, F_i]
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
    Naive MF uses only the defaults."""
    backbone: Backbone

    def build_backbone(self, n_users: int, n_items: int) -> Backbone: ...
    def configure_optimizer(self) -> torch.optim.Optimizer: ...       # default: Adam(lr, weight_decay)

    # hooks (defaults are the naive behaviour)
    def batch_loss(self, batch: "Batch") -> torch.Tensor:             # default: loss(backbone(u, i), target, sample_weight(batch))
        ...                                                            # overridden when the training score or target differs (MACR, D2Q)
    def sample_weight(self, batch: "Batch") -> torch.Tensor | None:   # default None → unweighted (IPS overrides, §13.3)
        return None
    def loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        ...                                                            # default: BCEWithLogits, mean (Σ w·ℓ / Σ w if weight)
    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        return logits                                                  # inference-time correction (MACR, D2Q)

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
- Construction is not part of the pipeline surface. A `TorchRecommender` is built as `cls(model_cfg, train_cfg)` and declares two class attributes: `model_config_cls` (the frozen dataclass for its `[model]` section) and `supported_labels` (checked at config load, which is how D2Q rejects every label but `raw_watch_time`).
- `Backbone` declares `n_users` and `n_items` (used by `save` and the default `score_all`).

### 5.3 Naive MF (`recommenders/naive_mf.py`)

- **Backbone:** `logit(u,i) = ⟨p_u, q_i⟩ + b_u + b_i + b`.
  - Embeddings are initialised with N(0, 0.01²) and biases with zero.
  - `score_all` is `P[u] @ Qᵀ + b_u + b_I + b`.
- **Loss:** pointwise BCE on logged exposures. There is no negative sampling, because the logs already contain the negatives.
  - For `watch_ratio`, the capped value is the soft target.
  - For `raw_watch_time` (a `LabelSpec` with `unit_interval = False`), the loss is MSE on watch time divided by its train mean instead (§13.2).
  - Supported training labels: `click`, `long_view`, `satisfaction`, `watch_ratio`, `raw_watch_time`.
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
- **Per batch:** `batch_loss` (by default forward → `sample_weight` → `loss`) → step.
- **After each epoch:** `selection = ctx.selection(model)`. This is calibration GAUC under the training label's binary relevance (calibration NDCG@10 for a label without one, §13.2), computed from `score` over the calibration candidate rows. The best `state_dict` is kept in memory. Training stops when `patience` epochs pass without improvement or `max_epochs` is reached.
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
- **Expansion.** Each list-valued parameter, plus `labels = "all"`, is expanded into a cross product. Each instance gets a canonical id, e.g. `ndcg@10[click]`, `gini@20`, `wasserstein_duration@10[click]`. Values of `Literal`-typed parameters follow the label, e.g. `<name>@10[click|a]`.
- **Validation before computing anything.** An unknown metric name, a missing or unknown parameter, `k > K_max`, or an unknown label fails before any metric is computed. `K_max` here is the width of the run's saved `topk.parquet`, so it grows after `kr-rescore`. A label without a binary relevance given explicitly to a metric that needs one also fails; under `labels = "all"` such labels are skipped.
- **Panel hash.** The SHA-256 of the expanded, canonicalised panel is stored with every result. Results are written to `outputs/runs/<run_id>/eval/<split>/<panel_hash>/`. The canonical form is the JSON (sorted keys) of `eval_labels`, the `bootstrap` settings and the instances sorted by id. Reordering `[[metric]]` tables therefore keeps the hash; changing entries or bootstrap settings changes it.

### 7.2 Metric contract

```python
# metrics/registry.py
MetricKind = Literal["per_user", "global"]

def register_metric(name: str, *, kind: MetricKind, params: dict[str, type], needs_label: bool,
                    needs_relevance: bool = False, weight: Literal["uniform", "candidates"] = "uniform") -> Callable: ...
# params: every parameter except `label` (e.g. {"k": int}); a Literal[...] type lists allowed values.
# needs_label adds the `labels` parameter. needs_relevance: the label must have a binary relevance, so
#   labels = "all" can skip labels without one at validation time (gauc, wasserstein_duration).
# weight: how a per_user metric is aggregated; "candidates" weights users by |C_u| (gauc).
# These two keywords were added because expansion/validation and GAUC's aggregation need them and §7.3 implies them.

# Return types by kind:
#   per_user  → np.ndarray float64 [U] aligned with bundle.users; NaN = user excluded for this metric
#   global    → dict[str, float]           (one or more named values, e.g. {"value": g})

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
    items: "ItemMeta"                 # numpy arrays by item_idx: duration_ms, pop_exposure
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

Per-user metrics report `n_excluded` next to their value. Training diagnostics, such as the IPS maximum weight and ESS (§13.3), are **not** panel metrics. They are logged by the training loop.

### 7.3 Iteration-1 panel

Notation:
- `L_u`: user u's top-k list from the full pool.
- `C_u`: u's candidate rows (random exposures in the split).
- `rel`: the label's binary relevance.
- `pop(i)` = `pop_exposure`.

Accuracy is `gauc` and `ndcg`. Each bias has one metric: popularity `arp`, exposure `gini`, duration `wasserstein_duration`.

| Id | Kind | Params | Definition |
|---|---|---|---|
| `gauc` | per_user → aggregated with candidate-count weights | label | AUC of `score` against `rel` within `C_u`. NaN if `C_u` has only one class. The aggregate is GAUC = Σ_u \|C_u\|·AUC_u / Σ_u \|C_u\| over non-NaN users. Labels with no binary relevance (`raw_watch_time`) are skipped. |
| `xgauc` | per_user → aggregated with candidate-count weights | label | XAUC (Zhan et al. KDD '22) within `C_u`: the share of row pairs with different label values that `score` orders the same way; score ties count ½. NaN if all label values in `C_u` are equal. The aggregate is XGAUC = Σ_u \|C_u\|·XAUC_u / Σ_u \|C_u\| over non-NaN users. The panel uses it on `raw_watch_time`. |
| `ndcg` | per_user | k, label | NDCG@k ranking `C_u` by score. The gain is the label value: 0/1, capped watch ratio, or `raw_watch_time` min-max scaled within the user. The ideal is taken over `C_u`. Lists are truncated at min(k, \|C_u\|). NaN if every gain is 0 or every gain is equal. |
| `arp` | per_user | k | popularity: mean over i ∈ `L_u` of pop(i) |
| `gini` | global | k | exposure: Gini coefficient of item recommendation frequency over all `n_items` pool items, zeros included |
| `wasserstein_duration` | global | k, label | duration: `scipy.stats.wasserstein_distance` between the durations (seconds) of pooled `L_u` items and of relevant candidate rows |
| `jsd_dur` | per_user | k, label | duration: User Popularity Deviation style (Abdollahpouri et al., UMAP '21), over 5 duration quintiles (items ranked by (`duration_ms`, item_idx); bin = rank · 5 // n_items): Jensen–Shannon divergence, base 2 (so in [0, 1]), between p = the share of the user's relevant candidate rows in each bin and q = the share of `L_u` in each bin. The target is the user's engagement on random exposure. NaN for users with no relevant candidate row. |
| `dur_bin_share` | global | k, label | duration: `{list_j}`: share of all `L_u` slots (pooled) in duration quintile j; `{ref_j}`: share of all relevant candidate rows in quintile j. The pooled ingredients of `jsd_dur`, for plots. |

Per-user values are averaged over non-NaN users unless the table says otherwise (as for `gauc`).

### 7.4 Statistics

- **Per run** (`kr-evaluate <run_id> --split calibration|test`), output to `eval/<split>/<panel_hash>/`:
  - `per_user.parquet`: `user_idx`, `n_candidates`, then one column per per-user metric instance;
  - `summary.json`: point estimates and exclusion counts;
  - `bootstrap.json`: a 95% percentile CI from 1,000 user resamples, with a fixed seed. Per-user metrics resample the saved per-user vectors. Global metrics are recomputed on `bundle.resample(positions)`. The raw draws are kept in `bootstrap_samples.npy` (`[B, S]`, columns named in `bootstrap.json`).
  - Scalar names: a per-user or single-valued global instance is named by its id; a multi-valued global instance becomes `id.key`.
  - *Caveat.* Resampling users with replacement leaves about 63% distinct users, so set-based global metrics such as `gini` are biased in the resamples, and their percentile CI can exclude the point estimate. Read those CIs as variability, not as coverage intervals.
- **Per sweep** (`kr-evaluate --sweep <sweep_id> --split test`): evaluates every `final` run, then writes `aggregate.json` with the mean ± std (ddof = 1) over seeds of every point estimate. The aggregate is logged to the sweep's parent MLflow run (`stage=sweep`). The CI in `aggregate.json` is the percentile CI of the seed-mean: all runs share the same users and bootstrap seed, so resample b uses the same positions in every run, and the CI is taken over the per-resample mean across runs.
- **Extension point:** paired per-user tests against the naive baseline, with Holm correction across the panel. The per-user Parquet files already hold the input those tests need.

## 8. Configuration

The precedence chain is: dataclass defaults < TOML < CLI `--section.field value`.

```toml
# configs/data.toml
split_seed = 4285
calibration_frac = 0.3
watch_ratio_cap = 1.0

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
4. The manifest contains every count and check in §4.2 step 9.
5. Item popularity and user train counts depend only on train rows: changing any calibration or test label leaves `items.parquet` and `users.parquet` unchanged.
6. `kr-train configs/experiments/mf_click.toml` trains Naive MF on the GPU and writes everything in §6 for calibration. The calibration GAUC at the best epoch is above 0.5. With `deterministic = true`, two runs with the same seed give the same `best_selection_score`.
7. `kr-train` with a training label that the recommender does not support (e.g. `click` for `d2q`) fails at config load with an error naming the unsupported label.
8. `kr-sweep` runs at most `tuning_budget` tuning runs and then 5 final runs, all visible in MLflow with the tags in §5.5.
9. `kr-evaluate <final_run> --split test` writes the test predictions and the panel output. Evaluating a non-final run on test raises an error. A second test evaluation warns and records `test_eval_count = 2`.
10. Adding a `[[metric]]` entry for a newly registered function, or deleting an entry, then rerunning `kr-evaluate`, gives results under a new panel hash. No retraining and no rescoring happen, because the existing `pred/` files are reused.
11. An unknown metric name, an unknown parameter, `k > K_max` or an unknown label in `panel.toml` fails before any metric is computed.
12. Metric correctness: unit tests with small hand-computed fixtures cover `gauc` (including single-class exclusion and candidate-count weighting), `xgauc` (score ties, equal-label exclusion and candidate-count weighting), `ndcg` (binary, graded and truncation), `gini`, `wasserstein_duration` and `jsd_dur`/`dur_bin_share` (including the no-positive exclusion).
13. `kr-evaluate --sweep <id> --split test` writes the mean ± std over the 5 seeds and a 95% bootstrap CI for every panel instance.
14. `kr-rescore <run_id> --k 200` rewrites `topk.parquet` with 200 columns, and a panel entry with `k = 200` then passes validation.

## 11. Departures from `draft_methodology.md`, and corrections

- **Embedding size is chosen by each recommender (§5).** The draft asks for "the same embedding size and tuning budget". Only the tuning budget is held fixed. Fix the paper text, or set the same `embedding_dim` in every recommender config.
- **No sensitivity run on all standard data (§2).** Only `guarded` is built. `kr-preprocess` takes the split name as a parameter so that `full` can be added later.
- **239 items dropped.** These items have no duration at all. The random log is therefore uniform over 7,344 items, not 7,583. List this in §10 threats.
- **`user_active_degree` has 9 levels, not 4** (the 4 levels are stated in `docs/literature/gao2022kuairand.md`). It enters only as a categorical side feature (§4.2 step 8), one code per level.
- **Calibration split usage.** Calibration is used for early stopping as well as hyperparameter selection, the same way for every method.
- **Found during implementation** (not in the draft):
  - *The random log is not statistically uniform over the pool.* Item frequency has CV 0.196 and χ² = 43,944 on 7,343 degrees of freedom (p ≈ 0). Every pool item appears, but the draft's sanity check ("close to uniform") holds only loosely. Report the CV and χ² rather than claim uniformity.
  - *Satisfaction is too sparse for per-user accuracy.* Most candidate sets hold no satisfaction positive. On calibration, `gauc[satisfaction]` and `ndcg@k[satisfaction]` exclude 7,357 of 8,186 users, so these cells rest on about 10% of users.
  - *Bootstrap CIs of set-based metrics are biased* (§7.4 caveat). The draft asks for a user-level bootstrap CI on every cell. For a set-based metric such as `gini`, that CI can fail to cover the point estimate.
  - *The default search grid is too narrow for MF on click.* The tuning winner (`lr = 1e-2`, `weight_decay = 1e-5`) sits on a corner of the grid. Two other points (`weight_decay = 1e-5` with `lr = 1e-3` and with `lr = 3e-3`) reach `max_epochs = 50` with their best epoch at 47–49. A fixed tuning budget only treats methods equally if each method's grid contains its optimum.

## 12. Extension points

| Addition | Where it plugs in |
|---|---|
| `full` split | `data/preprocess.py` split definition; `experiments/*.toml` `split = "full"` |
| MACR | done: `TorchRecommender.adjust_scores` + `batch_loss` (§13.1) |
| IPS | done: `sample_weight` / `loss`; diagnostics (max weight, ESS) logged once per run from `fit` (§13.3) |
| D2Q / duration-quantile label | done: `batch_loss` with a target built in `fit` + `adjust_scores` (§13.2) |
| DeepFM | new `Backbone` reading `TrainData.user_features` / `item_features` (§4.2 step 8) |
| Raw watch time training | done: Naive MF regression + NDCG@10 selection (§13.2) |
| Paired tests + Holm | `evaluation/bootstrap.py` over `per_user.parquet` of two sweeps |
| New metrics | one `@register_metric` function + one `panel.toml` entry |

## 13. Recommenders beyond iteration 1

Common protocol, so that only the correction varies:
- Every method uses the MF backbone of §5.3 (embedding size 64, the same initialisation, and the same restore of rows with no train data) and the MF `[search]` grid (`lr × weight_decay`, 9 points, so the tuning budget never samples).
- A method's frontier knob is not in `[search]`. Each knob value is one `kr-sweep` with `--model.<knob> <value>` (tuning, then 5 final seeds), and the curve is the set of those sweeps. Calibration selection always scores the method's inference-time output (`score` → `adjust_scores`).
- The training loop calls `batch_loss(batch)` once per batch (§5.4). Its default is `loss(backbone(u, i), target, sample_weight(batch))`, so Naive MF is unchanged: a deterministic rerun reproduces the iteration-1 tuning scores exactly.

### 13.1 MACR (`recommenders/macr.py`, `configs/recommenders/macr.toml`)

- **Branches** (Eq. 7): `ŷ_k` is the MF backbone logit, unchanged, because MACR is model-agnostic and takes the existing recommender as its matching branch. The item module is `ŷ_i = w_i · q_i + b'_i` and the user module is `ŷ_u = w_u · p_u + b'_u`: linear layers on the backbone's embeddings (the paper: "can be implemented as multi-layer perceptrons").
  - Departure: the paper's MF has no bias terms. Here `ŷ_k` keeps the backbone's per-item bias `b_i`, so the part of the item's direct effect that `b_i` absorbs stays in the matching branch and is not removed at inference. Dropping `b_i` would change the backbone that §13 holds fixed.
- **Training** (Eq. 8): `BCE(ŷ_k σ(ŷ_i) σ(ŷ_u)) + α BCE(ŷ_i) + β BCE(ŷ_u)` on the logged exposures, with the paper's defaults `α = β = 1e-3`.
- **Inference** (Eq. 9): rank by `(ŷ_k − c) σ(ŷ_i) σ(ŷ_u)`. `c = 0` ranks by the total effect, i.e. MACR's training without counterfactual inference.
- **Knob:** `c ∈ {0, 10, 20, 30, 40, 60, 100}`. This covers the paper's tuned range {20, …, 40}. With the paper's α and β, the learned `σ(ŷ_i)` spans only about 0.98–1.00 on `guarded`, so `c` has to be large before `c · Δσ(ŷ_i)` competes with the spread of `ŷ_k` (about −1.6 to 1.0 between the 1st and 99th percentiles). A calibration-only check on one tuning model gave an unmasked top-10 ARP, over 3,000 calibration users, of 1,968 → 869 as `c` goes from 0 to 100, while calibration GAUC fell from 0.5460 to 0.5434.

### 13.2 Raw watch time: VR baseline and D2Q (`recommenders/naive_mf.py`, `recommenders/d2q.py`)

- **Selection for labels without a binary relevance.** `make_selection` uses calibration NDCG@`SELECTION_NDCG_K` (10) under the training label, with the panel's gain (`raw_watch_time` min-max scaled within the user) and the panel's exclusion of users whose gains are all equal. Only `raw_watch_time` is affected; every other label keeps GAUC.
- **VR baseline** (`mf` on `raw_watch_time`, `configs/experiments/mf_raw_watch_time.toml`): the D2Q paper's value regression. MSE between the MF output and watch time divided by its train mean (23.6 s on `guarded`). The scale only keeps targets near 1 for Adam and does not change the ranking.
- **D2Q** (`configs/experiments/d2q_raw_watch_time.toml`, supports only `raw_watch_time`):
  - Train rows are cut into `n_groups` equal-frequency duration groups. Cut `j` is the sorted row duration at rank ⌈(j/M)(N−1)⌉, so it is an actual duration and equal durations always share a group. Fewer groups are formed when durations tie at a cut, and the count is logged as `d2q_groups`.
  - The training target of a row is its watch-time ECDF value within its group, Φ̂_k(w) = #{w' ≤ w} / n_k. The model is one MF, h = σ(f(u, i)), trained with soft-target BCE (the pipeline's convention for [0, 1] targets, e.g. `watch_ratio`). BCE has the same minimiser as the paper's MSE, E[Φ̂_k(w) | u, i]. The item id already identifies the duration group, so the paper's duration input feature is not needed.
  - Inference with `rank_by = "watch_time"` (the published D2Q) ranks by Φ̂_k⁻¹(h). The inverse CDF of each group is stored at 1,001 levels and interpolated linearly. With `rank_by = "quantile"`, it ranks by h, which is the methodology's duration-quantile label. Item groups and inverse CDFs are backbone buffers, saved in `best.pt`. Items outside the train duration range take the first or last group.
- **Knob:** `n_groups ∈ {1, 10, 20, 30, 60, 100}`: the paper's grid {1, 10, 20, 30, 50, 100}, with 50 replaced by 60, the KuaiRand-Pure setting used by D2Co and CWM. Each value is run once per `rank_by`. With `n_groups = 1`, both rankings are monotone in h and differ only through ties of the inverse CDF.

### 13.3 IPS with a feature-based propensity (`recommenders/ips_feature.py`, `configs/recommenders/ips_feature.toml`)

- **Target of the correction:** the loss under uniform item exposure, which is the condition of the random-exposure log. Each logged train row is weighted by its inverse exposure propensity, estimated from user and item side features. Clipping is the only variance control.
- **Propensity model:** a factorization machine (logit = `b + Σ_f w[x_f] + Σ_{f<g} ⟨v[x_f], v[x_g]⟩`, factor size 8) over the 25 user fields and 7 item fields of §4.2 step 8. It has no user or item ids, so it cannot memorise item popularity. Schnabel et al.'s covariate propensity is a logistic regression; the FM is that regression plus pairwise interactions, which are needed for the reason in the next bullet.
- **Propensity training,** inside `fit`, before the MF: every logged train exposure `(u, i)` is a positive and gets one negative `(u, j)`, with `j` uniform over the pool. Negatives are redrawn every epoch. The loss is BCE(pos) + BCE(neg), trained with Adam (lr 1e-2, batch 65,536) for 15 epochs. On `guarded` the loss plateaus by then: 1.2507 after 5 epochs, 1.2123 after 10, 1.2065 after 15. The generator is seeded with the run seed, so the propensity varies with the seed like every other part of the run.
  - With 1:1 matched sampling the odds estimate `P_log(i | u) / P_uniform(i | u)`, so the inverse propensity of a train row is `exp(−logit)`. A popularity propensity `p_i = n_i / max_j n_j` (`n_i = pop_exposure`) estimates the same ratio marginalised over users and without features.
  - Because each negative has the positive's user, user-only terms cannot separate the classes. User fields act only through their interactions with item fields, which makes the propensity user-dependent.
- **Weights:** `w = 1/p` rescaled so that the smallest weight is 1, clipped at `clip`, then divided by the train mean. After rescaling, `clip` is the largest allowed ratio between any two weights. How much a given `clip` clips depends on the tail of the propensity model, so the logged ESS share, not the raw `clip` value, measures how much variance control a setting applies.
- **Loss:** `mean(w · BCE)` over the batch. Because the weights have train mean 1, this is the IPS estimator with its fixed normaliser. It overrides the weighted default of `TorchRecommender.loss`, `Σ w·ℓ / Σ w` per batch, which is the self-normalised estimator.
- **Diagnostics,** logged to MLflow and `train_log.jsonl`. Once per run: `ips_max_weight` (after normalisation), `ips_ess_share` = (Σw)² / (N Σw²), `ips_clipped_share`, `propensity_auc` (the AUC of 200k logged rows against as many fresh uniform negatives) and `propensity_pop_spearman` (the Spearman correlation over train rows between the feature inverse propensity and the popularity inverse propensity `max_j n_j / n_i`). Per propensity epoch: `propensity_loss`. For seed 0 on `guarded`: AUC 0.734 and Spearman 0.459. The unclipped max/min weight ratio is about 6.4·10⁴, against 9,110 for the popularity inverse propensity.
- **Knob:** `clip ∈ {10, 30, 100, 300, 1000, 2000, 4000, 8000, inf}`. The smallest FM weight is an outlier: the 1st percentile of w/min is 68 and the median is 386. So clips up to 300 clip 64–100% of rows and leave the model close to Naive MF. The informative range is 300–8000, with seed-0 ESS shares 0.95, 0.73, 0.59, 0.49 and 0.44 at 300, 1000, 2000, 4000 and 8000, and 0.41 unclipped. Report the logged ESS share next to the `clip` value.
