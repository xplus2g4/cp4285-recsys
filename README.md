# cp4285-proj: bias-panel pipeline on KuaiRand-Pure

Iteration 1 of the pipeline specified in [`docs/spec_pipeline.md`](docs/spec_pipeline.md) (research context:
[`docs/draft_methodology.md`](docs/draft_methodology.md)). It preprocesses KuaiRand-Pure once into a frozen split,
trains recommenders through one interface, and evaluates them with an editable metric panel on the
random-exposure log.

## Recommenders

| Name | Method | Experiment files | Frontier knob (one sweep per value) |
|---|---|---|---|
| `mf` | Naive MF | `mf_<label>.toml` | — |
| `pda` | PD / PDA (Zhang et al. SIGIR '21), spec §13.1 | `pda_click.toml` | `--model.gamma g` with `--model.gamma_infer 0` (PD) or `g` (PDA) |

Every method tunes the shared MF backbone on the same `lr × weight_decay` grid. A frontier point is one
`kr-sweep` with the knob fixed on the command line, e.g.
`uv run kr-sweep configs/experiments/pda_click.toml --model.gamma 0.1 --model.gamma_infer 0.1`.

## Setup

```bash
uv sync                      # Python 3.12, torch from the pytorch-cu128 index (needed for the RTX 5080 / sm_120)
uv run python -c "import torch; print(torch.cuda.is_available())"
```

The raw data is expected in `data/raw/KuaiRand-Pure/data/*.csv` (git-ignored). `video_features_statistic_pure.csv`
is never read: it is averaged over both log windows and leaks the test window.

Tracking goes to MLflow on `sqlite:///mlflow.db`, experiment `kuairand-bias-panel`
(`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). Run artifacts are in `outputs/runs/<run_id>/`
(the MLflow run id), sweep summaries in `outputs/sweeps/<sweep_id>/`.

## Commands

| Command | What it does |
|---|---|
| `uv run kr-preprocess --split guarded` | Checks raw row counts, drops the 239 items without duration, splits the random log 30/70 by user into calibration/test, computes train-only item/user metadata, writes `data/processed/guarded/` and `manifest.json` (hashes, counts, sanity checks). Deterministic. |
| `uv run kr-train configs/experiments/mf_click.toml [--seed N] [--final]` | Trains one configuration on the GPU with early stopping on calibration GAUC; writes `best.pt`, `config.toml`, `fit.json` and calibration predictions. |
| `uv run kr-sweep configs/experiments/mf_click.toml` | Tuning stage (the recommender's `[search]` grid, at most `tuning_budget` points, seed 0), then the best point retrained with `final_seeds`; final runs are tagged `final=true`, `sweep_id=<id>`. |
| `uv run kr-evaluate <run_id> --split calibration\|test [--reason TEXT]` | Runs `configs/panel.toml` on saved predictions; writes `eval/<split>/<panel_hash>/{per_user.parquet,summary.json,bootstrap.json}`. Test evaluation needs a finished `final=true` run, writes test predictions on first use, and counts evaluations (`test_eval_count`, `eval/test/log.jsonl`). |
| `uv run kr-evaluate --sweep <sweep_id> --split test` | Evaluates every final run of a sweep and writes `outputs/sweeps/<id>/eval/test/<panel_hash>/aggregate.json` (mean ± std over seeds, bootstrap CI of the seed-mean); logged to the sweep's parent MLflow run. |
| `uv run kr-rescore <run_id> --k 200` | Reloads `best.pt` and rewrites `topk.parquet` with a larger `K_max`, for panel entries with `k > K_max`. |

Config precedence: dataclass defaults < TOML < CLI `--section.field value` (e.g. `--train.lr 0.01`,
`--model.embedding_dim 32`, `--data.split_seed 1`). `--pipeline.*` overrides are for debugging and tag the run
`pipeline_override=true`.

Tests: `uv run pytest` (the split tests need the raw data and are skipped without it).

## Editing the metric panel

`configs/panel.toml` is the only place the panel is defined. Changes need no retraining: rerun `kr-evaluate`,
which reuses `pred/` and writes results under a new panel hash.

- **Drop a metric:** delete its `[[metric]]` table.
- **Add a metric:** write one function and add one table. For example, in any module under
  `src/cp4285_proj/metrics/` (new modules must be added to `_ensure_builtins` in `metrics/registry.py`):

  ```python
  @register_metric("head_share", kind="per_user", params={"k": int}, needs_label=False)
  def head_share(b: EvalBundle, *, k: int) -> np.ndarray:
      return b.items.is_head[b.topk[:, :k]].mean(1, dtype=np.float64)
  ```

  ```toml
  [[metric]]
  name = "head_share"
  k = [10, 20]
  ```

  Kinds: `per_user` returns a `[U]` float array (NaN = user excluded); `global` returns `dict[str, float]`;
  `per_group` returns group → value plus `"gap"`. `needs_label=True` adds a `labels` parameter
  (`"all"` or a list); `needs_relevance=True` restricts it to labels with a binary relevance; `weight="candidates"`
  aggregates a per-user metric weighted by candidate count (as GAUC does). Per-user quantities that global or
  per-group metrics reuse should go through `b.user_cache(key, fn)` so the bootstrap stays fast.

Validation runs before anything is computed: unknown metric names, unknown or missing parameters, unknown labels,
and `k` beyond the saved top-K width (the error names the `kr-rescore` command) all fail immediately.

## Adding a recommender

1. Create `src/cp4285_proj/recommenders/<name>.py` with a model-config dataclass and a `TorchRecommender` subclass:

   ```python
   @dataclass(frozen=True)
   class MyModelConfig:
       embedding_dim: int = 64

   @register_recommender("my_method")
   class MyMethod(TorchRecommender):
       model_config_cls = MyModelConfig
       supported_labels = ("click", "long_view", "satisfaction", "watch_ratio")

       def build_backbone(self, n_users, n_items):
           return MFBackbone(n_users, n_items, self.model_cfg.embedding_dim)

       # override only what the method changes: batch_loss, sample_weight, loss, adjust_scores (PDA: batch_loss + adjust_scores)
   ```

   Anything that is not a `TorchRecommender` only has to satisfy the `Recommender` protocol in
   `recommenders/base.py` (`fit`, `score`, `recommend`, `save`, `load`).
2. Import the module in `_ensure_builtins` in `recommenders/registry.py`.
3. Add `configs/recommenders/<name>.toml` (`[model]`, `[train]`, `[search]`) and an experiment file
   `configs/experiments/<name>_<label>.toml` with `recommender = "<name>"`.

The pipeline-owned settings (tuning budget, selection rule, seeds, `K_max`, masking) stay in
`configs/pipeline.toml` and are the same for every method.
