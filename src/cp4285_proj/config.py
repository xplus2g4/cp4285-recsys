"""Configuration: TOML → frozen dataclasses, with `--section.field value` CLI overrides.

Precedence: dataclass defaults < TOML < CLI (spec §8).
"""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# --------------------------------------------------------------------------- data (configs/data.toml)


@dataclass(frozen=True)
class ClickbaitConfig:
    min_support: int = 20
    click_rate_q: float = 0.75
    watch_ratio_q: float = 0.25
    hate_rate_q: float = 0.90


@dataclass(frozen=True)
class DataConfig:
    split_seed: int = 4285
    calibration_frac: float = 0.3
    watch_ratio_cap: float = 1.0
    n_bins: int = 5
    head_frac: float = 0.2
    min_history: int = 5
    clickbait: ClickbaitConfig = field(default_factory=ClickbaitConfig)
    raw_dir: str = "data/raw/KuaiRand-Pure/data"
    processed_dir: str = "data/processed"


# --------------------------------------------------------------------------- pipeline (configs/pipeline.toml)


@dataclass(frozen=True)
class PipelineConfig:
    tuning_budget: int = 9
    sweep_seed: int = 0
    final_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    k_max: int = 100
    mask_train_seen: bool = True
    deterministic: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "kuairand-bias-panel"
    device: str = "cuda"
    runs_dir: str = "outputs/runs"
    sweeps_dir: str = "outputs/sweeps"


# --------------------------------------------------------------------------- recommender (configs/recommenders/<name>.toml)


@dataclass(frozen=True)
class TrainConfig:
    optimizer: Literal["adam"] = "adam"
    lr: float = 3e-3
    weight_decay: float = 1e-6
    batch_size: int = 8192
    max_epochs: int = 50
    patience: int = 5


@dataclass(frozen=True)
class RecommenderConfig:
    name: str
    model: Any  # the recommender's own frozen dataclass (`model_config_cls`)
    train: TrainConfig
    search: tuple[tuple[str, tuple[Any, ...]], ...]  # ordered (key, values); keys resolve to train.* or model.*

    def with_point(self, point: dict[str, Any]) -> RecommenderConfig:
        """Apply a search point (bare keys resolved against `train` first, then `model`)."""
        train_kw, model_kw = {}, {}
        for key, value in point.items():
            section = _search_section(key, self.model)
            (train_kw if section == "train" else model_kw)[key] = value
        train = from_dict(TrainConfig, {**dataclasses.asdict(self.train), **train_kw}, "train")
        model = from_dict(type(self.model), {**dataclasses.asdict(self.model), **model_kw}, "model")
        return dataclasses.replace(self, train=train, model=model)


def _search_section(key: str, model_cfg: Any) -> str:
    in_train = key in {f.name for f in dataclasses.fields(TrainConfig)}
    in_model = key in {f.name for f in dataclasses.fields(model_cfg)}
    if in_train and in_model:
        raise ValueError(f"search key {key!r} is ambiguous (both [train] and [model])")
    if not (in_train or in_model):
        raise ValueError(f"search key {key!r} is neither a [train] nor a [model] field")
    return "train" if in_train else "model"


# --------------------------------------------------------------------------- experiment (configs/experiments/*.toml)


@dataclass(frozen=True)
class ExperimentConfig:
    recommender: str = "mf"
    train_label: str = "click"
    split: Literal["guarded"] = "guarded"
    pipeline_config: str = "configs/pipeline.toml"
    recommender_config: str = ""  # empty → configs/recommenders/<recommender>.toml

    @property
    def recommender_config_path(self) -> Path:
        return Path(self.recommender_config or f"configs/recommenders/{self.recommender}.toml")


@dataclass(frozen=True)
class RunConfig:
    """Everything a single training run depends on."""

    experiment: ExperimentConfig
    pipeline: PipelineConfig
    recommender: RecommenderConfig
    pipeline_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": dataclasses.asdict(self.experiment),
            "pipeline": dataclasses.asdict(self.pipeline),
            "recommender": {
                "name": self.recommender.name,
                "model": dataclasses.asdict(self.recommender.model),
                "train": dataclasses.asdict(self.recommender.train),
                "search": {k: list(v) for k, v in self.recommender.search},
            },
            "pipeline_override": self.pipeline_override,
        }


# --------------------------------------------------------------------------- generic dataclass building


def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is Literal:
        choices = typing.get_args(tp)
        if value not in choices:
            raise ValueError(f"{where}={value!r} not in {choices}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list, got {value!r}")
        (elem, _ellipsis) = typing.get_args(tp)
        return tuple(_coerce(v, elem, f"{where}[]") for v in value)
    if tp is Any:
        return value
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool):
        if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
            raise ValueError(f"{where} must be {tp.__name__}, got {value!r}")
        return value
    return value


def from_dict(cls: type, values: dict[str, Any], where: str) -> Any:
    """Build a (possibly nested) frozen dataclass from a dict; unknown keys and bad types fail."""
    hints = _hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in [{where}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp):
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            kwargs[name] = from_dict(tp, value, f"{where}.{name}")
        else:
            kwargs[name] = _coerce(value, tp, f"{where}.{name}")
    return cls(**kwargs)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --------------------------------------------------------------------------- CLI overrides


def add_override_args(parser: argparse.ArgumentParser, cls: type, prefix: str) -> None:
    """Add `--prefix.field` flags for every (nested) dataclass field, typed from the hints."""
    hints = _hints(cls)
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        flag = f"--{prefix}.{f.name}"
        dest = f"{prefix}.{f.name}"
        if dataclasses.is_dataclass(tp):
            add_override_args(parser, tp, dest)
        elif tp is bool:
            parser.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, default=None)
        elif typing.get_origin(tp) is tuple:
            parser.add_argument(flag, dest=dest, type=typing.get_args(tp)[0], nargs="+", default=None)
        elif typing.get_origin(tp) is Literal:
            parser.add_argument(flag, dest=dest, choices=typing.get_args(tp), default=None)
        elif tp in (int, float, str):
            parser.add_argument(flag, dest=dest, type=tp, default=None)
        elif isinstance(tp, types.UnionType) or tp is Any:
            parser.add_argument(flag, dest=dest, default=None)


def collect_overrides(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """Nested dict of the `--prefix.*` flags that were given."""
    out: dict[str, Any] = {}
    for dest, value in vars(args).items():
        if value is None or not dest.startswith(prefix + "."):
            continue
        node = out
        *parents, leaf = dest[len(prefix) + 1 :].split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


# --------------------------------------------------------------------------- loaders


def load_data_config(path: str | Path | None, overrides: dict | None = None) -> DataConfig:
    values = read_toml(path) if path is not None else {}
    return from_dict(DataConfig, _merge(values, overrides or {}), "data")


def load_pipeline_config(path: str | Path, overrides: dict | None = None) -> PipelineConfig:
    return from_dict(PipelineConfig, _merge(read_toml(path), overrides or {}), "pipeline")


def load_recommender_config(
    name: str, path: str | Path, model_overrides: dict | None = None, train_overrides: dict | None = None
) -> RecommenderConfig:
    from cp4285_proj.recommenders.registry import get_recommender

    cls = get_recommender(name)
    values = read_toml(path)
    unknown = set(values) - {"model", "train", "search"}
    if unknown:
        raise ValueError(f"unknown sections in {path}: {sorted(unknown)}")
    model = from_dict(cls.model_config_cls, _merge(values.get("model", {}), model_overrides or {}), "model")
    train = from_dict(TrainConfig, _merge(values.get("train", {}), train_overrides or {}), "train")
    search = []
    for key, grid in values.get("search", {}).items():
        _search_section(key, model)
        if not isinstance(grid, list) or not grid:
            raise ValueError(f"[search].{key} must be a non-empty list")
        search.append((key, tuple(grid)))
    cfg = RecommenderConfig(name=name, model=model, train=train, search=tuple(search))
    for key, grid in cfg.search:  # every grid value must type-check
        for v in grid:
            cfg.with_point({key: v})
    return cfg


def load_run_config(experiment_path: str | Path, overrides: dict[str, dict] | None = None) -> RunConfig:
    """Resolve experiment + pipeline + recommender configs. `overrides` keys: experiment, pipeline, model, train."""
    from cp4285_proj.data.labels import LABELS
    from cp4285_proj.recommenders.registry import get_recommender

    overrides = overrides or {}
    exp = from_dict(ExperimentConfig, _merge(read_toml(experiment_path), overrides.get("experiment", {})), "experiment")
    if exp.train_label not in LABELS:
        raise ValueError(f"unknown train_label {exp.train_label!r}; known labels: {sorted(LABELS)}")
    cls = get_recommender(exp.recommender)
    if exp.train_label not in cls.supported_labels:
        raise ValueError(
            f"train_label {exp.train_label!r} is not supported by recommender {exp.recommender!r} "
            f"(supported: {list(cls.supported_labels)})"
        )
    pipeline = load_pipeline_config(exp.pipeline_config, overrides.get("pipeline"))
    rec = load_recommender_config(
        exp.recommender, exp.recommender_config_path, overrides.get("model"), overrides.get("train")
    )
    return RunConfig(experiment=exp, pipeline=pipeline, recommender=rec, pipeline_override=bool(overrides.get("pipeline")))


def build_run_arg_parser(description: str, argv: list[str] | None) -> argparse.ArgumentParser:
    """Two-phase parser: the experiment's recommender decides which `--model.*` flags exist."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("experiment")
    pre.add_argument("--experiment.recommender", dest="experiment.recommender", default=None)
    known, _ = pre.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("experiment", help="experiment TOML (configs/experiments/*.toml)")
    add_override_args(parser, ExperimentConfig, "experiment")
    add_override_args(parser, PipelineConfig, "pipeline")
    add_override_args(parser, TrainConfig, "train")
    name = getattr(known, "experiment.recommender", None)
    if name is None and Path(known.experiment).exists():
        name = read_toml(known.experiment).get("recommender")
    if name is not None:
        from cp4285_proj.recommenders.registry import get_recommender

        add_override_args(parser, get_recommender(name).model_config_cls, "model")
    return parser


def run_overrides(args: argparse.Namespace) -> dict[str, dict]:
    return {s: collect_overrides(args, s) for s in ("experiment", "pipeline", "model", "train")}


# --------------------------------------------------------------------------- TOML writing (resolved config snapshot)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} to TOML")


def dump_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for nested dicts of scalars/lists (tables after scalars)."""
    lines: list[str] = []

    def emit(table: dict[str, Any], path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if path and (scalars or not tables):
            lines.append(f"[{'.'.join(path)}]")
        for k, v in scalars.items():
            lines.append(f"{k} = {_toml_value(v)}")
        if scalars:
            lines.append("")
        for k, v in tables.items():
            emit(v, [*path, k])

    emit(data, [])
    return "\n".join(lines).rstrip() + "\n"


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
