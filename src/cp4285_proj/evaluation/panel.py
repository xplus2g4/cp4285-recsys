"""`configs/panel.toml` parsing, expansion, validation, hashing and the panel runner (spec §7.1, §7.3)."""

from __future__ import annotations

import hashlib
import itertools
import json
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cp4285_proj.config import read_toml
from cp4285_proj.data.labels import LABELS
from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.registry import MetricDef, get_metric


class PanelError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int = 1000
    seed: int = 0
    ci: float = 0.95


@dataclass(frozen=True)
class MetricInstance:
    id: str
    name: str
    params: tuple[tuple[str, Any], ...]  # sorted (param, value), includes k / label when present

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def k(self) -> int | None:
        return self.kwargs.get("k")

    @property
    def metric(self) -> MetricDef:
        return get_metric(self.name)


@dataclass(frozen=True)
class Panel:
    eval_labels: tuple[str, ...]
    bootstrap: BootstrapConfig
    instances: tuple[MetricInstance, ...]
    hash: str

    @property
    def max_k(self) -> int:
        return max((i.k for i in self.instances if i.k is not None), default=0)

    def canonical(self) -> dict[str, Any]:
        return {
            "eval_labels": list(self.eval_labels),
            "bootstrap": {"resamples": self.bootstrap.resamples, "seed": self.bootstrap.seed, "ci": self.bootstrap.ci},
            "instances": [{"id": i.id, "name": i.name, "params": dict(i.params)} for i in self.instances],
        }


def instance_id(name: str, params: dict[str, Any]) -> str:
    """`ndcg@10[click]`, `rsp_pop@20`, `ndcg_by_group@10[click|mainstream]`, `gauc[click]`."""
    out = name + (f"@{params['k']}" if "k" in params else "")
    tags = ([str(params["label"])] if "label" in params else []) + [
        str(params[p]) for p in sorted(params) if p not in ("k", "label")
    ]
    return out + (f"[{'|'.join(tags)}]" if tags else "")


def _check_value(metric: str, param: str, value: Any, tp: Any) -> None:
    if typing.get_origin(tp) is Literal:
        if value not in typing.get_args(tp):
            raise PanelError(f"metric {metric!r}: {param}={value!r} not in {typing.get_args(tp)}")
    elif tp is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PanelError(f"metric {metric!r}: {param}={value!r} must be a number")
    elif not isinstance(value, tp) or isinstance(value, bool):
        raise PanelError(f"metric {metric!r}: {param}={value!r} must be {tp.__name__}")
    if param == "k" and value < 1:
        raise PanelError(f"metric {metric!r}: k must be >= 1")


def _labels_for(entry: dict[str, Any], m: MetricDef, eval_labels: tuple[str, ...]) -> list[str]:
    spec = entry["labels"]
    if spec == "all":
        return [l for l in eval_labels if not m.needs_relevance or LABELS[l].has_relevance]
    names = [spec] if isinstance(spec, str) else list(spec)
    for l in names:
        if l not in LABELS:
            raise PanelError(f"metric {m.name!r}: unknown label {l!r}; known labels: {sorted(LABELS)}")
        if m.needs_relevance and not LABELS[l].has_relevance:
            raise PanelError(f"metric {m.name!r} needs a binary relevance; label {l!r} has none")
    return names


def parse_panel(data: dict[str, Any]) -> Panel:
    """Expand and validate a panel dict. Every error is raised here, before any metric is computed."""
    unknown_top = set(data) - {"eval_labels", "bootstrap", "metric"}
    if unknown_top:
        raise PanelError(f"unknown panel keys: {sorted(unknown_top)}")
    eval_labels = tuple(data.get("eval_labels", list(LABELS)))
    for l in eval_labels:
        if l not in LABELS:
            raise PanelError(f"eval_labels: unknown label {l!r}; known labels: {sorted(LABELS)}")
    boot = data.get("bootstrap", {})
    if set(boot) - {"resamples", "seed", "ci"}:
        raise PanelError(f"unknown bootstrap keys: {sorted(set(boot) - {'resamples', 'seed', 'ci'})}")
    bootstrap = BootstrapConfig(**boot)
    if not 0 < bootstrap.ci < 1 or bootstrap.resamples < 1:
        raise PanelError(f"invalid bootstrap settings {bootstrap}")

    instances: dict[str, MetricInstance] = {}
    for n, entry in enumerate(data.get("metric", [])):
        if "name" not in entry:
            raise PanelError(f"[[metric]] #{n + 1} has no name")
        try:
            m = get_metric(entry["name"])
        except ValueError as e:
            raise PanelError(str(e)) from None
        given = set(entry) - {"name", "labels"}
        unknown = given - set(m.params)
        if unknown:
            raise PanelError(f"metric {m.name!r}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(m.params)}")
        missing = set(m.params) - given
        if missing:
            raise PanelError(f"metric {m.name!r}: missing parameter(s) {sorted(missing)}")
        if m.needs_label and "labels" not in entry:
            raise PanelError(f"metric {m.name!r}: missing parameter 'labels'")
        if not m.needs_label and "labels" in entry:
            raise PanelError(f"metric {m.name!r}: unknown parameter(s) ['labels'] (metric takes no label)")
        grids: list[tuple[str, list[Any]]] = []
        for p in sorted(m.params):
            values = entry[p] if isinstance(entry[p], list) else [entry[p]]
            if not values:
                raise PanelError(f"metric {m.name!r}: {p} is an empty list")
            for v in values:
                _check_value(m.name, p, v, m.params[p])
            grids.append((p, values))
        if m.needs_label:
            grids.append(("label", _labels_for(entry, m, eval_labels)))
        for combo in itertools.product(*(v for _, v in grids)):
            params = dict(zip((p for p, _ in grids), combo))
            iid = instance_id(m.name, params)
            if iid in instances:
                raise PanelError(f"duplicate panel instance {iid}")
            instances[iid] = MetricInstance(id=iid, name=m.name, params=tuple(sorted(params.items())))

    ordered = tuple(instances[i] for i in sorted(instances))
    panel = Panel(eval_labels=eval_labels, bootstrap=bootstrap, instances=ordered, hash="")
    digest = hashlib.sha256(json.dumps(panel.canonical(), sort_keys=True).encode()).hexdigest()
    return Panel(eval_labels=eval_labels, bootstrap=bootstrap, instances=ordered, hash=digest)


def load_panel(path: str | Path) -> Panel:
    return parse_panel(read_toml(path))


def check_k(panel: Panel, k_available: int, run_id: str) -> None:
    """Refuse instances with k beyond the saved top-K width, naming the rescore command."""
    too_big = [i.id for i in panel.instances if i.k is not None and i.k > k_available]
    if too_big:
        raise PanelError(
            f"panel entries {too_big} need k={panel.max_k} > K_max={k_available} saved for run {run_id}; "
            f"run `kr-rescore {run_id} --k {panel.max_k}` first"
        )


# --------------------------------------------------------------------------- runner


def aggregate_per_user(values: np.ndarray, weights: np.ndarray | None) -> float:
    ok = ~np.isnan(values)
    if not ok.any():
        return float("nan")
    if weights is None:
        return float(values[ok].mean())
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def user_weights(m: MetricDef, b: EvalBundle) -> np.ndarray | None:
    return b.n_candidates.astype(np.float64) if m.weight == "candidates" else None


def scalarize(inst: MetricInstance, value: dict[str, float]) -> dict[str, float]:
    """Global/per-group dict → flat scalar names: `id` for {"value": v}, else `id.key`."""
    if set(value) == {"value"}:
        return {inst.id: float(value["value"])}
    return {f"{inst.id}.{k}": float(v) for k, v in value.items()}


@dataclass
class PanelResult:
    per_user: dict[str, np.ndarray]  # per_user instance id → [U]
    scalars: dict[str, float]  # every scalar point estimate
    n_excluded: dict[str, int]  # per_user instance id → NaN users
    kinds: dict[str, str]  # scalar name → metric kind


def run_panel(panel: Panel, b: EvalBundle) -> PanelResult:
    per_user, scalars, excluded, kinds = {}, {}, {}, {}
    for inst in panel.instances:
        m = inst.metric
        value = m.fn(b, **inst.kwargs)
        if m.kind == "per_user":
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (len(b.users),):
                raise ValueError(f"{inst.id}: per_user metric returned shape {value.shape}, expected ({len(b.users)},)")
            per_user[inst.id] = value
            scalars[inst.id] = aggregate_per_user(value, user_weights(m, b))
            excluded[inst.id] = int(np.isnan(value).sum())
            kinds[inst.id] = m.kind
        else:
            for name, v in scalarize(inst, value).items():
                scalars[name] = v
                kinds[name] = m.kind
    return PanelResult(per_user=per_user, scalars=scalars, n_excluded=excluded, kinds=kinds)
