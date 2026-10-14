"""Metric registry (spec §7.2).

Return types by kind:
  per_user  → np.ndarray float64 [U] aligned with bundle.users; NaN = user excluded for this metric
  global    → dict[str, float]  (one or more named values, e.g. {"value": g} or {"mean": …, "delta_mean": …})
  per_group → dict[str, float]  group name → value, plus "gap" (max − min over groups other than no_history)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from cp4285_proj.evaluation.bundle import EvalBundle, ItemMeta  # noqa: F401  (re-exported contract types)

MetricKind = Literal["per_user", "global", "per_group"]
Weighting = Literal["uniform", "candidates"]


@dataclass(frozen=True)
class MetricDef:
    name: str
    fn: Callable[..., Any]
    kind: MetricKind
    params: dict[str, Any]  # param name → type (int/float/str or a Literal[...] of allowed values)
    needs_label: bool
    needs_relevance: bool  # label must have a binary relevance; labels="all" skips labels without one
    weight: Weighting  # per_user aggregation: plain mean, or weighted by |C_u|


METRICS: dict[str, MetricDef] = {}


def register_metric(
    name: str,
    *,
    kind: MetricKind,
    params: dict[str, Any],
    needs_label: bool,
    needs_relevance: bool = False,
    weight: Weighting = "uniform",
) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in METRICS:
            raise ValueError(f"metric {name!r} registered twice")
        if weight != "uniform" and kind != "per_user":
            raise ValueError(f"metric {name!r}: weight only applies to per_user metrics")
        if needs_relevance and not needs_label:
            raise ValueError(f"metric {name!r}: needs_relevance requires needs_label")
        METRICS[name] = MetricDef(name, fn, kind, dict(params), needs_label, needs_relevance, weight)
        return fn

    return deco


def _ensure_builtins() -> None:
    from cp4285_proj.metrics import accuracy, clickbait, coverage, duration, groups, popularity  # noqa: F401


def get_metric(name: str) -> MetricDef:
    _ensure_builtins()
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; registered: {sorted(METRICS)}") from None


def all_metrics() -> dict[str, MetricDef]:
    _ensure_builtins()
    return dict(METRICS)
