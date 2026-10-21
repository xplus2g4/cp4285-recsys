"""LabelSpec registry (spec §4.2 step 3). Every label is computed on every log row."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


@dataclass(frozen=True)
class LabelSpec:
    name: str
    kind: Literal["binary", "graded"]
    dtype: type[pl.DataType]
    expr: Callable[[float], pl.Expr]  # raw log columns → label column, given watch_ratio_cap
    relevance: Callable[[np.ndarray], np.ndarray] | None  # label values → bool; None = no binary relevance
    ndcg_gain: Literal["value", "user_minmax"] = "value"  # NDCG gain: label value, or min-max scaled within user
    unit_interval: bool = True  # values lie in [0, 1], so the label is a BCE target; False → regression target

    @property
    def has_relevance(self) -> bool:
        return self.relevance is not None

    def binary(self, values: np.ndarray) -> np.ndarray:
        if self.relevance is None:
            raise ValueError(f"label {self.name!r} has no binary relevance")
        return self.relevance(values)


def _eq1(v: np.ndarray) -> np.ndarray:
    return v == 1


def _full_play(v: np.ndarray) -> np.ndarray:
    return v >= 1.0


LABELS: dict[str, LabelSpec] = {
    spec.name: spec
    for spec in (
        LabelSpec("click", "binary", pl.Int8, lambda cap: pl.col("is_click").cast(pl.Int8), _eq1),
        LabelSpec("long_view", "binary", pl.Int8, lambda cap: pl.col("long_view").cast(pl.Int8), _eq1),
        LabelSpec(
            "satisfaction",
            "binary",
            pl.Int8,
            lambda cap: (
                ((pl.col("is_like") == 1) | (pl.col("is_follow") == 1) | (pl.col("is_forward") == 1) | (pl.col("is_comment") == 1))
                & (pl.col("is_hate") != 1)
            ).cast(pl.Int8),
            _eq1,
        ),
        LabelSpec(
            "watch_ratio",
            "graded",
            pl.Float32,
            lambda cap: pl.min_horizontal(pl.col("play_time_ms") / pl.col("duration_ms"), pl.lit(cap)).cast(pl.Float32),
            _full_play,
        ),
        LabelSpec("raw_watch_time", "graded", pl.Int64, lambda cap: pl.col("play_time_ms").cast(pl.Int64), None, "user_minmax", False),
    )
}


def get_label(name: str) -> LabelSpec:
    try:
        return LABELS[name]
    except KeyError:
        raise ValueError(f"unknown label {name!r}; known labels: {sorted(LABELS)}") from None
