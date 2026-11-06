"""Exposure metric (spec §7.3): Gini coefficient of item recommendation frequency."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import rec_freq
from cp4285_proj.metrics.registry import register_metric


def gini_coefficient(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * x).sum() / (n * x.sum()))


@register_metric("gini", kind="global", params={"k": int}, needs_label=False)
def gini(b: EvalBundle, *, k: int) -> dict[str, float]:
    return {"value": gini_coefficient(rec_freq(b, k))}
