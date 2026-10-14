"""Clickbait share of top-k lists (spec §7.3). `is_clickbait` is defined from calibration rows only."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import top
from cp4285_proj.metrics.registry import register_metric


@register_metric("clickbait_share", kind="per_user", params={"k": int}, needs_label=False)
def clickbait_share(b: EvalBundle, *, k: int) -> np.ndarray:
    return b.items.is_clickbait[top(b, k)].mean(1, dtype=np.float64)
