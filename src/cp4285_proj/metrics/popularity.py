"""Popularity metric (spec §7.3): average recommendation popularity (ARP)."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import top
from cp4285_proj.metrics.registry import register_metric


@register_metric("arp", kind="per_user", params={"k": int}, needs_label=False)
def arp(b: EvalBundle, *, k: int) -> np.ndarray:
    return b.items.pop_exposure[top(b, k)].mean(1, dtype=np.float64)
