"""Exposure / coverage metrics (spec §7.3)."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import rec_freq
from cp4285_proj.metrics.registry import register_metric


@register_metric("coverage", kind="global", params={"k": int}, needs_label=False)
def coverage(b: EvalBundle, *, k: int) -> dict[str, float]:
    return {"value": float((rec_freq(b, k) > 0).sum() / b.n_items)}


@register_metric("exposure_entropy", kind="global", params={"k": int}, needs_label=False)
def exposure_entropy(b: EvalBundle, *, k: int) -> dict[str, float]:
    """Shannon entropy of item recommendation frequency, normalised by log(n_items)."""
    f = rec_freq(b, k)
    p = f[f > 0] / f.sum()
    return {"value": float(-(p * np.log(p)).sum() / np.log(b.n_items))}

