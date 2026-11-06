"""Duration metric (spec §7.3): Wasserstein distance between list durations and relevant-candidate durations."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import rec_freq, relevance
from cp4285_proj.metrics.registry import register_metric


@register_metric("wasserstein_duration", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def wasserstein_duration(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    """W1 between durations (s) of pooled list items and of relevant candidate rows (count-weighted support)."""
    rec = rec_freq(b, k)
    ref = np.bincount(b.cand_item[relevance(b, label)], minlength=b.n_items)  # relevant rows, not distinct items
    if rec.sum() == 0 or ref.sum() == 0:
        return {"value": float("nan")}
    d = b.items.duration_ms / 1000.0
    return {"value": float(wasserstein_distance(d, d, u_weights=rec, v_weights=ref))}
