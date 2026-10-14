"""Duration metrics (spec §7.3): list duration vs relevant candidates, Wasserstein, RSP/REO over duration bins."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import rec_freq, relevance, weighted_mean_median
from cp4285_proj.metrics.popularity import reo_value, rsp_value
from cp4285_proj.metrics.registry import register_metric


def _ref_freq(b: EvalBundle, label: str) -> np.ndarray:
    """Count of relevant candidate rows per item (rows, not distinct items)."""
    return b.memo(("ref_freq", label), lambda bb: np.bincount(bb.cand_item[relevance(bb, label)], minlength=bb.n_items))


@register_metric("list_duration", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def list_duration(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    d = b.items.duration_ms.astype(np.float64)
    order = b.items.duration_order
    mean, median = weighted_mean_median(d, rec_freq(b, k), order)
    ref_mean, ref_median = weighted_mean_median(d, _ref_freq(b, label), order)
    return {
        "mean": mean,
        "median": median,
        "ref_mean": ref_mean,
        "ref_median": ref_median,
        "delta_mean": mean - ref_mean,
        "delta_median": median - ref_median,
    }


@register_metric("wasserstein_duration", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def wasserstein_duration(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    """W1 between durations (s) of pooled list items and of relevant candidate rows (count-weighted support)."""
    rec, ref = rec_freq(b, k), _ref_freq(b, label)
    if rec.sum() == 0 or ref.sum() == 0:
        return {"value": float("nan")}
    d = b.items.duration_ms / 1000.0
    return {"value": float(wasserstein_distance(d, d, u_weights=rec, v_weights=ref))}


@register_metric("rsp_dur", kind="global", params={"k": int}, needs_label=False)
def rsp_dur(b: EvalBundle, *, k: int) -> dict[str, float]:
    return {"value": rsp_value(b, k, "dur")}


@register_metric("reo_dur", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def reo_dur(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    return {"value": reo_value(b, k, label, "dur")}
