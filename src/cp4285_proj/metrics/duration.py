"""Duration metrics (spec §7.3): list duration vs relevant candidates, Wasserstein, JSD over duration bins, RSP/REO."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import N_BINS, bin_counts, list_bin_counts, rec_freq, relevance, weighted_mean_median
from cp4285_proj.metrics.popularity import reo_value, rsp_value
from cp4285_proj.metrics.registry import register_metric


def _ref_freq(b: EvalBundle, label: str) -> np.ndarray:
    """Count of relevant candidate rows per item (rows, not distinct items)."""
    return b.memo(("ref_freq", label), lambda bb: np.bincount(bb.cand_item[relevance(bb, label)], minlength=bb.n_items))


def _user_ref_bins(b: EvalBundle, label: str) -> np.ndarray:
    """[U, N_BINS] relevant candidate rows of each user per duration bin."""
    rel = relevance(b, label)
    return bin_counts(b.cand_user_pos[rel], b.items.dur_bin.astype(np.int64)[b.cand_item[rel]], len(b.users))


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


@register_metric("jsd_dur", kind="per_user", params={"k": int}, needs_label=True, needs_relevance=True)
def jsd_dur(b: EvalBundle, *, k: int, label: str) -> np.ndarray:
    """UPD-style (Abdollahpouri et al., UMAP '21): Jensen–Shannon divergence (base 2, in [0, 1]) between the
    user's relevant candidate rows and their top-k list over duration bins. NaN for users with no relevant row."""
    ref = _user_ref_bins(b, label).astype(np.float64)
    lst = list_bin_counts(b, k, b.items.dur_bin.astype(np.int64)).astype(np.float64)
    out = np.full(len(b.users), np.nan)
    ok = ref.sum(1) > 0
    out[ok] = jensenshannon(ref[ok], lst[ok], base=2, axis=1) ** 2
    return out


@register_metric("dur_bin_share", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def dur_bin_share(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    """Pooled share of list slots (`list_j`) and of relevant candidate rows (`ref_j`) in each duration bin j."""

    def per_user(root: EvalBundle) -> np.ndarray:
        return np.concatenate([list_bin_counts(root, k, root.items.dur_bin.astype(np.int64)), _user_ref_bins(root, label)], 1)

    counts = b.user_cache(("dur_bin_share", k, label), per_user).sum(0).astype(np.float64)
    lst, ref = counts[:N_BINS], counts[N_BINS:]
    with np.errstate(invalid="ignore", divide="ignore"):
        lst, ref = lst / lst.sum(), ref / ref.sum()
    return {**{f"list_{j}": float(v) for j, v in enumerate(lst)}, **{f"ref_{j}": float(v) for j, v in enumerate(ref)}}


@register_metric("rsp_dur", kind="global", params={"k": int}, needs_label=False)
def rsp_dur(b: EvalBundle, *, k: int) -> dict[str, float]:
    return {"value": rsp_value(b, k, "dur")}


@register_metric("reo_dur", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def reo_dur(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    return {"value": reo_value(b, k, label, "dur")}
