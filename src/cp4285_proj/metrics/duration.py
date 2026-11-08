"""Duration metrics (spec §7.3): Wasserstein distance, and per-user JSD and pooled shares over duration quintiles,
between list durations and relevant-candidate durations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import rec_freq, relevance, top
from cp4285_proj.metrics.registry import register_metric

N_BINS = 5  # duration quintiles


def _dur_bins(b: EvalBundle) -> np.ndarray:
    """Duration quintile of every pool item: rank by (duration_ms, item_idx), then rank · N_BINS // n_items."""

    def bins(bb: EvalBundle) -> np.ndarray:
        rank = np.empty(bb.n_items, dtype=np.int64)
        rank[np.lexsort((np.arange(bb.n_items), bb.items.duration_ms))] = np.arange(bb.n_items)
        return rank * N_BINS // bb.n_items

    return b.memo("dur_bins", bins)


def _bin_counts(pos: np.ndarray, bins: np.ndarray, n_rows: int) -> np.ndarray:
    """[n_rows, N_BINS] counts of (row, bin) pairs."""
    return np.bincount(pos * N_BINS + bins, minlength=n_rows * N_BINS).reshape(n_rows, N_BINS)


def _user_list_bins(b: EvalBundle, k: int) -> np.ndarray:
    """[U, N_BINS] number of each user's top-k items in each duration bin."""
    t = top(b, k)
    return _bin_counts(np.repeat(np.arange(t.shape[0]), t.shape[1]), _dur_bins(b)[t.ravel()], t.shape[0])


def _user_ref_bins(b: EvalBundle, label: str) -> np.ndarray:
    """[U, N_BINS] relevant candidate rows of each user per duration bin."""
    rel = relevance(b, label)
    return _bin_counts(b.cand_user_pos[rel], _dur_bins(b)[b.cand_item[rel]], len(b.users))


@register_metric("wasserstein_duration", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def wasserstein_duration(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    """W1 between durations (s) of pooled list items and of relevant candidate rows (count-weighted support)."""
    rec = rec_freq(b, k)
    ref = np.bincount(b.cand_item[relevance(b, label)], minlength=b.n_items)  # relevant rows, not distinct items
    if rec.sum() == 0 or ref.sum() == 0:
        return {"value": float("nan")}
    d = b.items.duration_ms / 1000.0
    return {"value": float(wasserstein_distance(d, d, u_weights=rec, v_weights=ref))}


@register_metric("jsd_dur", kind="per_user", params={"k": int}, needs_label=True, needs_relevance=True)
def jsd_dur(b: EvalBundle, *, k: int, label: str) -> np.ndarray:
    """UPD-style (Abdollahpouri et al., UMAP '21): Jensen–Shannon divergence (base 2, in [0, 1]) between the
    user's relevant candidate rows and their top-k list over duration bins. NaN for users with no relevant row."""
    ref = _user_ref_bins(b, label).astype(np.float64)
    lst = _user_list_bins(b, k).astype(np.float64)
    out = np.full(len(b.users), np.nan)
    ok = ref.sum(1) > 0
    out[ok] = jensenshannon(ref[ok], lst[ok], base=2, axis=1) ** 2
    return out


@register_metric("dur_bin_share", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def dur_bin_share(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    """Pooled share of list slots (`list_j`) and of relevant candidate rows (`ref_j`) in each duration bin j."""

    def per_user(root: EvalBundle) -> np.ndarray:
        return np.concatenate([_user_list_bins(root, k), _user_ref_bins(root, label)], 1)

    counts = b.user_cache(("dur_bin_share", k, label), per_user).sum(0).astype(np.float64)
    lst, ref = counts[:N_BINS], counts[N_BINS:]
    with np.errstate(invalid="ignore", divide="ignore"):
        lst, ref = lst / lst.sum(), ref / ref.sum()
    return {**{f"list_{j}": float(v) for j, v in enumerate(lst)}, **{f"ref_{j}": float(v) for j, v in enumerate(ref)}}
