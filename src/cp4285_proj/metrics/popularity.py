"""Popularity metrics (spec §7.3): ARP, APLT, Gini, and ranking-based RSP / REO over item bins.

RSP/REO follow Zhu, Wang & Caverlee (SIGIR '20): std_a(P_a) / mean_a(P_a), population std.
"""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import N_BINS, bin_counts, list_bin_counts, rec_freq, relevance, top
from cp4285_proj.metrics.registry import register_metric


def _bins(b: EvalBundle, group: str) -> np.ndarray:
    return {"pop": b.items.pop_bin, "dur": b.items.dur_bin}[group].astype(np.int64)


def _rel_std(p: np.ndarray) -> float:
    if not np.all(np.isfinite(p)) or p.mean() == 0:
        return float("nan")
    return float(p.std() / p.mean())


def rsp_value(b: EvalBundle, k: int, group: str) -> float:
    """P_a = Σ_u |L_u ∩ G_a| / Σ_u |G_a \\ seen_u|."""
    bins = _bins(b, group)

    def unseen(root: EvalBundle) -> np.ndarray:
        pos, item = root.seen.gather(root.users)
        sizes = np.bincount(bins, minlength=N_BINS)
        return sizes[None, :] - bin_counts(pos, bins[item], len(root.users))

    num = b.user_cache(("rsp_num", k, group), lambda r: list_bin_counts(r, k, bins)).sum(0)
    den = b.user_cache(("rsp_unseen", group), unseen).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _rel_std(num / den)


def reo_value(b: EvalBundle, k: int, label: str, group: str) -> float:
    """P_a = Σ_u |L_u ∩ G_a ∩ Pos_u| / Σ_u |G_a ∩ Pos_u|, Pos_u = distinct relevant items in C_u."""
    bins = _bins(b, group)

    def per_user(root: EvalBundle) -> np.ndarray:
        rel = relevance(root, label)
        key = np.unique(root.cand_user_pos[rel].astype(np.int64) * root.n_items + root.cand_item[rel])
        upos, uitem = key // root.n_items, key % root.n_items
        hit = (top(root, k)[upos] == uitem[:, None]).any(1)
        n_users = len(root.users)
        return np.stack([bin_counts(upos[hit], bins[uitem[hit]], n_users), bin_counts(upos, bins[uitem], n_users)], 1)

    counts = b.user_cache(("reo", k, label, group), per_user).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _rel_std(counts[0] / counts[1])


@register_metric("arp", kind="per_user", params={"k": int}, needs_label=False)
def arp(b: EvalBundle, *, k: int) -> np.ndarray:
    return b.items.pop_exposure[top(b, k)].mean(1, dtype=np.float64)


@register_metric("aplt", kind="per_user", params={"k": int}, needs_label=False)
def aplt(b: EvalBundle, *, k: int) -> np.ndarray:
    return (~b.items.is_head[top(b, k)]).mean(1, dtype=np.float64)


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


@register_metric("rsp_pop", kind="global", params={"k": int}, needs_label=False)
def rsp_pop(b: EvalBundle, *, k: int) -> dict[str, float]:
    return {"value": rsp_value(b, k, "pop")}


@register_metric("reo_pop", kind="global", params={"k": int}, needs_label=True, needs_relevance=True)
def reo_pop(b: EvalBundle, *, k: int, label: str) -> dict[str, float]:
    return {"value": reo_value(b, k, label, "pop")}
