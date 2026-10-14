"""Accuracy within each user's random candidate set: GAUC and NDCG@k (spec §7.3)."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.common import relevance
from cp4285_proj.metrics.registry import register_metric


def _user_starts(pos: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(pos, minlength=n_users)
    return counts, np.cumsum(counts) - counts


def per_user_auc(pos: np.ndarray, score: np.ndarray, rel: np.ndarray, n_users: int) -> np.ndarray:
    """AUC of `score` against boolean `rel` within each user (Mann–Whitney, average ranks for ties).

    NaN for users whose candidates hold a single class.
    """
    order = np.lexsort((score, pos))
    p, s, r = pos[order], score[order], rel[order].astype(np.float64)
    counts, starts = _user_starts(p, n_users)
    rank = np.arange(len(p)) - starts[p] + 1.0
    new_tie = np.ones(len(p), dtype=bool)
    new_tie[1:] = (p[1:] != p[:-1]) | (s[1:] != s[:-1])
    tie = np.cumsum(new_tie) - 1
    avg_rank = (np.bincount(tie, weights=rank) / np.bincount(tie))[tie]
    n_pos = np.bincount(p, weights=r, minlength=n_users)
    n_neg = counts - n_pos
    sum_rank_pos = np.bincount(p, weights=avg_rank * r, minlength=n_users)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def gauc_value(pos: np.ndarray, score: np.ndarray, rel: np.ndarray, n_users: int) -> float:
    """GAUC = Σ_u |C_u|·AUC_u / Σ_u |C_u| over users with a defined AUC (the selection rule, spec §5.1)."""
    auc = per_user_auc(pos, score, rel, n_users)
    weight = np.bincount(pos, minlength=n_users).astype(np.float64)
    ok = ~np.isnan(auc)
    return float((auc[ok] * weight[ok]).sum() / weight[ok].sum())


@register_metric("gauc", kind="per_user", params={}, needs_label=True, needs_relevance=True, weight="candidates")
def gauc(b: EvalBundle, *, label: str) -> np.ndarray:
    return per_user_auc(b.cand_user_pos, b.cand_score, relevance(b, label), len(b.users))


def per_user_ndcg(pos: np.ndarray, score: np.ndarray, gain: np.ndarray, n_users: int, k: int, minmax: bool) -> np.ndarray:
    """NDCG@k ranking each user's candidates by score; linear gain; ideal over the same candidates.

    NaN when every gain of the user is equal (this includes all-zero gains and single-row users).
    """
    gain = gain.astype(np.float64)
    counts, starts = _user_starts(pos, n_users)
    ideal = np.lexsort((-gain, pos))
    g_ideal = gain[ideal]
    nonempty = counts > 0
    g_max = np.full(n_users, np.nan)
    g_min = np.full(n_users, np.nan)
    g_max[nonempty] = g_ideal[starts[nonempty]]
    g_min[nonempty] = g_ideal[starts[nonempty] + counts[nonempty] - 1]
    if minmax:
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = (gain - g_min[pos]) / (g_max - g_min)[pos]
        gain[~np.isfinite(gain)] = 0.0
        g_ideal = gain[ideal]

    def dcg(order: np.ndarray, g_sorted: np.ndarray) -> np.ndarray:
        p = pos[order]
        rank = np.arange(len(order)) - starts[p]
        keep = rank < k
        return np.bincount(p[keep], weights=g_sorted[keep] / np.log2(rank[keep] + 2.0), minlength=n_users)

    ranked = np.lexsort((-score, pos))  # stable: equal scores keep row order
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dcg(ranked, gain[ranked]) / dcg(ideal, g_ideal)
    out[~(g_max > g_min)] = np.nan
    return out


@register_metric("ndcg", kind="per_user", params={"k": int}, needs_label=True)
def ndcg(b: EvalBundle, *, k: int, label: str) -> np.ndarray:
    def compute(root: EvalBundle) -> np.ndarray:
        spec = root.labels[label]
        return per_user_ndcg(
            root.cand_user_pos, root.cand_score, root.cand_labels[label], len(root.users), k, spec.ndcg_gain == "user_minmax"
        )

    return b.user_cache(("ndcg", k, label), compute)
