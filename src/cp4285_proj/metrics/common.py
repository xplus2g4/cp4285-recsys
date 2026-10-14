"""Vectorised helpers shared by metric modules."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle

N_BINS = 5


def top(b: EvalBundle, k: int) -> np.ndarray:
    """[U, k] top-k item_idx."""
    return b.topk[:, :k]


def rec_freq(b: EvalBundle, k: int) -> np.ndarray:
    """Recommendation count of every pool item over all lists (zeros included)."""
    return b.memo(("rec_freq", k), lambda bb: np.bincount(top(bb, k).ravel(), minlength=bb.n_items))


def relevance(b: EvalBundle, label: str) -> np.ndarray:
    return b.labels[label].binary(b.cand_labels[label])


def bin_counts(pos: np.ndarray, bins: np.ndarray, n_rows: int, weights: np.ndarray | None = None) -> np.ndarray:
    """[n_rows, N_BINS] counts of (row, bin) pairs."""
    flat = np.bincount(pos * N_BINS + bins, weights=weights, minlength=n_rows * N_BINS)
    return flat.reshape(n_rows, N_BINS)


def list_bin_counts(b: EvalBundle, k: int, bins: np.ndarray) -> np.ndarray:
    """[U, N_BINS] number of each user's top-k items in each bin."""
    t = top(b, k)
    pos = np.repeat(np.arange(t.shape[0]), t.shape[1])
    return bin_counts(pos, bins[t.ravel()].astype(np.int64), t.shape[0])


def weighted_mean_median(values: np.ndarray, weights: np.ndarray, order: np.ndarray | None = None) -> tuple[float, float]:
    """Mean and median of the multiset where values[i] occurs weights[i] times (matches np.median).

    `order` = argsort(values), when the caller already has it."""
    n = weights.sum()
    if n == 0:
        return float("nan"), float("nan")
    if order is None:
        order = np.argsort(values, kind="stable")
    v, c = values[order], np.cumsum(weights[order])
    lo = v[np.searchsorted(c, (n - 1) // 2, side="right")]
    hi = v[np.searchsorted(c, n // 2, side="right")]
    return float((values * weights).sum() / n), float((lo + hi) / 2)
