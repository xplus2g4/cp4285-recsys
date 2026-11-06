"""Vectorised helpers shared by metric modules."""

from __future__ import annotations

import numpy as np

from cp4285_proj.evaluation.bundle import EvalBundle


def top(b: EvalBundle, k: int) -> np.ndarray:
    """[U, k] top-k item_idx."""
    return b.topk[:, :k]


def rec_freq(b: EvalBundle, k: int) -> np.ndarray:
    """Recommendation count of every pool item over all lists (zeros included)."""
    return b.memo(("rec_freq", k), lambda bb: np.bincount(top(bb, k).ravel(), minlength=bb.n_items))


def relevance(b: EvalBundle, label: str) -> np.ndarray:
    return b.labels[label].binary(b.cand_labels[label])
