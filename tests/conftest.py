from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp4285_proj.data.labels import LABELS
from cp4285_proj.data.schema import CSRMask
from cp4285_proj.evaluation.bundle import EvalBundle, ItemMeta

RAW_DIR = Path("data/raw/KuaiRand-Pure/data")
needs_raw = pytest.mark.skipif(not RAW_DIR.exists(), reason="KuaiRand-Pure raw data not downloaded")


def make_bundle(
    cand_user_pos,
    cand_item,
    cand_score,
    topk,
    *,
    n_items=10,
    labels=None,
    seen_rows=None,
    pop_bin=None,
    dur_bin=None,
    duration_ms=None,
    pop_exposure=None,
    is_head=None,
    is_clickbait=None,
    hist_pop=None,
    hist_dur=None,
    groups=None,
) -> EvalBundle:
    """Small hand-built bundle. `users` are 0..U-1; `labels` maps label → [R] values (missing → zeros)."""
    topk = np.asarray(topk)
    n_users = topk.shape[0]
    r = len(cand_item)
    cand_labels = {name: np.zeros(r) for name in LABELS}
    cand_labels.update({k: np.asarray(v, dtype=float) for k, v in (labels or {}).items()})
    seen_rows = seen_rows or {}
    rows = np.array([u for u, items in seen_rows.items() for _ in items], dtype=np.int64)
    cols = np.array([i for items in seen_rows.values() for i in items], dtype=np.int64)
    bins = np.arange(n_items) * 5 // n_items
    nan5 = np.full((n_users, 5), np.nan)
    return EvalBundle(
        split="calibration",
        users=np.arange(n_users),
        cand_user_pos=np.asarray(cand_user_pos),
        cand_item=np.asarray(cand_item),
        cand_score=np.asarray(cand_score, dtype=float),
        cand_labels=cand_labels,
        labels=dict(LABELS),
        topk=topk,
        seen=CSRMask.from_pairs(rows, cols, n_users),
        items=ItemMeta(
            duration_ms=np.asarray(duration_ms if duration_ms is not None else np.full(n_items, 1000)),
            pop_exposure=np.asarray(pop_exposure if pop_exposure is not None else np.arange(n_items)),
            pop_bin=np.asarray(pop_bin if pop_bin is not None else bins),
            dur_bin=np.asarray(dur_bin if dur_bin is not None else bins),
            is_head=np.asarray(is_head if is_head is not None else np.zeros(n_items, bool)),
            is_clickbait=np.asarray(is_clickbait if is_clickbait is not None else np.zeros(n_items, bool)),
        ),
        user_groups=groups or {g: np.array(["other"] * n_users) for g in ("activity", "mainstream", "dur_pref")},
        hist_dist={"pop": np.asarray(hist_pop) if hist_pop is not None else nan5, "dur": np.asarray(hist_dur) if hist_dur is not None else nan5},
        n_items=n_items,
    )
