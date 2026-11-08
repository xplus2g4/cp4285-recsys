from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cp4285_proj.data.labels import LABELS
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
    duration_ms=None,
    pop_exposure=None,
) -> EvalBundle:
    """Small hand-built bundle. `users` are 0..U-1; `labels` maps label → [R] values (missing → zeros)."""
    topk = np.asarray(topk)
    n_users = topk.shape[0]
    r = len(cand_item)
    cand_labels = {name: np.zeros(r) for name in LABELS}
    cand_labels.update({k: np.asarray(v, dtype=float) for k, v in (labels or {}).items()})
    return EvalBundle(
        split="calibration",
        users=np.arange(n_users),
        cand_user_pos=np.asarray(cand_user_pos),
        cand_item=np.asarray(cand_item),
        cand_score=np.asarray(cand_score, dtype=float),
        cand_labels=cand_labels,
        labels=dict(LABELS),
        topk=topk,
        items=ItemMeta(
            duration_ms=np.asarray(duration_ms if duration_ms is not None else np.full(n_items, 1000)),
            pop_exposure=np.asarray(pop_exposure if pop_exposure is not None else np.arange(n_items)),
        ),
        n_items=n_items,
    )
