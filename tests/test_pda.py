"""PDA local popularity (spec §13.1)."""

from __future__ import annotations

import numpy as np

from cp4285_proj.recommenders.pda import local_log_popularity


def test_rows_use_their_own_stage_and_inference_uses_the_last_stage():
    # 2 stages over t ∈ [0, 99]: item 0 dominates stage 0, item 1 dominates stage 1; item 2 never appears.
    time = np.array([0, 10, 20, 30, 60, 70, 80, 99])
    item = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    row, last = local_log_popularity(time, item, n_items=3, n_stages=2)

    stage0 = np.log(np.array([3 + 1, 1 + 1, 0 + 1]) / (4 + 3) * 3)
    stage1 = np.log(np.array([1 + 1, 3 + 1, 0 + 1]) / (4 + 3) * 3)
    np.testing.assert_allclose(row, np.r_[stage0[[0, 0, 0, 1]], stage1[[1, 1, 1, 0]]], rtol=1e-6)
    np.testing.assert_allclose(last, stage1, rtol=1e-6)  # the latest row (t = 99) is in the last stage
    assert np.isfinite(last[2])  # unseen items get a finite, lowest popularity
