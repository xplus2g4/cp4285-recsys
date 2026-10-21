"""D2Q duration groups, quantile labels and inverse CDFs (spec §13.2)."""

from __future__ import annotations

import numpy as np
import torch

from cp4285_proj.recommenders.d2q import D2QBackbone, d2q_tables


def test_groups_keep_equal_durations_together_and_labels_are_within_group_quantiles():
    duration = np.array([10, 10, 10, 20, 20, 20, 20, 30, 30, 30])
    watch = np.array([5, 1, 3, 9, 2, 7, 7, 4, 8, 6], dtype=np.float32)
    t = d2q_tables(duration, watch, item_duration=np.array([5, 10, 25, 99]), n_groups=2)

    # the median cut is 20, so the 20s go together into the upper group
    assert t.n_used == 2
    np.testing.assert_allclose(t.row_quantile[:3], [1.0, 1 / 3, 2 / 3])  # ECDF within {5, 1, 3}
    upper = np.sort(watch[3:])
    np.testing.assert_allclose(t.row_quantile[3:], np.searchsorted(upper, watch[3:], side="right") / 7)
    np.testing.assert_array_equal(t.item_group, [0, 0, 1, 1])  # out-of-range items go to the end groups


def test_predicted_watch_time_inverts_the_group_cdf():
    duration = np.array([10, 10, 10, 20, 20, 20])
    watch = np.array([1, 3, 5, 100, 200, 300], dtype=np.float32)
    t = d2q_tables(duration, watch, item_duration=np.array([10, 20]), n_groups=2)
    bb = D2QBackbone(1, 2, 2, n_groups=2)
    bb.item_group.copy_(torch.as_tensor(t.item_group))
    bb.inverse_cdf.copy_(torch.as_tensor(t.inverse_cdf))

    h = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    group = bb.item_group[:, None].expand(2, 3)
    np.testing.assert_allclose(bb.predicted_watch_time(h, group).numpy(), [[1, 3, 5], [100, 200, 300]], rtol=1e-5)
