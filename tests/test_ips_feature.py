"""Feature-based propensity model (spec §13.6)."""

from __future__ import annotations

import itertools

import torch

from cp4285_proj.recommenders.ips_feature import FeatureFM


def test_fm_logit_is_bias_plus_field_weights_plus_all_pairwise_interactions():
    torch.manual_seed(0)
    fm = FeatureFM([3, 2, 4], dim=5)
    with torch.no_grad():
        fm.w.normal_()
        fm.b.fill_(0.7)
    codes = torch.tensor([[2, 0, 3], [0, 1, 1]])

    idx = codes + torch.tensor([0, 3, 5])  # each field's codes live in its own block
    expected = torch.stack([
        fm.b + fm.w[row].sum() + sum((fm.v[row[f]] * fm.v[row[g]]).sum() for f, g in itertools.combinations(range(3), 2))
        for row in idx
    ])
    torch.testing.assert_close(fm(codes), expected)
