"""IPS weights (spec §13.3)."""

from __future__ import annotations

import math

import numpy as np
import torch

from cp4285_proj.recommenders.ips import ips_weights


def test_weights_are_clipped_relative_to_the_smallest_then_mean_normalised():
    inverse_propensity = torch.tensor([200.0, 20.0, 2.0, 2.0])  # items exposed 1x, 10x, 100x, 100x as often
    w, diag = ips_weights(inverse_propensity, clip=30.0)

    expected = np.array([30.0, 10.0, 1.0, 1.0])  # 100, 10, 1, 1 after rescaling; 100 clipped to 30
    np.testing.assert_allclose(w.numpy(), expected / expected.mean(), rtol=1e-6)
    assert diag["ips_clipped_share"] == 0.25
    np.testing.assert_allclose(diag["ips_ess_share"], expected.sum() ** 2 / (expected**2).sum() / 4)


def test_unclipped_weights_keep_the_propensity_ratios():
    w, diag = ips_weights(torch.tensor([9110.0, 1.0, 50.0]), clip=math.inf)
    np.testing.assert_allclose((w / w.min()).numpy(), [9110.0, 1.0, 50.0], rtol=1e-5)
    assert diag["ips_clipped_share"] == 0.0
