"""MACR counterfactual inference (spec §13.5)."""

from __future__ import annotations

import numpy as np
import torch

from cp4285_proj.config import TrainConfig
from cp4285_proj.recommenders.macr import MACR, MACRModelConfig


def test_recommend_and_score_apply_the_same_counterfactual_adjustment():
    torch.manual_seed(0)
    model = MACR(MACRModelConfig(embedding_dim=4, c=3.0), TrainConfig())
    model.backbone = model.build_backbone(n_users=3, n_items=5)
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.normal_()  # distinct users/items so a transposed broadcast would show

    items, scores = model.recommend(np.arange(3), k=5, exclude=None)
    pair_scores = model.score(np.repeat(np.arange(3), 5), items.reshape(-1))
    np.testing.assert_allclose(scores.reshape(-1), pair_scores, rtol=1e-5)

    bb = model.backbone
    u, i = torch.tensor([1]), torch.tensor([2])
    expected = (bb(u, i) - 3.0) * torch.sigmoid(bb.item_logit(i)) * torch.sigmoid(bb.user_logit(u))  # Eq. 9
    np.testing.assert_allclose(model.score(np.array([1]), np.array([2])), expected.detach().numpy(), rtol=1e-5)
