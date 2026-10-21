"""MACR (Wei et al., KDD '21) on the MF backbone (spec §13.5).

Three branches: ŷ_k is the MF backbone logit, i.e. the existing recommender used unchanged, as MACR is
model-agnostic. The item module is ŷ_i = w_i·q_i + b_i' and the user module is ŷ_u = w_u·p_u + b_u', both
linear layers on the backbone's embeddings.
Training (Eq. 7–8): ŷ = ŷ_k·σ(ŷ_i)·σ(ŷ_u), L = BCE(ŷ) + α·BCE(ŷ_i) + β·BCE(ŷ_u).
Inference (Eq. 9): rank by (ŷ_k − c)·σ(ŷ_i)·σ(ŷ_u), which removes the natural direct effect of the item.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cp4285_proj.recommenders.base import Backbone, Batch
from cp4285_proj.recommenders.naive_mf import MFBackbone, NaiveMF
from cp4285_proj.recommenders.registry import register_recommender


@dataclass(frozen=True)
class MACRModelConfig:
    embedding_dim: int = 64
    alpha: float = 1e-3  # weight of the item-module loss (the paper's default)
    beta: float = 1e-3  # weight of the user-module loss (the paper's default)
    c: float = 0.0  # reference value of ŷ_k at inference; 0 ranks by the total effect ŷ


class MACRBackbone(MFBackbone):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__(n_users, n_items, dim)
        self.item_module = nn.Linear(dim, 1)
        self.user_module = nn.Linear(dim, 1)

    def item_logit(self, item: torch.Tensor | None) -> torch.Tensor:
        """ŷ_i for the given items, or for every item when `item` is None."""
        return self.item_module(self.Q if item is None else self.Q[item]).squeeze(-1)

    def user_logit(self, user: torch.Tensor) -> torch.Tensor:
        return self.user_module(self.P[user]).squeeze(-1)


@register_recommender("macr")
class MACR(NaiveMF):
    model_config_cls = MACRModelConfig
    supported_labels = ("click", "long_view", "satisfaction", "watch_ratio")

    def build_backbone(self, n_users: int, n_items: int) -> Backbone:
        return MACRBackbone(n_users, n_items, self.model_cfg.embedding_dim)

    def batch_loss(self, batch: Batch) -> torch.Tensor:
        bb, cfg, t = self.backbone, self.model_cfg, batch.target
        y_k, y_i, y_u = bb(batch.user, batch.item), bb.item_logit(batch.item), bb.user_logit(batch.user)
        bce = nn.functional.binary_cross_entropy_with_logits
        return bce(y_k * torch.sigmoid(y_i) * torch.sigmoid(y_u), t) + cfg.alpha * bce(y_i, t) + cfg.beta * bce(y_u, t)

    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        s_i = torch.sigmoid(self.backbone.item_logit(item))
        s_u = torch.sigmoid(self.backbone.user_logit(user))
        if item is None:  # logits [B, n_items] from score_all
            s_i, s_u = s_i[None, :], s_u[:, None]
        return (logits - self.model_cfg.c) * s_i * s_u
