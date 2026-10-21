"""Naive MF (spec §5.3): logit(u,i) = <p_u, q_i> + b_u + b_i + b, pointwise BCE on logged exposures.

On `raw_watch_time` it is the D2Q paper's value regression (VR, spec §13.2): MSE on watch time divided by its
train mean, so the output is in units of the mean watch time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cp4285_proj.recommenders.base import Backbone, Batch, FitContext, FitResult, TorchRecommender, TrainData
from cp4285_proj.recommenders.registry import register_recommender


@dataclass(frozen=True)
class MFModelConfig:
    embedding_dim: int = 64


class MFBackbone(Backbone):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.P = nn.Parameter(torch.randn(n_users, dim) * 0.01)
        self.Q = nn.Parameter(torch.randn(n_items, dim) * 0.01)
        self.b_user = nn.Parameter(torch.zeros(n_users))
        self.b_item = nn.Parameter(torch.zeros(n_items))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return (self.P[user] * self.Q[item]).sum(-1) + self.b_user[user] + self.b_item[item] + self.b

    def score_all(self, user: torch.Tensor) -> torch.Tensor:
        return self.P[user] @ self.Q.T + self.b_user[user, None] + self.b_item[None, :] + self.b


@register_recommender("mf")
class NaiveMF(TorchRecommender):
    model_config_cls = MFModelConfig
    supported_labels = ("click", "long_view", "satisfaction", "watch_ratio", "raw_watch_time")
    _target_scale: float | None = None  # train mean of a regression target; None for BCE targets

    def build_backbone(self, n_users: int, n_items: int) -> Backbone:
        return MFBackbone(n_users, n_items, self.model_cfg.embedding_dim)

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult:
        self._cold_users = torch.bincount(train.user, minlength=train.n_users) == 0
        self._cold_items = torch.bincount(train.item, minlength=train.n_items) == 0
        if not ctx.label.unit_interval:
            self._target_scale = float(train.target.mean())
        return super().fit(train, ctx)

    def batch_loss(self, batch: Batch) -> torch.Tensor:
        if self._target_scale is None:
            return super().batch_loss(batch)
        return nn.functional.mse_loss(self.backbone(batch.user, batch.item), batch.target / self._target_scale)

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """Default Adam(lr, weight_decay), plus: rows with no train data keep their starting values.

        Torch's Adam applies weight decay to every row, so without the post-step restore the coupled
        decay would move embeddings and biases of users/items that never occur in train.
        """
        opt = super().configure_optimizer()
        bb = self.backbone
        frozen = [(bb.P, self._cold_users), (bb.b_user, self._cold_users), (bb.Q, self._cold_items), (bb.b_item, self._cold_items)]
        initial = [(p, idx, p.detach()[idx].clone()) for p, mask in frozen if (idx := mask.nonzero().squeeze(1)).numel()]

        def restore(_opt, _args, _kwargs) -> None:
            with torch.no_grad():
                for p, idx, init in initial:
                    p.index_copy_(0, idx, init)

        if initial:
            opt.register_step_post_hook(restore)
        return opt
